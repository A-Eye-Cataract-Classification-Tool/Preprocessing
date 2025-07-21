import cv2
import numpy as np
import os
import argparse
import torch
from segment_anything import sam_model_registry, SamPredictor

def _calculate_confidence(pupil, iris, gray_img_inpainted):
    """
    Helper function to score a given pupil detection based on a
    combination of metrics.
    """
    if pupil is None or iris is None:
        return -1

    px, py, pr = pupil
    ix, iy, ir = iris

    if ir == 0 or pr == 0: return -1

    # 1. Size Score: Pupil radius should be a reasonable fraction of iris radius
    size_ratio = pr / ir
    size_score = 1.0 if 0.15 < size_ratio < 0.6 else 0

    # 2. Location Score: Pupil should be very close to the iris center
    dist = np.sqrt((px - ix)**2 + (py - iy)**2)
    location_score = max(0, 1.0 - (dist / (ir * 0.5)))

    # 3. Darkness Score: Pupil area should be dark
    mask = np.zeros(gray_img_inpainted.shape, dtype=np.uint8)
    cv2.circle(mask, (px, py), pr, 255, -1)
    intensity_mean = cv2.mean(gray_img_inpainted, mask=mask)[0]
    darkness_score = 1.0 - (intensity_mean / 255.0)

    # Final weighted score
    return (2.0 * size_score) + (1.5 * location_score) + (1.0 * darkness_score)

def get_pupil_from_sam(predictor, image, prompt_point):
    """Uses the SAM predictor to get a pupil mask from a prompt point."""
    predictor.set_image(image)
    input_point = np.array([prompt_point])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    if masks is None: return None
    best_mask = masks[np.argmax(scores)]
    contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    pupil_contour = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(pupil_contour)
    return (int(x), int(y), int(r))

def process_eye_image(image_path, predictor, output_dir='processed_images'):
    """
    Final automatic version. Detects the pupil and creates a zoomed-in,
    masked crop of the detected region.
    """
    # --- Steps 1-4: Detection (remain the same) ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None, None
    img_original = img_bgr.copy()
    height, width, _ = img_original.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, bright_mask = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    inpaint_mask = cv2.dilate(bright_mask, None, iterations=2)
    gray_inpainted = cv2.inpaint(img_gray, inpaint_mask, 3, cv2.INPAINT_NS)

    pupil_candidate_A, pupil_candidate_B, iris_candidate = None, None, None

    blur_A = cv2.medianBlur(gray_inpainted, 21)
    thresh_A = cv2.adaptiveThreshold(blur_A, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 7)
    contours_A, _ = cv2.findContours(thresh_A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_A:
        contours_A = sorted(contours_A, key=cv2.contourArea, reverse=True)
        for c in contours_A:
            p, a = cv2.arcLength(c, True), cv2.contourArea(c)
            if p == 0 or a < 100: continue
            if (4 * np.pi * a) / (p**2) > 0.6:
                (x, y), r = cv2.minEnclosingCircle(c)
                pupil_candidate_A = (int(x), int(y), int(r))
                break

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    blur_B = cv2.medianBlur(img_clahe, 15)
    circles = cv2.HoughCircles(blur_B, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(height*0.4), param1=50, param2=45, minRadius=int(width*0.1), maxRadius=int(width*0.45))
    if circles is not None:
        circles = np.int32(np.around(circles))
        best_iris = min(circles[0, :], key=lambda c: np.sqrt((c[0] - width//2)**2 + (c[1] - height//2)**2))
        ix, iy, ir = best_iris
        mask_val = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.circle(mask_val, (ix, iy), ir, 255, -1)
        if cv2.mean(img_gray, mask=mask_val)[0] < 180:
            iris_candidate = (ix, iy, ir)
            x1, y1 = max(0, ix - ir), max(0, iy - ir)
            iris_roi = gray_inpainted[y1:y1 + 2*ir, x1:x1 + 2*ir]
            if iris_roi.size > 100:
                _, _, min_loc, _ = cv2.minMaxLoc(cv2.medianBlur(iris_roi, 25))
                prompt_point = (min_loc[0] + x1, min_loc[1] + y1)
                pupil_candidate_B = get_pupil_from_sam(predictor, img_original, prompt_point)

    iris_for_scoring = iris_candidate if iris_candidate else (pupil_candidate_A[0], pupil_candidate_A[1], int(pupil_candidate_A[2] * 2.8)) if pupil_candidate_A else None
    score_A = _calculate_confidence(pupil_candidate_A, iris_for_scoring, gray_inpainted)
    score_B = _calculate_confidence(pupil_candidate_B, iris_for_scoring, gray_inpainted)

    final_pupil, final_iris = (None, None)
    if score_A > score_B and score_A > 0.7:
        final_pupil, final_iris = pupil_candidate_A, iris_for_scoring
    elif score_B >= score_A and score_B > 0.7:
        final_pupil, final_iris = pupil_candidate_B, iris_candidate
    else:
        print(f"⚠️ Warning: No candidate passed confidence check for {image_path}.")
        return None, None
        
    px, py, pr = final_pupil
    ix, iy, ir = final_iris

    # --- 5. Cropping, Masking, and Resizing (New Logic) ---
    
    # MODIFIED: Decreased multiplier from 3 to 1.8 for a tighter zoom.
    crop_radius = int(pr * 2.0)
    
    # Calculate the coordinates for the square crop box, centered on the pupil.
    x1 = max(0, px - crop_radius)
    y1 = max(0, py - crop_radius)
    x2 = min(width, px + crop_radius)
    y2 = min(height, py + crop_radius)
    
    # Perform the crop.
    cropped_eye = img_original[y1:y2, x1:x2]
    
    # Create a mask for this new, smaller cropped image.
    mask = np.zeros(cropped_eye.shape[:2], dtype="uint8")
    
    # Calculate the pupil's new coordinates within the cropped image.
    px_local = px - x1
    py_local = py - y1
    
    # Draw the pupil circle on the mask.
    cv2.circle(mask, (px_local, py_local), pr, 255, -1)
    
    # Apply the mask to the cropped image.
    masked_crop = cv2.bitwise_and(cropped_eye, cropped_eye, mask=mask)
    
    # Resize the final masked crop to 128x128.
    processed_image = cv2.resize(masked_crop, (128, 128), interpolation=cv2.INTER_AREA)

    # --- 6. Create Debug Image ---
    debug_image = img_original.copy()
    cv2.circle(debug_image, (ix, iy), ir, (255, 100, 0), 3) # Blue Iris
    cv2.circle(debug_image, (px, py), pr, (0, 255, 0), 2)   # Green Pupil
    cv2.circle(debug_image, (px, py), 2, (0, 0, 255), 3)    # Red Center

    return processed_image, debug_image

def main():
    parser = argparse.ArgumentParser(description='Preprocess eye images using SAM.')
    parser.add_argument('input_files', nargs='+', help='Path(s) to input eye image(s).')
    parser.add_argument('--output_dir', type=str, default='processed_images', help='Directory to save output.')
    args = parser.parse_args()

    print("Loading SAM model... (This may take a moment)")
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(sam_checkpoint):
        print(f"FATAL ERROR: SAM checkpoint file not found at '{sam_checkpoint}'")
        return

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")

    os.makedirs(args.output_dir, exist_ok=True)

    for image_file in args.input_files:
        if not os.path.exists(image_file):
            print(f"Error: Input file not found at {image_file}")
            continue

        print(f"\nProcessing {image_file}...")
        processed_img, debug_img = process_eye_image(image_file, predictor, args.output_dir)

        if processed_img is not None and debug_img is not None:
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            p_filename = os.path.join(args.output_dir, f"{base_name}_pupil_128x128.png")
            d_filename = os.path.join(args.output_dir, f"{base_name}_debug.png")
            cv2.imwrite(p_filename, processed_img)
            cv2.imwrite(d_filename, debug_img)
            print(f"✅ Success! Images saved to '{args.output_dir}'")

if __name__ == '__main__':
    main()