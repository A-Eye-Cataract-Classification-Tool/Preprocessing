import cv2
import numpy as np
import os
import argparse
import torch
import albumentations as A
from segment_anything import sam_model_registry, SamPredictor

# =============================================================================
# 1. PUPIL DETECTION
# =============================================================================

def _calculate_confidence(pupil, iris, gray_img_inpainted):
    """
    Scores a given pupil detection based on size, location, and darkness.
    This helps filter out bad detections.
    """
    if pupil is None or iris is None:
        return -1

    px, py, pr = pupil
    ix, iy, ir = iris

    if ir == 0 or pr == 0: return -1

    # Score based on pupil-to-iris size ratio
    size_ratio = pr / ir
    size_score = 1.0 if 0.15 < size_ratio < 0.6 else 0

    # Score based on how centered the pupil is within the iris
    dist = np.sqrt((px - ix)**2 + (py - iy)**2)
    location_score = max(0, 1.0 - (dist / (ir * 0.5)))

    # Score based on the darkness of the pupil area
    mask = np.zeros(gray_img_inpainted.shape, dtype=np.uint8)
    cv2.circle(mask, (px, py), pr, 255, -1)
    intensity_mean = cv2.mean(gray_img_inpainted, mask=mask)[0]
    darkness_score = 1.0 - (intensity_mean / 255.0)

    # Return a final weighted score
    return (2.0 * size_score) + (1.5 * location_score) + (1.0 * darkness_score)

def get_pupil_from_sam(predictor, image, prompt_point):
    """Uses SAM to segment the pupil from a given point."""
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([prompt_point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    if masks is None: return None
    
    best_mask = masks[np.argmax(scores)]
    contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    pupil_contour = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(pupil_contour)
    return (int(x), int(y), int(r))

def detect_pupil_robust(image_bgr, predictor):
    """
    Uses a multi-method approach to robustly find the pupil.
    Returns pupil (x, y, r) on success, otherwise None.
    """
    height, width, _ = image_bgr.shape
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Inpaint bright glare spots to avoid confusing detectors
    _, bright_mask = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    inpaint_mask = cv2.dilate(bright_mask, None, iterations=2)
    gray_inpainted = cv2.inpaint(img_gray, inpaint_mask, 3, cv2.INPAINT_NS)

    pupil_candidate_A, pupil_candidate_B, iris_candidate = None, None, None

    # Method A: Contour Detection
    blur_A = cv2.medianBlur(gray_inpainted, 21)
    thresh_A = cv2.adaptiveThreshold(blur_A, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 7)
    contours_A, _ = cv2.findContours(thresh_A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_A:
        for c in sorted(contours_A, key=cv2.contourArea, reverse=True):
            p, a = cv2.arcLength(c, True), cv2.contourArea(c)
            if p > 0 and a > 100 and (4 * np.pi * a) / (p**2) > 0.6:
                (x, y), r = cv2.minEnclosingCircle(c)
                pupil_candidate_A = (int(x), int(y), int(r))
                break

    # Method B: HoughCircles for Iris -> SAM for Pupil
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    blur_B = cv2.medianBlur(img_clahe, 15)
    circles = cv2.HoughCircles(blur_B, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(height*0.4), param1=50, param2=45, minRadius=int(width*0.1), maxRadius=int(width*0.45))
    if circles is not None:
        best_iris = min(circles[0, :], key=lambda c: np.sqrt((c[0] - width//2)**2 + (c[1] - height//2)**2))
        ix, iy, ir = np.int32(np.around(best_iris))
        iris_candidate = (ix, iy, ir)
        
        # Find darkest point in iris ROI to prompt SAM
        x1, y1 = max(0, ix - ir), max(0, iy - ir)
        iris_roi = gray_inpainted[y1:y1 + 2*ir, x1:x1 + 2*ir]
        if iris_roi.size > 100:
            _, _, min_loc, _ = cv2.minMaxLoc(cv2.medianBlur(iris_roi, 25))
            prompt_point = (min_loc[0] + x1, min_loc[1] + y1)
            pupil_candidate_B = get_pupil_from_sam(predictor, image_bgr, prompt_point)

    # Score and select the best candidate
    iris_for_scoring = iris_candidate if iris_candidate else (pupil_candidate_A[0], pupil_candidate_A[1], int(pupil_candidate_A[2] * 2.8)) if pupil_candidate_A else None
    score_A = _calculate_confidence(pupil_candidate_A, iris_for_scoring, gray_inpainted)
    score_B = _calculate_confidence(pupil_candidate_B, iris_for_scoring, gray_inpainted)

    if score_A > 0.7 and score_A > score_B:
        print("✅ Pupil detected using Method A (Contours).")
        return pupil_candidate_A
    elif score_B > 0.7 and score_B >= score_A:
        print("✅ Pupil detected using Method B (Hough + SAM).")
        return pupil_candidate_B
    else:
        print(f"⚠️ Warning: No pupil candidate passed confidence check. Scores: A={score_A:.2f}, B={score_B:.2f}")
        return None

# =============================================================================
# 2. PREPROCESSING & AUGMENTATION
# =============================================================================

def preprocess_and_crop(image_bgr, pupil_coords):
    """
    Creates a perfectly circular, transparent-background crop of the pupil,
    applies CLAHE, and resizes to 128x128.
    """
    px, py, pr = pupil_coords
    height, width, _ = image_bgr.shape

    # Create a tight square crop box around the pupil
    x1 = max(0, px - pr)
    y1 = max(0, py - pr)
    x2 = min(width, px + pr)
    y2 = min(height, py + pr)
    
    cropped_square = image_bgr[y1:y2, x1:x2]
    if cropped_square.size == 0: return None

    # Create a circular mask for transparency
    mask = np.zeros(cropped_square.shape[:2], dtype="uint8")
    px_local, py_local = px - x1, py - y1
    cv2.circle(mask, (px_local, py_local), pr, 255, -1)
    
    # Apply mask to create a transparent background
    cropped_bgra = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2BGRA)
    cropped_bgra[:, :, 3] = mask
    
    # Resize to final dimensions
    resized_image = cv2.resize(cropped_bgra, (128, 128), interpolation=cv2.INTER_AREA)

    # Apply CLAHE enhancement to the final BGR channels
    resized_bgr = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Re-apply the alpha channel
    final_bgra = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2BGRA)
    final_bgra[:, :, 3] = resized_image[:, :, 3]

    return final_bgra

def get_semantic_augmentations():
    """Defines the BRSDA augmentation pipeline."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=20, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    ])

# =============================================================================
# 3. MAIN WORKFLOW
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Robust A-EYE Preprocessing Pipeline with SAM and BRSDA.')
    parser.add_argument('input_file', type=str, help='Path to the input eye image.')
    parser.add_argument('--output_dir', type=str, default='a_eye_output', help='Directory to save the output files.')
    parser.add_argument('--augment', action='store_true', help='Apply BRSDA augmentations.')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_b_01ec64.pth', help='Path to the SAM checkpoint file.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load SAM model
    if not os.path.exists(args.sam_checkpoint):
        print(f"FATAL ERROR: SAM checkpoint file not found at '{args.sam_checkpoint}'")
        return

    print("Loading SAM model...")
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")

    image_bgr = cv2.imread(args.input_file)
    if image_bgr is None:
        print(f"Error: Could not read image from {args.input_file}")
        return

    print(f"\nProcessing {args.input_file}...")
    pupil_coords = detect_pupil_robust(image_bgr, predictor)

    if pupil_coords:
        processed_image = preprocess_and_crop(image_bgr, pupil_coords)
        if processed_image is None:
            print("❌ Error: Cropping failed after detection.")
            return

        # Save preprocessed image (as PNG to preserve transparency)
        proc_base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        proc_filename = os.path.join(args.output_dir, f"{proc_base_name}_preprocessed.png")
        cv2.imwrite(proc_filename, processed_image)
        print(f"✅ Saved preprocessed image to: {proc_filename}")

        if args.augment:
            print("Applying BRSDA augmentations...")
            # Note: Augmentations work on BGR, so we convert, augment, then re-apply alpha
            bgr_to_augment = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)
            augmented_bgr = get_semantic_augmentations()(image=bgr_to_augment)['image']
            
            # Create final augmented BGRA image
            augmented_image = cv2.cvtColor(augmented_bgr, cv2.COLOR_BGR2BGRA)
            augmented_image[:, :, 3] = processed_image[:, :, 3] # Keep original mask

            aug_filename = os.path.join(args.output_dir, f"{proc_base_name}_augmented.png")
            cv2.imwrite(aug_filename, augmented_image)
            print(f"✅ Saved augmented image to: {aug_filename}")
    else:
        print(f"❌ Could not process image {args.input_file} due to low confidence pupil detection.")

if __name__ == '__main__':
    main()
