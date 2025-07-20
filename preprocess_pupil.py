from turtle import width
import cv2
import numpy as np
import os
import argparse

def process_eye_image(image_path, output_dir='processed_images'):
    """
    Detects, isolates, and crops a perfectly centered pupil from an eye image.
    This version includes fixes for data type overflows and applies a circular
    mask to isolate the pupil visually.
    """
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    img_original = img_bgr.copy()
    height, width, _ = img_original.shape

    # 2. Preprocessing for Pupil Detection
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_blur = cv2.medianBlur(img_clahe, 15)
    

    # 3. Pupil Detection with HoughCircles
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=int(height / 2), param1=100, param2=25,
                               minRadius=int(width * 0.01), maxRadius=int(width * 0.25))
    if circles is None:
        print(f"⚠️ Warning: No pupil detected in {image_path}. Skipping.")
        return None, None

    # FIX: Convert to signed 32-bit integers to prevent overflow errors
    circles = np.int32(np.around(circles))

    # 4. Select the Most Central Circle
    img_center_xy = (width // 2, height // 2)
    best_circle = min(circles[0, :], key=lambda c: np.sqrt((c[0] - img_center_xy[0])**2 + (c[1] - img_center_xy[1])**2))
    x, y, r = best_circle

    # 5. Crop a Centered Square Region
    crop_factor = 1.8
    crop_radius = int(r * crop_factor)
    y1 = max(0, y - crop_radius)
    y2 = min(height, y + crop_radius)
    x1 = max(0, x - crop_radius)
    x2 = min(width, x + crop_radius)
    cropped_pupil = img_original[y1:y2, x1:x2]

    # 6. Create and Apply a Mask for the PUPIL ONLY
    mask = np.zeros(cropped_pupil.shape[:2], dtype="uint8")
    h, w = cropped_pupil.shape[:2]

    # The key change: The circle's radius is now the pupil's actual radius 'r'.
    # This ensures only the pupil is visible.
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)

    # Apply the mask to black-out everything except the pupil
    final_crop = cv2.bitwise_and(cropped_pupil, cropped_pupil, mask=mask)

    # 7. Resize to Final Dimensions
    processed_image = cv2.resize(final_crop, (128, 128), interpolation=cv2.INTER_AREA)
    # 8. Create Debug Image (code for this remains the same)
    debug_image = img_original.copy()
    cv2.circle(debug_image, (x, y), r, (0, 255, 0), 3)
    cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return processed_image, debug_image

def main():
    """
    Main function to parse command-line arguments and process images.
    """
    parser = argparse.ArgumentParser(description='Preprocess eye images to isolate and crop the pupil.')
    parser.add_argument('input_files', nargs='+', help='Path(s) to one or more input eye image(s).')
    parser.add_argument('--output_dir', type=str, default='processed_images', help='Directory to save the output files.')
    args = parser.parse_args()

    for image_file in args.input_files:
        if not os.path.exists(image_file):
            print(f"Error: Input file not found at {image_file}")
            continue

        print(f"\nProcessing {image_file}...")
        processed_img, debug_img = process_eye_image(image_file, args.output_dir)

        if processed_img is not None and debug_img is not  None:
            # Construct output filenames
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            processed_filename = os.path.join(args.output_dir, f"{base_name}_pupil_128x128.png")
            debug_filename = os.path.join(args.output_dir, f"{base_name}_debug.png")

            # Save the images
            cv2.imwrite(processed_filename, processed_img)
            cv2.imwrite(debug_filename, debug_img)
            print(f"✅ Success! Images saved to '{args.output_dir}' directory:")
            print(f"   - Cropped Pupil: {processed_filename}")
            print(f"   - Debug Image: {debug_filename}")

if __name__ == '__main__':
    # To run this script:
    # 1. Save it as a Python file (e.g., `preprocess_pupil.py`).
    # 2. Place your eye images in the same directory.
    # 3. Open your terminal or command prompt.
    # 4. Run the script, passing the image filenames as arguments:
    #    python preprocess_pupil.py test_eye.jpg test2_eye.jpg test3_eye.jpg test4_eye.jpg test5_eye.jpg
    main()