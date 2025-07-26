import cv2
import numpy as np
import os
import argparse
import torch
import albumentations as A
from segment_anything import sam_model_registry, SamPredictor 

"""
A-EYE: Cataract Maturity Classification Pipeline

This script implements the full preprocessing and augmentation pipeline for the A-EYE project,
as detailed in the associated thesis documentation. It takes a raw eye image as input,
processes it to be model-ready, and optionally applies semantic-aware augmentations
for training data.

The workflow is as follows:
1.  **Pupil Detection & Cropping**: An initial computer vision step isolates the
    ocular region to focus the model's analysis on relevant features.
2.  **Preprocessing**: The cropped image is enhanced with Contrast Limited Adaptive
    Histogram Equalization (CLAHE) and resized to the model's input dimensions (128x128).
3.  **Semantic-Aware Augmentation**: For training data, the script applies a series
    of realistic augmentations (the BRSDA methodology) using the Albumentations
    library to simulate clinical variations like glare, blur, and lighting changes.
"""

# =============================================================================
# 1. PUPIL DETECTION AND CROPPING 
# =============================================================================

def detect_and_crop_pupil(image_bgr):
    """
    Detects the pupil using HoughCircles and crops the image to that region.
    This focuses the model's attention on the relevant anatomical area.
    """
    height, width, _ = image_bgr.shape
    
    # Convert to grayscale and apply blur for better circle detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 15)

    # Use HoughCircles to detect the pupil
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=int(height / 4), param1=100, param2=30,
                               minRadius=int(width * 0.05), maxRadius=int(width * 0.25))

    if circles is not None:
        # Convert coordinates to integers
        circles = np.uint16(np.around(circles))
        
        # Select the most central circle as the pupil
        img_center = np.array([width // 2, height // 2])
        best_circle = min(circles[0, :], key=lambda c: np.linalg.norm(np.array(c[:2]) - img_center))
        x, y, r = best_circle
        
        # Define a cropping box with a bit of margin around the pupil
        crop_factor = 2.0
        crop_radius = int(r * crop_factor)
        
        # Calculate crop coordinates, ensuring they are within image bounds
        y1 = max(0, y - crop_radius)
        y2 = min(height, y + crop_radius)
        x1 = max(0, x - crop_radius)
        x2 = min(width, x + crop_radius)
        
        cropped_image = image_bgr[y1:y2, x1:x2]
        
        print("✅ Pupil detected and image cropped.")
        return cropped_image
    else:
        print("⚠️ Warning: No pupil detected. Using the full image.")
        return image_bgr

# =============================================================================
# 2. PREPROCESSING PIPELINE 
# =============================================================================

def preprocess_image(image_bgr):
    # 2.1 Image Cropping
    # First, detect and crop the pupil to focus on the ocular region.
    cropped_image = detect_and_crop_pupil(image_bgr)

    # 2.2 Contrast Enhancement (CLAHE)
    # Applied to the cropped image to improve visibility of lens opacities.
    lab = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 2.3 Image Normalization & Final Resize
    # Scales pixel values and standardizes input dimensions for the model.
    normalized_image = enhanced_image / 255.0
    resized_image = cv2.resize(normalized_image, (128, 128), interpolation=cv2.INTER_AREA)

    # Convert to a format suitable for augmentation (uint8)
    final_preprocessed = (resized_image * 255).astype(np.uint8)
    
    return final_preprocessed

# =============================================================================
# 3. SEMANTIC-AWARE AUGMENTATION 
# =============================================================================

def get_semantic_augmentations():
    """
    Defines the semantic-aware augmentation pipeline using Albumentations.
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1,
                         num_flare_circles_lower=1, num_flare_circles_upper=2,
                         src_radius=100, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])

def apply_augmentations(image_np):
    augmentation_pipeline = get_semantic_augmentations()
    augmented_data = augmentation_pipeline(image=image_np)
    return augmented_data['image']

# =============================================================================
# 4. MAIN PROCESSING WORKFLOW (No change)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='A-EYE: Cataract Maturity Classification Pipeline with Pupil Cropping.')
    parser.add_argument('input_file', type=str, help='Path to the input eye image.')
    parser.add_argument('--output_dir', type=str, default='a_eye_output', help='Directory to save the output files.')
    parser.add_argument('--is_training_set', action='store_true', help='Apply semantic-aware augmentation for the training set.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_bgr = cv2.imread(args.input_file)
    if image_bgr is None:
        print(f"Error: Could not read image from {args.input_file}")
        return

    print(f"Processing {args.input_file}...")
    preprocessed_image = preprocess_image(image_bgr)

    if args.is_training_set:
        print("Applying semantic-aware augmentation for training set...")
        augmented_image = apply_augmentations(preprocessed_image)
        
        aug_base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        aug_filename = os.path.join(args.output_dir, f"{aug_base_name}_augmented.png")
        cv2.imwrite(aug_filename, augmented_image)
        print(f"✅ Saved augmented image to: {aug_filename}")

    proc_base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    proc_filename = os.path.join(args.output_dir, f"{proc_base_name}_preprocessed.png")
    cv2.imwrite(proc_filename, preprocessed_image)
    print(f"✅ Saved preprocessed (and cropped) image to: {proc_filename}")

if __name__ == '__main__':
    main()