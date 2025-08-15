#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone BRSDA (Bayesian Random Semantic Data Augmentation) Utility.

This script implements the core augmentation logic from the BRSDA paper, adapted
for model-free, offline usage on a directory of images. It can also expose a
callable transform for integration into PyTorch Datasets.

The "semantic" mask, originally derived from model attention maps, is simulated
here using spatially-coherent random noise (blurred random maps).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


def generate_spatially_coherent_random_mask(
    height: int,
    width: int,
    bsda_lambda: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulates the "Bayesian Mask" from the BRSDA paper without a model.

    It creates a random noise mask and applies a heavy Gaussian blur to make it
    spatially coherent, mimicking the smooth nature of attention maps.

    Args:
        height: The height of the mask.
        width: The width of the mask.
        bsda_lambda: Controls the smoothness. Ranges from 0.0 (noisy) to 1.0 (very smooth).
        rng: A NumPy random number generator for reproducibility.

    Returns:
        A (H, W, 1) numpy array representing the normalized mask.
    """
    # Create a random noise image
    noise = rng.random((height, width), dtype=np.float32)

    # bsda_lambda controls the blur intensity. Map [0, 1] to a kernel size.
    # We want an odd kernel size. A larger lambda gives a larger kernel -> smoother mask.
    # Max kernel size is capped at roughly half the image width to be effective.
    max_ksize = min(height, width) // 2
    # Ensure it's an odd number
    if max_ksize % 2 == 0:
        max_ksize -= 1
    
    # Clamp lambda to avoid zero or negative ksize
    clamped_lambda = np.clip(bsda_lambda, 0.01, 1.0)
    ksize = int(max_ksize * clamped_lambda)
    if ksize % 2 == 0:
        ksize += 1 # Ensure ksize is odd

    # Apply a strong Gaussian blur to create smooth, regional patterns
    mask = cv2.GaussianBlur(noise, (ksize, ksize), 0)

    # Normalize the mask to be in the [0, 1] range
    min_val, max_val = mask.min(), mask.max()
    if max_val > min_val:
        mask = (mask - min_val) / (max_val - min_val)
    
    # Add channel dimension for broadcasting
    return mask[:, :, np.newaxis]

def brsda_augment(
    img_rgb: np.ndarray,
    bsda_lambda: float = 0.8,
    bsda_multi: int = 10,
    use_original: bool = True,
    alpha: float = 0.5,
    rng: Optional[np.random.Generator] = None
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Applies a model-free version of BRSDA to an image.

    This function generates multiple augmented samples by mixing the original image
    with a randomly shifted version of itself, guided by a simulated Bayesian mask.

    Args:
        img_rgb: The input RGB image as a NumPy array (H, W, 3).
        bsda_lambda: Augmentation strength/smoothness of the mask (0.0 to 1.0).
        bsda_multi: The number of BRSDA variants to generate.
        use_original: If True, includes the original image in the returned list.
        alpha: The mixing factor between the original and the BRSDA-augmented sample.
               0.0 returns the original, 1.0 returns the full BRSDA effect.
        rng: A NumPy random number generator for reproducibility.

    Returns:
        A tuple containing:
        - A list of augmented RGB images (as H, W, 3 uint8 NumPy arrays).
        - The last generated Bayesian mask for debugging purposes (or None).
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w, _ = img_rgb.shape
    augmented_samples = []
    last_mask = None

    if use_original:
        augmented_samples.append(img_rgb.copy())

    # Generate `bsda_multi` augmented versions
    for _ in range(bsda_multi):
        # 1. Create the mixing partner (simulating x_rand from the paper)
        # We use a randomly "rolled" version of the image.
        roll_x = rng.integers(-w // 4, w // 4)
        roll_y = rng.integers(-h // 4, h // 4)
        img_rand = np.roll(img_rgb, shift=(roll_y, roll_x), axis=(0, 1))

        # 2. Generate the simulated Bayesian mask
        bayesian_mask = generate_spatially_coherent_random_mask(h, w, bsda_lambda, rng)
        last_mask = bayesian_mask # Save for debug output

        # 3. Mix the images using the mask
        # Convert to float for multiplication
        img_rgb_float = img_rgb.astype(np.float32)
        img_rand_float = img_rand.astype(np.float32)
        
        img_brsda = img_rgb_float * (1 - bayesian_mask) + img_rand_float * bayesian_mask
        
        # 4. Alpha blend the BRSDA result with the original image
        final_blended = cv2.addWeighted(
            src1=img_rgb_float,
            alpha=1 - alpha,
            src2=img_brsda,
            beta=alpha,
            gamma=0
        )
        
        # Convert back to uint8 and append
        augmented_samples.append(final_blended.clip(0, 255).astype(np.uint8))
        
    return augmented_samples, last_mask


def get_albumentations_transform() -> A.Compose:
    """
    Returns an Albumentations pipeline that adjusts brightness and contrast
    without altering the core colors (hue) of the image.
    """
    return A.Compose([
        # --- Spatial Augmentations (Do not change color) ---
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.06,
            rotate_limit=20,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),

        # --- Brightness & Contrast (Added back in) ---
        # This function adjusts brightness and contrast without shifting the hue.
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),

        # --- Blur Augmentation (Affects sharpness, not color) ---
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    ])


class BRSDAAlbumentations:
    """
    A callable class for inline use in PyTorch Datasets.
    
    Applies BRSDA and then a standard Albumentations pipeline.
    """
    def __init__(
        self,
        bsda_lambda: float = 0.8,
        alpha: float = 0.5,
        rng: Optional[np.random.Generator] = None
    ):
        self.bsda_lambda = bsda_lambda
        self.alpha = alpha
        self.rng = rng if rng is not None else np.random.default_rng()
        self.albumentations_transform = get_albumentations_transform()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: An RGB image as a NumPy array.

        Returns:
            An augmented RGB image as a NumPy array.
        """
        # Get a single BRSDA sample
        # We set bsda_multi=1 and use_original=False to get one random augmentation
        augmented_list, _ = brsda_augment(
            img_rgb=image,
            bsda_lambda=self.bsda_lambda,
            bsda_multi=1,
            use_original=False,
            alpha=self.alpha,
            rng=self.rng
        )
        
        brsda_image = augmented_list[0]
        
        # Apply subsequent Albumentations
        final_image = self.albumentations_transform(image=brsda_image)['image']
        return final_image

def get_brsda_transform(
    bsda_lambda: float = 0.8,
    alpha: float = 0.5,
    seed: Optional[int] = None
) -> BRSDAAlbumentations:
    """
    Factory function to create a callable BRSDA transform object.
    
    Args:
        bsda_lambda: Augmentation strength/smoothness (0.0 to 1.0).
        alpha: Mixing factor for BRSDA effect (0.0 to 1.0).
        seed: Optional random seed for reproducibility.

    Returns:
        A callable BRSDAAlbumentations object.
    """
    rng = np.random.default_rng(seed)
    return BRSDAAlbumentations(bsda_lambda=bsda_lambda, alpha=alpha, rng=rng)


def main():
    parser = argparse.ArgumentParser(description="Standalone BRSDA Augmentation Utility")
    parser.add_argument("input_dir", type=str, help="Path to the directory with input PNG images.")
    parser.add_argument("--output_dir", type=str, default="./augmented", help="Path to save augmented images.")
    parser.add_argument("--n_augments", type=int, default=5, help="Number of augmented versions to generate per image.")
    parser.add_argument("--bsda_lambda", type=float, default=0.8, help="BRSDA strength/smoothness (0.0-1.0).")
    parser.add_argument("--bsda_multi", type=int, default=10, help="Number of BRSDA candidates to sample from.")
    parser.add_argument("--bsda_use_ori", action="store_true", help="If set, one of the outputs may be the original image.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mixing factor between original and BRSDA sample (0.0-1.0).")
    parser.add_argument("--mode", type=str, choices=["offline", "inline"], default="offline", help="Execution mode.")
    parser.add_argument("--debug", action="store_true", help="Save debug visualizations (comparison triptych and mask).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # --- INLINE MODE ---
    if args.mode == 'inline':
        print("✅ BRSDA is ready for inline use.")
        print("Instantiate the transform like this:\n")
        
        usage_snippet = f"""
import numpy as np
from torch.utils.data import Dataset
# Assume this script is named 'augment_with_brsda.py'
from augment_with_brsda import get_brsda_transform

class MyIrisDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load your RGB image as a NumPy array
        # image_rgb = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        image_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8) # Placeholder

        if self.transform:
            image_rgb = self.transform(image=image_rgb)['image']
        
        # ... convert to tensor and return ...
        return image_rgb

# --- Usage ---
# 1. Get the BRSDA transform, which includes Albumentations
brsda_transform_callable = get_brsda_transform(
    bsda_lambda={args.bsda_lambda},
    alpha={args.alpha},
    seed={args.seed}
)

# 2. To use it with Albumentations' Compose, wrap it
# NOTE: The provided get_brsda_transform already includes the Albumentations chain.
# If you wanted to separate them, you'd implement the callable differently.
# For simplicity, we just use our combined transform.
class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, image):
        return {{'image': self.transform(image)}}

# This is a bit redundant as our callable already returns the dict format
# but shows the principle. Our BRSDAAlbumentations is already a complete pipeline.
# So, the easiest way is:

brsda_pipeline = BRSDAAlbumentations(
    bsda_lambda={args.bsda_lambda}, alpha={args.alpha}, rng=np.random.default_rng({args.seed})
)

# Example of getting one augmented image:
dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)
augmented_image = brsda_pipeline(dummy_image) 
print(f"Augmented image shape: {{augmented_image.shape}}")

# Then in your dataset:
# my_dataset = MyIrisDataset(image_paths, transform=brsda_pipeline)
"""
        print(usage_snippet)
        sys.exit(0)

    # --- OFFLINE MODE ---
    rng = np.random.default_rng(args.seed)
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(input_path.glob("*.png")))
    if not image_files:
        print(f"❌ No .png files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(image_files)} images. Starting augmentation...")
    
    albumentations_transform = get_albumentations_transform()

    for image_file in tqdm(image_files, desc="Augmenting Images"):
        # Load image with alpha channel
        img_rgba = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        if img_rgba is None or img_rgba.shape[2] != 4:
            print(f"⚠️ Skipping {image_file.name}, not a valid RGBA image.")
            continue

        img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGB)
        alpha_channel = img_rgba[:, :, 3]
        
        # --- BRSDA Step ---
        # Get a pool of BRSDA-augmented candidates
        brsda_candidates, last_mask = brsda_augment(
            img_rgb,
            bsda_lambda=args.bsda_lambda,
            bsda_multi=args.bsda_multi,
            use_original=args.bsda_use_ori,
            alpha=args.alpha,
            rng=rng
        )

        # --- Save N augmented versions ---
        # Randomly pick from the generated candidates
        choices = rng.choice(len(brsda_candidates), args.n_augments, replace=True)
        
        for i, choice_idx in enumerate(choices):
            chosen_img = brsda_candidates[choice_idx]
            
            # --- Albumentations Step ---
            augmented_img_rgb = albumentations_transform(image=chosen_img)['image']

            # CORRECTED CODE
            # Re-apply original alpha channel
            final_img_rgba = cv2.cvtColor(augmented_img_rgb, cv2.COLOR_RGB2RGBA)
            final_img_rgba[:, :, 3] = alpha_channel

            # CONVERT FINAL IMAGE TO BGRA FOR SAVING WITH OPENCV
            image_to_save = cv2.cvtColor(final_img_rgba, cv2.COLOR_RGBA2BGRA)

            # Save the file
            out_filename = f"{image_file.stem}_aug{i+1}.png"
            cv2.imwrite(str(output_path / out_filename), image_to_save)
     

        # --- Debug Mode Visualizations ---
        if args.debug:
            # 1. BRSDA-only version (alpha=1.0, no Albumentations)
            brsda_only_img, debug_mask = brsda_augment(
                img_rgb, bsda_lambda=args.bsda_lambda, bsda_multi=1, use_original=False, alpha=1.0, rng=rng
            )
            brsda_only_img = brsda_only_img[0]
            
            # 2. Final augmented version for comparison
            # (Re-run one sample to ensure it matches the flow)
            final_aug_sample, _ = brsda_augment(
                img_rgb, bsda_lambda=args.bsda_lambda, bsda_multi=1, use_original=False, alpha=args.alpha, rng=rng
            )
            final_aug_sample_rgb = albumentations_transform(image=final_aug_sample[0])['image']

            # 3. Create triptych: Original | BRSDA-only | Final
            # Ensure all are same size and type
            h, w, _ = img_rgb.shape
            comparison_strip = np.zeros((h, w * 3, 3), dtype=np.uint8)
            comparison_strip[:, :w] = img_rgb
            comparison_strip[:, w:w*2] = brsda_only_img
            comparison_strip[:, w*2:] = final_aug_sample_rgb

            # Save triptych
            debug_filename = f"{image_file.stem}_comparison.png"
            cv2.imwrite(str(output_path / debug_filename), cv2.cvtColor(comparison_strip, cv2.COLOR_RGB2BGR))
            
            # 4. Save the BRSDA mask
            mask_filename = f"{image_file.stem}_bsda_mask.png"
            mask_vis = (debug_mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_path / mask_filename), mask_vis)

    print(f"✅ Augmentation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
