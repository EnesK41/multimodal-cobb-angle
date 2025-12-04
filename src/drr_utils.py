"""
DRR (Digitally Reconstructed Radiograph) generation and data augmentation.
Creates 2D radiograph-like images from CT volumes.
"""
import os
import glob
import numpy as np
import nibabel as nib
import scipy.ndimage
import cv2
from tqdm import tqdm

from config import (
    SEGMENTED_MASKS_DIR, TOBESEGMENTED_DIR, 
    AUGMENTED_DIR, AUGMENTATION_COUNT
)

CT_ROOT_DIR = os.path.join(TOBESEGMENTED_DIR, "CT")


def apply_bone_enhancement(ct_data):
    """Removes soft tissue, keeps only bone."""
    cleaned = np.copy(ct_data)
    cleaned[cleaned < 200] = -1000
    return cleaned


def create_augmented_drr(ct_path, mask_path, output_root):
    """Creates augmented DRR images from a single CT-mask pair."""
    filename = os.path.basename(ct_path).replace(".nii.gz", "")
    
    try:
        ct_nii = nib.load(ct_path)
        mask_nii = nib.load(mask_path)
        
        ct_data = ct_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        if ct_data.shape != mask_data.shape:
            print(f"[Skipped] Shape mismatch: {filename}")
            return

        ct_data = apply_bone_enhancement(ct_data)

        for i in range(AUGMENTATION_COUNT):
            angle = np.random.uniform(-10, 10)
            
            ct_rot = scipy.ndimage.rotate(
                ct_data, angle, axes=(0, 2), reshape=False, order=1, cval=-1000
            )
            mask_rot = scipy.ndimage.rotate(
                mask_data, angle, axes=(0, 2), reshape=False, order=0, cval=0
            )

            ct_proj = np.max(ct_rot, axis=1)
            mask_proj = np.max(mask_rot, axis=1)

            ct_proj = np.clip(ct_proj, -1000, 3000)
            
            _min, _max = np.min(ct_proj), np.max(ct_proj)
            if _max > _min:
                ct_proj = ((ct_proj - _min) / (_max - _min) * 255).astype(np.uint8)
            else:
                ct_proj = np.zeros_like(ct_proj, dtype=np.uint8)
            
            mask_proj = (mask_proj > 0).astype(np.uint8) * 255

            ct_proj = np.rot90(ct_proj)
            mask_proj = np.rot90(mask_proj)

            save_name = f"{filename}_aug{i}_rot{int(angle)}"
            
            img_out_dir = os.path.join(output_root, "images")
            msk_out_dir = os.path.join(output_root, "masks")
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(msk_out_dir, exist_ok=True)

            cv2.imwrite(os.path.join(img_out_dir, save_name + ".png"), ct_proj)
            cv2.imwrite(os.path.join(msk_out_dir, save_name + ".png"), mask_proj)

    except Exception as e:
        print(f"[Error] {filename}: {e}")


if __name__ == "__main__":
    ct_files = glob.glob(os.path.join(CT_ROOT_DIR, "*.nii.gz"))
    
    print(f"CT folder: {CT_ROOT_DIR}")
    print(f"Mask folder: {SEGMENTED_MASKS_DIR}")
    print(f"Files found: {len(ct_files)}")

    if len(ct_files) == 0:
        print("Error: No CT files found!")
    else:
        for ct_path in tqdm(ct_files):
            base_name = os.path.basename(ct_path)
            mask_name = base_name.replace(".nii.gz", "_mask.nii.gz")
            mask_path = os.path.join(SEGMENTED_MASKS_DIR, mask_name)
            
            if os.path.exists(mask_path):
                create_augmented_drr(ct_path, mask_path, AUGMENTED_DIR)
            else:
                print(f"[Warning] Mask not found: {mask_name}")

    print(f"\nCompleted. Output: {AUGMENTED_DIR}")