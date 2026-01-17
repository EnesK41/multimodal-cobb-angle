
"""
KAGGLE ARCHIVE - DRR Generation from CT Scans
=============================================
This file is adapted from the original Kaggle notebook.
For local use: see src/preprocessing/drr_generator.py

Original Source: Kaggle Spine Segmentation Dataset
"""

# !pip install segmentation-models-pytorch
# !pip uninstall -y numpy scipy albumentations
# !pip install "numpy<2.0" "scipy<1.13" "albumentations==1.4.0"

import os
import glob
import numpy as np
import nibabel as nib
import cv2
import shutil
import random
from tqdm import tqdm 

 # ==========================================
# 1. SETTINGS (FINAL - STATIC NORMALIZATION)
# ==========================================
ROOT_DIR = "/kaggle/input/spine-segmentation-from-ct-scans/spine_segmentation_nnunet_v2"
SEG_DIR = os.path.join(ROOT_DIR, "segmentations")
VOL_DIR = os.path.join(ROOT_DIR, "volumes")

 # Output directory
OUTPUT_DIR = "/kaggle/working/final_dataset_v1"

TARGET_H = 1024
TARGET_W = 512
HU_THRESHOLD = 150

 # Padding (fixed)
PAD_RATIO_X = 0.35 

 # Required vertebrae for strict control (T1=8 ... S1=26)
TARGET_LABELS = list(range(8, 27))

 # Clean output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

def check_spine_coverage_strict(mask_vol):
    """
    Only accepts 'complete' spines.
    Incomplete data is excluded from the dataset.
    """
    unique_labels = np.unique(mask_vol)
    bones = [int(x) for x in unique_labels if x > 0]
    if len(bones) < 5:
        return False
    # Check for upper thoracic (T1-T5) and lower lumbar/sacral (L4-S1)
    has_upper = any(b in [8, 9, 10, 11, 12] for b in bones)
    has_lower = any(b in [23, 24, 25, 26] for b in bones)
    return has_upper and has_lower

def get_smart_bbox(mask_vol):
    # Calculate bounding box for the spine region
    target_indices = np.where(np.isin(mask_vol, TARGET_LABELS))
    rows, cols, slices = target_indices
    if rows.size == 0:
        return None
    x_min, x_max = np.min(rows), np.max(rows)
    z_min, z_max = np.min(slices), np.max(slices)
    return x_min, x_max, z_min, z_max

def make_standard_size_noisy(img):
    h, w = img.shape
    scale = min(TARGET_H / h, TARGET_W / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    delta_h = TARGET_H - new_h
    delta_w = TARGET_W - new_w
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    # Black padding first
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    # --- Add noise ---
    mask = np.ones((TARGET_H, TARGET_W), dtype=np.uint8)
    mask[top:top+new_h, left:left+new_w] = 0
    
    noise = np.random.normal(30, 10, (TARGET_H, TARGET_W)).astype(np.uint8)
    
    final_img = np.where(mask == 1, noise, padded)
    
    return final_img

# ==========================================
# 5. MAIN PROJECTION FUNCTION
# ==========================================
def process_patient(ct_vol, mask_vol, filename):
    # 1. Find BBox
    bbox = get_smart_bbox(mask_vol)
    if bbox is None: return None
    x_min, x_max, z_min, z_max = bbox
    
    # 2. Z Axis (Strict Top - 0 Padding)
    spine_h = z_max - z_min
    z_start = z_min 
    z_end = min(ct_vol.shape[2], z_max + int(spine_h * 0.05)) 
    
    # 3. X Axis (0.35 Padding)
    spine_w = x_max - x_min
    pad_w_amount = int(spine_w * PAD_RATIO_X)
    
    x_start = max(0, x_min - pad_w_amount)
    x_end = min(ct_vol.shape[0], x_max + pad_w_amount)
    
    # 4. Cutting (on RAW CT)
    temp_ct = ct_vol.copy()
    temp_ct[temp_ct < HU_THRESHOLD] = -1000
    
    crop = temp_ct[x_start:x_end, :, z_start:z_end]
    if crop.size == 0: return None
    
    # 5. Generate DRR
    drr = np.max(crop, axis=1)
    drr = np.rot90(drr, k=1) 
    
    # --- STATIC NORMALIZATION ---
    MIN_VAL = -1000.0  # Air
    MAX_VAL = 3000.0   # Hard Bone / Metal
    
    drr = np.clip(drr, MIN_VAL, MAX_VAL)
    drr = (drr - MIN_VAL) / (MAX_VAL - MIN_VAL) * 255.0
    drr = drr.astype(np.uint8)
    
    # 6. Resize + Noise Padding
    final_img = make_standard_size_noisy(drr)
    
    return final_img

# ==========================================
# 6. BATCH PROCESSING LOOP
# ==========================================
if __name__ == "__main__":
    mask_files = sorted(glob.glob(os.path.join(SEG_DIR, "case_*.nii.gz")) + 
                        glob.glob(os.path.join(SEG_DIR, "case_*.nii")))

    print(f"Starting... Total Candidates: {len(mask_files)}")
    processed_count = 0
    skipped_count = 0

    for mask_path in tqdm(mask_files):
        try:
            msk = nib.as_closest_canonical(nib.load(mask_path)).get_fdata()
            
            if not check_spine_coverage_strict(msk):
                skipped_count += 1
                continue
                
            filename = os.path.basename(mask_path)
            vol_path = os.path.join(VOL_DIR, filename)
            if not os.path.exists(vol_path):
                if filename.endswith(".nii"): vol_path += ".gz"
                elif filename.endswith(".nii.gz"): vol_path = vol_path[:-3]
            if not os.path.exists(vol_path): continue
                
            ct = nib.as_closest_canonical(nib.load(vol_path)).get_fdata()
            
            final_result = process_patient(ct, msk, filename)
            
            if final_result is not None:
                save_name = filename.replace(".nii.gz", ".png").replace(".nii", ".png")
                cv2.imwrite(os.path.join(OUTPUT_DIR, "images", save_name), final_result)
                processed_count += 1
                
        except Exception as e:
            print(f"Error occurred ({mask_path}): {e}")
            continue

    print(f"\nProcessing Completed.")
    print(f"✅ Generated Images: {processed_count}")
    print(f"❌ Skipped (Incomplete Spine): {skipped_count}")

    shutil.make_archive("/kaggle/working/final_dataset", 'zip', OUTPUT_DIR)
    print("Zip file ready: final_dataset.zip")
