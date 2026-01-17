"""
KAGGLE ARCHIVE - Data Augmentation & Dataset Merging
===================================================
This file is adapted from the original Kaggle notebook.
For local use: see src/preprocessing/augmentation.py

Combines Hospital and TotalSegmentator datasets and applies balanced augmentation.
"""

import os
import cv2
import numpy as np
import albumentations as A
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. FILE PATHS
# ==========================================
HOSPITAL_IMG_DIR = "/kaggle/input/inputs/final/final/trainCT"
HOSPITAL_MSK_DIR = "/kaggle/input/inputs/final/final/masks"

TOTAL_IMG_DIR = "/kaggle/input/inputs/totalsegmentator_strict_dataset/trainCT"
TOTAL_MSK_DIR = "/kaggle/input/inputs/totalsegmentator_strict_dataset/masks"

OUTPUT_DIR = "/kaggle/working/Final_Training_Set"
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(OUTPUT_DIR, "masks")

if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MSK_DIR, exist_ok=True)

 # ==========================================
# 2. COMBO AUGMENTATION PIPELINE (x3 generation)
# ==========================================
def get_combo_pipelines():
    """
    Combines techniques to generate fewer but richer samples.
    Goal: Increase TotalSegmentator set from 300 to 1200 (without excessive duplication).
    """
    
    # COMBO 1: Geometry + Lighting (Position and Dose Variation)
    aug_pos_light = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, border_mode=0, value=0, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)
    ])
    
    # COMBO 2: Noise + Blur + Gamma (Old Device / Poor Quality)
    aug_quality = A.Compose([
        A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.8)
    ])
    
    # COMBO 3: Dropout + Shift (Difficult/Challenging Image)
    aug_hard = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0, rotate_limit=0, border_mode=0, value=0, p=1.0),
        A.CoarseDropout(max_holes=4, max_height=25, max_width=25, min_holes=1, fill_value=0, p=1.0)
    ])
    
    return {
        "combo1_light": aug_pos_light,
        "combo2_quality": aug_quality,
        "combo3_hard": aug_hard
    }

 # ==========================================
# 3. PROCESSING STEPS
# ==========================================
if __name__ == "__main__":
    # --- STEP A: COPY HOSPITAL DATA (~560 samples) ---
    print("Step 1: Copying hospital data...")
    hosp_files = [f for f in os.listdir(HOSPITAL_IMG_DIR) if f.endswith('.png')]

    for f in tqdm(hosp_files, desc="Hospital"):
        shutil.copy(os.path.join(HOSPITAL_IMG_DIR, f), os.path.join(OUT_IMG_DIR, f))
        shutil.copy(os.path.join(HOSPITAL_MSK_DIR, f), os.path.join(OUT_MSK_DIR, f))

    print(f"Hospital Data: {len(hosp_files)} samples.")

    # --- STEP B: AUGMENT TOTALSEGMENTATOR DATA (~1200 samples) ---
    print("\nStep 2: Applying 'Combo' Augmentation to TotalSegmentator data (x3)...")
    total_files = [f for f in os.listdir(TOTAL_IMG_DIR) if f.endswith('.png')]
    pipelines_dict = get_combo_pipelines()

    count_total = 0
    for f in tqdm(total_files, desc="TotalSegmentator"):
        img_path = os.path.join(TOTAL_IMG_DIR, f)
        msk_path = os.path.join(TOTAL_MSK_DIR, f)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        # 1. ORIGINAL
        base_name = f"totalsegmentator_orig_{f}"
        cv2.imwrite(os.path.join(OUT_IMG_DIR, base_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT_MSK_DIR, base_name), mask)
        count_total += 1

        # 2. APPLY COMBO AUGMENTATIONS (3 types)
        for aug_name_key, pipeline in pipelines_dict.items():
            transformed = pipeline(image=image, mask=mask)

            save_name = f"totalsegmentator_aug_{aug_name_key}_{f}"
            cv2.imwrite(os.path.join(OUT_IMG_DIR, save_name), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_MSK_DIR, save_name), transformed['mask'])
            count_total += 1

    print(f"TotalSegmentator processing completed. Total Count: {count_total}")

    # ==========================================
    # 4. FINAL REPORT
    # ==========================================
    total_imgs = len(os.listdir(OUT_IMG_DIR))
    ratio = count_total / len(hosp_files)

    print("\n" + "="*40)
    print(f"FINAL DATASET REPORT")
    print(f"Hospital Data : {len(hosp_files)}")
    print(f"TotalSegmentator Data    : {count_total}")
    print(f"Total         : {total_imgs}")
    print(f"Balance Ratio : {ratio:.1f} TotalSegmentator per 1 Hospital sample")
    print("="*40)
    print(f"Output Folder: {OUTPUT_DIR}")
