import os
import sys
import glob
import shutil
import torch
import nibabel as nib
import nibabel.processing
import numpy as np
from totalsegmentator.python_api import totalsegmentator

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_ROOT = os.path.join(BASE_DIR, "data", "tobesegmented") 
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "segmented_masks") 
TEMP_FOLDER = os.path.join(BASE_DIR, "data", "temp_processing") 

TARGET_SPACING = (1.5, 1.5, 1.5) 

CT_SPINE_SUBSET = [
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", 
    "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "sacrum"
]

def check_gpu_status():
    print("\n" + "="*30)
    print("   HARDWARE CHECK")
    print("="*30)
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU NOT FOUND! Running on CPU (Slow).")
    print("="*30 + "\n")

def resample_image(input_path, output_path, target_spacing=TARGET_SPACING):
    """Resamples the image to standard resolution."""
    print(f"      [INFO] Resampling to {target_spacing}mm...")
    try:
        img = nib.load(input_path)
        resampled_img = nibabel.processing.resample_to_output(img, target_spacing)
        nib.save(resampled_img, output_path)
        return True
    except Exception as e:
        print(f"      [ERROR] Resampling failed: {e}")
        return False

def run_segmentation_api(input_path, output_folder_path, modality):
    """Runs TotalSegmentator API."""
    print(f"      [API] Starting TotalSegmentator ({modality})...")
    try:
        if modality == "MR":
            totalsegmentator(input_path, output_folder_path, task="vertebrae_mr", fast=False, quiet=False)
        elif modality == "CT":
            totalsegmentator(input_path, output_folder_path, task="total", fast=True, roi_subset=CT_SPINE_SUBSET, quiet=False)
        return True
    except Exception as e:
        print(f"      [API ERROR] Execution failed: {e}")
        return False

def cleanup_output_folder(folder_path):
    """
    Recursively deletes the output folder if segmentation fails.
    """
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"      [CLEANUP] Deleted failed output folder: {os.path.basename(folder_path)}")
        except OSError as e:
            print(f"      [ERROR] Could not delete folder: {e}")

def process_file(file_path, modality):
    file_name = os.path.basename(file_path)
    
    # Output directory naming convention
    folder_name = file_name.replace(".nii.gz", "_mask.nii.gz")
    output_folder_path = os.path.join(OUTPUT_ROOT, folder_name)

    print(f"\n--> Processing ({modality}): {file_name}")

    # Check if valid output folder exists
    if os.path.exists(output_folder_path):
        if len(os.listdir(output_folder_path)) > 0:
            print("      [SKIP] Valid output folder exists.")
            return True
        else:
            print("      [RETRY] Empty output folder found. Cleaning up...")
            cleanup_output_folder(output_folder_path)

    os.makedirs(output_folder_path, exist_ok=True)

    # Resampling
    temp_input_path = os.path.join(TEMP_FOLDER, "temp_" + file_name)
    if not resample_image(file_path, temp_input_path): 
        cleanup_output_folder(output_folder_path) 
        return False

    # Run Segmentation
    success = run_segmentation_api(temp_input_path, output_folder_path, modality)

    # Cleanup Temp Input
    if os.path.exists(temp_input_path): os.remove(temp_input_path)

    # Verification
    if success and os.path.exists(output_folder_path) and len(os.listdir(output_folder_path)) > 0:
        print("      [SUCCESS] Segmentation completed.")
        return True
    else:
        print("      [FAIL] Segmentation failed or folder empty.")
        cleanup_output_folder(output_folder_path)
        return False

if __name__ == "__main__":
    check_gpu_status()

    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    if not os.path.exists(TEMP_FOLDER): os.makedirs(TEMP_FOLDER)

    mr_files = glob.glob(os.path.join(INPUT_ROOT, "MR", "*.nii.gz"))
    ct_files = glob.glob(os.path.join(INPUT_ROOT, "CT", "*.nii.gz"))

    print(f"=== PIPELINE STARTED ===")
    print(f"MR Files: {len(mr_files)} | CT Files: {len(ct_files)}")

    for f in mr_files: 
        try: process_file(f, modality="MR")
        except Exception as e: print(f"      [CRASH] {e}")

    for f in ct_files: 
        try: process_file(f, modality="CT")
        except Exception as e: print(f"      [CRASH] {e}")

    try:
        if os.path.exists(TEMP_FOLDER): os.rmdir(TEMP_FOLDER)
    except: pass
    
    print("\n=== PIPELINE FINISHED ===")