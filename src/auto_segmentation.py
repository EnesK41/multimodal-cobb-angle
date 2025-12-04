"""
Spine segmentation from CT images using TotalSegmentator.
"""
import os
import glob
import torch
from totalsegmentator.python_api import totalsegmentator

from config import TOBESEGMENTED_DIR, SEGMENTED_MASKS_DIR, CT_SPINE_LABELS


def check_gpu():
    """Checks GPU availability."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: Running on CPU, this will be slow.")


def process_file(file_path):
    """Segments a single CT file."""
    file_name = os.path.basename(file_path)
    output_name = file_name.replace(".nii.gz", "_mask.nii.gz")
    output_path = os.path.join(SEGMENTED_MASKS_DIR, output_name)

    print(f"\nProcessing: {file_name}")

    if os.path.exists(output_path):
        print("  [Skipped] Mask already exists.")
        return

    try:
        totalsegmentator(
            file_path, output_path, 
            task="total", 
            fast=False, ml=True, 
            roi_subset=CT_SPINE_LABELS, quiet=False
        )
        
        print("  [Success]")

    except Exception as e:
        print(f"  [Error] {e}")
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    check_gpu()
    os.makedirs(SEGMENTED_MASKS_DIR, exist_ok=True)

    ct_files = glob.glob(os.path.join(TOBESEGMENTED_DIR, "CT", "*.nii.gz"))

    print(f"Total files: {len(ct_files)}")

    for f in ct_files:
        process_file(f)