import os
import glob
import torch
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_ROOT = os.path.join(BASE_DIR, "data", "tobesegmented")
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "segmented_masks")

# Sadece omurga ile ilgilendiğimiz için bu subset'i koruyoruz (Hız ve odak için)
CT_SPINE_SUBSET = [
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", 
    "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "sacrum"
]

def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU POWER: {torch.cuda.get_device_name(0)} (High-Res Mode ON)")
    else:
        print("⚠️ WARNING: Running on CPU. This will be slow for high-res files.")

def process_file(file_path, modality):
    file_name = os.path.basename(file_path)
    # Output yine tek bir dosya (ml=True sayesinde)
    output_name = file_name.replace(".nii.gz", "_mask.nii.gz")
    output_path = os.path.join(OUTPUT_ROOT, output_name)

    print(f"\n--> Processing ({modality}): {file_name}")

    if os.path.exists(output_path):
        print("      [SKIP] Mask already exists.")
        return

    try:
        print(f"      [API] Segmenting at ORIGINAL resolution...")
        
        # 'roi_subset' kullansak bile, TotalSegmentator çıktı boyutunu 
        # orijinal görüntüyle aynı tutmaya çalışır (crop=None default davranışı).
        if modality == "MR":
            totalsegmentator(file_path, output_path, task="vertebrae_mr", 
                             fast=False, ml=True, quiet=False)
        elif modality == "CT":
            totalsegmentator(file_path, output_path, task="total", 
                             fast=False, ml=True, roi_subset=CT_SPINE_SUBSET, quiet=False)
        
        print("      [SUCCESS] Done.")

    except Exception as e:
        print(f"      [ERROR] Failed: {e}")
        if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    check_gpu()
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    print("Bu yeni kod ghost code slindi")
    # Dosyaları topla
    mr_files = glob.glob(os.path.join(INPUT_ROOT, "MR", "*.nii.gz"))
    ct_files = glob.glob(os.path.join(INPUT_ROOT, "CT", "*.nii.gz"))

    print(f"=== HIGH-RES PIPELINE STARTED ===")
    print(f"Total: {len(mr_files) + len(ct_files)} volumes.")

    for f in mr_files: process_file(f, "MR")
    for f in ct_files: process_file(f, "CT")