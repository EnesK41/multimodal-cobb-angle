import os
import glob
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Girdi: Maskelerin olduğu klasörler (segmented_masks)
MASKS_ROOT_DIR = os.path.join(BASE_DIR, "data", "segmented_masks")

# Girdi: Orijinal CT dosyaları (tobesegmented/CT)
CT_ROOT_DIR = os.path.join(BASE_DIR, "data", "tobesegmented", "CT")

# Çıktı: 2D resimlerin kaydedileceği yer (DÜZELTİLEN DEĞİŞKEN)
OUTPUT_2D_DIR = os.path.join(BASE_DIR, "data", "processed_2d")

# Omurgaları sayısal bir ID'ye çevirmek için harita (Class Mapping)
VERTEBRAE_MAP = {
    "vertebrae_C1": 1, "vertebrae_C2": 2, "vertebrae_C3": 3, "vertebrae_C4": 4, "vertebrae_C5": 5, "vertebrae_C6": 6, "vertebrae_C7": 7,
    "vertebrae_T1": 8, "vertebrae_T2": 9, "vertebrae_T3": 10, "vertebrae_T4": 11, "vertebrae_T5": 12, "vertebrae_T6": 13,
    "vertebrae_T7": 14, "vertebrae_T8": 15, "vertebrae_T9": 16, "vertebrae_T10": 17, "vertebrae_T11": 18, "vertebrae_T12": 19,
    "vertebrae_L1": 20, "vertebrae_L2": 21, "vertebrae_L3": 22, "vertebrae_L4": 23, "vertebrae_L5": 24,
    "sacrum": 25
}

def load_and_combine_masks(mask_folder_path, shape_reference):
    """
    TotalSegmentator çıktısı olan klasörü okur ve birleştirir.
    """
    # Boş bir 3D matris oluştur
    combined_mask = np.zeros(shape_reference, dtype=np.uint8)
    
    # Klasördeki tüm parçaları bul
    part_files = glob.glob(os.path.join(mask_folder_path, "*.nii.gz"))
    
    if not part_files:
        return None

    for p_file in part_files:
        filename = os.path.basename(p_file)
        
        label_id = 0
        for v_name, v_id in VERTEBRAE_MAP.items():
            if v_name in filename:
                label_id = v_id
                break
        
        if label_id == 0: continue 

        # Parçayı yükle
        part_data = nib.load(p_file).get_fdata()
        
        # Ana maskeye ekle
        combined_mask[part_data > 0.5] = label_id
        
    return combined_mask

def create_drr(ct_path, mask_folder_path, output_root):
    """
    3D CT ve Maske klasörünü -> 2D PNG'ye çevirir.
    """
    filename = os.path.basename(ct_path).replace(".nii.gz", "")
    
    try:
        # 1. CT Yükle
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        
        # 2. Maskeleri Yükle ve Birleştir
        mask_data = load_and_combine_masks(mask_folder_path, ct_data.shape)
        if mask_data is None:
            print(f"   [SKIP] No masks found in {os.path.basename(mask_folder_path)}")
            return False

        # 3. PROJEKSİYON (3D -> 2D)
        # axis=1 (Y ekseni) Coronal bakış
        ct_proj = np.mean(ct_data, axis=1) 
        mask_proj = np.max(mask_data, axis=1)

        # 4. Görüntü İyileştirme
        ct_proj = np.clip(ct_proj, -500, 1500) 
        ct_proj = ((ct_proj - np.min(ct_proj)) / (np.max(ct_proj) - np.min(ct_proj)) * 255).astype(np.uint8)
        mask_proj = mask_proj.astype(np.uint8)

        # 90 derece çevir (NIfTI oryantasyonu için)
        ct_proj = np.rot90(ct_proj)
        mask_proj = np.rot90(mask_proj)

        # 5. Kaydet
        img_dir = os.path.join(output_root, "images")
        msk_dir = os.path.join(output_root, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        cv2.imwrite(os.path.join(img_dir, f"{filename}.png"), ct_proj)
        cv2.imwrite(os.path.join(msk_dir, f"{filename}.png"), mask_proj)
        
        return True

    except Exception as e:
        print(f"   [ERROR] Processing {filename}: {e}")
        return False

# --- MAIN ---
if __name__ == "__main__":
    # Çıktı klasörünü temizle/oluştur
    if not os.path.exists(OUTPUT_2D_DIR): os.makedirs(OUTPUT_2D_DIR)

    # Maske KLASÖRLERİNİ bul
    mask_folders = glob.glob(os.path.join(MASKS_ROOT_DIR, "*_mask.nii.gz"))
    
    print(f"=== DRR GENERATOR STARTED ===")
    print(f"Found {len(mask_folders)} segmented cases.")

    count = 0
    for mask_folder in tqdm(mask_folders):
        folder_name = os.path.basename(mask_folder)
        # Orijinal dosya adını bul
        ct_filename = folder_name.replace("_mask.nii.gz", ".nii.gz")
        ct_path = os.path.join(CT_ROOT_DIR, ct_filename)
        
        if os.path.exists(ct_path):
            if create_drr(ct_path, mask_folder, OUTPUT_2D_DIR):
                count += 1
        else:
            print(f"   [WARN] CT not found for mask: {folder_name}")

    print(f"\n=== COMPLETED. {count} images generated in '{OUTPUT_2D_DIR}' ===")