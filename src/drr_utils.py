import os
import glob
import numpy as np
import nibabel as nib
import scipy.ndimage
import cv2
from tqdm import tqdm

# --- AYARLAR (SENİN KLASÖR YAPINA GÖRE GÜNCELLENDİ) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SENİN MEVCUT KLASÖRLERİN:
MASKS_ROOT_DIR = os.path.join(BASE_DIR, "data", "segmented_masks")   # Maskeler burada
CT_ROOT_DIR = os.path.join(BASE_DIR, "data", "tobesegmented", "CT") # Orijinal CT'ler burada
OUTPUT_2D_DIR = os.path.join(BASE_DIR, "data", "augmented_dataset")  # Çıktı buraya

# Augmentation Sayısı (Her hastadan 20 tane)
AUGMENTATION_COUNT = 20 

def apply_bone_enhancement(ct_data):
    """
    Yumuşak dokuyu temizler, sadece kemiği bırakır.
    """
    cleaned = np.copy(ct_data)
    # 200 HU altını sil (Siyah yap)
    cleaned[cleaned < 200] = -1000
    return cleaned

def create_augmented_drr(ct_path, mask_path, output_root):
    filename = os.path.basename(ct_path).replace(".nii.gz", "")
    
    try:
        # 1. Yükle
        ct_nii = nib.load(ct_path)
        mask_nii = nib.load(mask_path)
        
        ct_data = ct_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        # Boyut Kontrolü (Shape Mismatch önlemi)
        if ct_data.shape != mask_data.shape:
            # Ufak farklar varsa bile işlemi durdurmamak için log basıp geçiyoruz
            # Ancak kodun sağlam çalışması için burayı 'return' ile geçiyoruz.
            # Eğer çok fazla dosya atlanıyorsa buraya 'resample' ekleyebiliriz.
            print(f"[ATLANDI] Boyut uyumsuzluğu: {filename} (CT:{ct_data.shape} vs Mask:{mask_data.shape})")
            return

        # 2. TEMİZLİK (Yumuşak Doku Silme)
        ct_data = apply_bone_enhancement(ct_data)

        # 3. Augmentation Döngüsü
        for i in range(AUGMENTATION_COUNT):
            # A. Rastgele Açı
            angle = np.random.uniform(-10, 10)
            
            # B. Rotasyon
            ct_rot = scipy.ndimage.rotate(ct_data, angle, axes=(0, 2), reshape=False, order=1, cval=-1000)
            mask_rot = scipy.ndimage.rotate(mask_data, angle, axes=(0, 2), reshape=False, order=0, cval=0)

            # C. Projeksiyon (MIP - Max Intensity) -> KEMİKLER PARLASIN
            ct_proj = np.max(ct_rot, axis=1)
            mask_proj = np.max(mask_rot, axis=1)

            # D. Görüntü İşleme
            ct_proj = np.clip(ct_proj, -1000, 3000)
            
            _min, _max = np.min(ct_proj), np.max(ct_proj)
            if _max > _min:
                ct_proj = ((ct_proj - _min) / (_max - _min) * 255).astype(np.uint8)
            else:
                ct_proj = np.zeros_like(ct_proj, dtype=np.uint8)
            
            mask_proj = (mask_proj > 0).astype(np.uint8) * 255

            # Yön Düzeltme
            ct_proj = np.rot90(ct_proj)
            mask_proj = np.rot90(mask_proj)

            # E. Kaydetme
            save_name = f"{filename}_aug{i}_rot{int(angle)}"
            
            img_out_dir = os.path.join(output_root, "images")
            msk_out_dir = os.path.join(output_root, "masks")
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(msk_out_dir, exist_ok=True)

            cv2.imwrite(os.path.join(img_out_dir, save_name + ".png"), ct_proj)
            cv2.imwrite(os.path.join(msk_out_dir, save_name + ".png"), mask_proj)

    except Exception as e:
        print(f"[HATA] {filename}: {e}")

if __name__ == "__main__":
    # CT dosyalarını bul
    ct_files = glob.glob(os.path.join(CT_ROOT_DIR, "*.nii.gz"))
    
    print(f"=== PATH DÜZELTİLMİŞ DRR BAŞLIYOR ===")
    print(f"CT Klasörü: {CT_ROOT_DIR}")
    print(f"Maske Klasörü: {MASKS_ROOT_DIR}")
    print(f"Bulunan Dosya: {len(ct_files)}")

    if len(ct_files) == 0:
        print("❌ HATA: CT dosyaları bulunamadı! Lütfen 'CT_ROOT_DIR' yolunu kontrol et.")
    else:
        for ct_path in tqdm(ct_files):
            base_name = os.path.basename(ct_path)
            
            # Maske ismini tahmin et (Senin yapına uygun: AO.nii.gz -> AO_mask.nii.gz)
            mask_name = base_name.replace(".nii.gz", "_mask.nii.gz")
            mask_path = os.path.join(MASKS_ROOT_DIR, mask_name)
            
            if os.path.exists(mask_path):
                create_augmented_drr(ct_path, mask_path, OUTPUT_2D_DIR)
            else:
                print(f"[UYARI] Maske yok: {mask_name}")

    print(f"\n=== BİTTİ. Kontrol et: {OUTPUT_2D_DIR} ===")