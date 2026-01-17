import os
import glob
import numpy as np
import cv2
import shutil
from tqdm.notebook import tqdm
import nibabel as nib
from scipy.ndimage import gaussian_filter

 # ==========================================
# IMPROVED DRR GENERATOR V2
# ==========================================
VOL_DIR = "/kaggle/input/spine-segmentation-from-ct-scans/spine_segmentation_nnunet_v2/volumes"
SEG_DIR = "/kaggle/input/spine-segmentation-from-ct-scans/spine_segmentation_nnunet_v2/segmentations"
OUTPUT_PATH = "/kaggle/working/final_dataset_bone_v2"

class DRRConfig:
    target_height, target_width = 1024, 512
    
    # NEW: Much lower threshold - balance of bone and soft tissue
    hu_threshold = 50  # Previously 200, now 50 - more detail
    hu_min, hu_max = -200.0, 1500.0  # Wider dynamic range

    # NEW: X-ray physics parameters
    use_exponential_projection = True  # Beer-Lambert law
    attenuation_coefficient = 0.0004  # X-ray attenuation coefficient

    # NEW: Image enhancement
    apply_gaussian_blur = True
    blur_sigma = 0.8  # Mild blur - artifact reduction
    apply_clahe = True  # Contrast enhancement
    clahe_clip = 2.0
    clahe_grid = (8, 8)

    # NEW: Normalization strategy
    use_adaptive_normalization = True
    percentile_min = 1  # Discard bottom 1%
    percentile_max = 99  # Discard top 1%

    pad_ratio_x, pad_ratio_z = 0.40, 0.08  # Slightly more padding
    target_labels = list(range(8, 25))  # T1-L5

class DRRGenerator:
    def __init__(self, config=DRRConfig()):
        self.cfg = config

    def check_spine_full_chain(self, mask_vol):
        """Check that all labels from T1 to L5 are present."""
        unique_labels = set(np.unique(mask_vol).astype(int))
        for label in self.cfg.target_labels:
            if label not in unique_labels:
                return False
        return True

    def get_spine_bbox(self, mask_vol):
        """Calculate spine bounding box."""
        target_indices = np.where(np.isin(mask_vol, self.cfg.target_labels))
        if target_indices[0].size == 0:
            return None
        return (np.min(target_indices[0]), np.max(target_indices[0]),
                np.min(target_indices[2]), np.max(target_indices[2]))

    def exponential_projection(self, volume):
        """
        NEW: Realistic X-ray simulation based on Beer-Lambert law
        Uses I = I₀ * exp(-μ * Σd)
        """
        # Normalize to attenuation coefficients
        mu = volume * self.cfg.attenuation_coefficient
        # Sum along beam direction (axis=1, coronal projection)
        thickness_map = np.sum(mu, axis=1)
        # Apply Beer-Lambert law
        intensity = np.exp(-thickness_map)
        # Invert for radiograph appearance (denser = darker)
        drr = 1.0 - intensity
        return drr

    def adaptive_normalization(self, image):
        """
        NEW: Percentile-based normalization
        Ignores outliers, provides better contrast
        """
        if self.cfg.use_adaptive_normalization:
            p_min = np.percentile(image, self.cfg.percentile_min)
            p_max = np.percentile(image, self.cfg.percentile_max)
            image = np.clip(image, p_min, p_max)
            if p_max > p_min:
                image = (image - p_min) / (p_max - p_min)
        else:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image

    def apply_clahe(self, image_8bit):
        """
        NEW: Contrast Limited Adaptive Histogram Equalization
        Local contrast enhancement - reveals details
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip, 
            tileGridSize=self.cfg.clahe_grid
        )
        return clahe.apply(image_8bit)

    def process_single_volume(self, ct_vol, mk_vol, x_s, x_e, z_s, z_e):
        """DRR generation for single volume"""
        # 1. Cropping
        temp_ct = ct_vol[x_s:x_e, :, z_s:z_e].copy()
        crop_mk = mk_vol[x_s:x_e, :, z_s:z_e]
        
        # 2. Thresholding (lower threshold)
        temp_ct[temp_ct < self.cfg.hu_threshold] = 0
        temp_ct = np.clip(temp_ct, self.cfg.hu_min, self.cfg.hu_max)
        
        # 3. Projection (exponential or max)
        if self.cfg.use_exponential_projection:
            drr = self.exponential_projection(temp_ct)
        else:
            drr = np.max(temp_ct, axis=1)  # Old method
        
        drr = np.rot90(drr, k=1)
        mask = np.rot90(np.max(crop_mk, axis=1), k=1)
        
        # 4. Gaussian blur (artifact reduction)
        if self.cfg.apply_gaussian_blur:
            drr = gaussian_filter(drr, sigma=self.cfg.blur_sigma)
        
        # 5. Adaptive normalization
        drr = self.adaptive_normalization(drr)
        
        # 6. 8-bit conversion
        drr_8bit = (drr * 255).astype(np.uint8)
        
        # 7. CLAHE (contrast enhancement)
        if self.cfg.apply_clahe:
            drr_8bit = self.apply_clahe(drr_8bit)
        
        # 8. Resize (INTER_CUBIC is higher quality)
        drr_final = cv2.resize(
            drr_8bit, 
            (self.cfg.target_width, self.cfg.target_height), 
            interpolation=cv2.INTER_CUBIC
        )
        mask_final = cv2.resize(
            mask.astype(np.uint8), 
            (self.cfg.target_width, self.cfg.target_height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return drr_final, mask_final

    def process_and_save(self, vol_dir, seg_dir, output_dir):
        """Main processing function"""
        img_out = os.path.join(output_dir, "images")
        msk_out = os.path.join(output_dir, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(msk_out, exist_ok=True)

        seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.nii")))
        processed = 0
        
        print(f"🔍 IMPROVED DRR GENERATION: Scanning {len(seg_files)} files...")
        print(f"⚙️  Parameters:")
        print(f"   - HU Threshold: {self.cfg.hu_threshold}")
        print(f"   - HU Range: [{self.cfg.hu_min}, {self.cfg.hu_max}]")
        print(f"   - Exponential Projection: {self.cfg.use_exponential_projection}")
        print(f"   - Gaussian Blur (σ={self.cfg.blur_sigma}): {self.cfg.apply_gaussian_blur}")
        print(f"   - CLAHE: {self.cfg.apply_clahe}")
        print(f"   - Adaptive Norm: {self.cfg.use_adaptive_normalization}\n")
        
        for seg_path in tqdm(seg_files, desc="Processing"):
            try:
                fname = os.path.basename(seg_path)
                vol_path = os.path.join(vol_dir, fname)
                if not os.path.exists(vol_path):
                    vol_path = os.path.join(vol_dir, fname.replace(".nii", "_0000.nii"))
                    if not os.path.exists(vol_path): 
                        continue

                # Load volumes
                mk_nif = nib.as_closest_canonical(nib.load(seg_path))
                mk_vol = mk_nif.get_fdata()

                # Full chain kontrolü
                if not self.check_spine_full_chain(mk_vol): 
                    continue

                ct_vol = nib.as_closest_canonical(nib.load(vol_path)).get_fdata()
                bbox = self.get_spine_bbox(mk_vol)
                if bbox is None: 
                    continue

                # Bounding box ve padding
                x_min, x_max, z_min, z_max = bbox
                sw, sh = x_max - x_min, z_max - z_min
                x_s = max(0, x_min - int(sw * self.cfg.pad_ratio_x))
                x_e = min(ct_vol.shape[0], x_max + int(sw * self.cfg.pad_ratio_x))
                z_s = z_min
                z_e = min(ct_vol.shape[2], z_max + int(sh * self.cfg.pad_ratio_z))

                # DRR production
                drr_final, mask_final = self.process_single_volume(
                    ct_vol, mk_vol, x_s, x_e, z_s, z_e
                )
                
                # Saving
                save_name = fname.split('.')[0] + ".png"
                cv2.imwrite(os.path.join(img_out, save_name), drr_final)
                cv2.imwrite(os.path.join(msk_out, save_name), mask_final)
                processed += 1

            except Exception as e:
                print(f"❌ Error ({fname}): {e}")
                continue

        print(f"\n📊 FINISHED! {processed} improved DRRs generated.")
        return processed

# ==========================================
# COMPARATIVE TEST (Optional)
# ==========================================
def compare_methods():
    """Compare old and new methods"""
    print("🔬 COMPARATIVE TEST MODE\n")
    
    # Method 1: Old (max projection)
    print("1️⃣ OLD METHOD (Max Projection)")
    config_old = DRRConfig()
    config_old.hu_threshold = 200
    config_old.use_exponential_projection = False
    config_old.apply_gaussian_blur = False
    config_old.apply_clahe = False
    gen_old = DRRGenerator(config_old)
    count_old = gen_old.process_and_save(VOL_DIR, SEG_DIR, OUTPUT_PATH + "_old")
    
    # Method 2: New (exponential + enhancements)
    print("\n2️⃣ NEW METHOD (Exponential + Enhancements)")
    config_new = DRRConfig()
    gen_new = DRRGenerator(config_new)
    count_new = gen_new.process_and_save(VOL_DIR, SEG_DIR, OUTPUT_PATH + "_new")
    
    print(f"\n📊 RESULT:")
    print(f"   Old: {count_old} DRR")
    print(f"   New: {count_new} DRR")

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    # Single version (new method)
    gen = DRRGenerator()
    gen.process_and_save(VOL_DIR, SEG_DIR, OUTPUT_PATH)
    
    # Archiving
    shutil.make_archive("/kaggle/working/results_v2", 'zip', OUTPUT_PATH)
    print(f"✅ Ready: /kaggle/working/results_v2.zip")
    
    # Karşılaştırma yapmak istersen:
    # compare_methods()
