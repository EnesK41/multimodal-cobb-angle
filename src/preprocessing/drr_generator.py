
"""
DRR Generator Module
====================
Generates DRR (Digitally Reconstructed Radiograph) from CT volumetric data.

This module is configurable for both Kaggle/Colab and local usage.
"""

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from typing import Optional, Tuple, List
from dataclasses import dataclass


# Optional NIfTI support
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("nibabel not installed. Cannot read NIfTI files. Please install nibabel.")


@dataclass
class DRRConfig:
    """DRR generation settings."""
    # Output dimensions
    target_height: int = 1024
    target_width: int = 512
    
    # HU threshold values
    hu_threshold: int = 150  # Bone visibility threshold
    hu_min: float = -1000.0  # Air
    hu_max: float = 3000.0   # Hard bone / Metal
    
    # Padding and crop settings
    pad_ratio_x: float = 0.35  # X-axis padding ratio
    pad_ratio_z: float = 0.05  # Z-axis padding ratio
    
     # Vertebra labels (T1=8 ... S1=26)
    target_labels: List[int] = None
    
     # Noise settings
    add_noise: bool = True
    noise_mean: float = 30.0
    noise_std: float = 10.0
    
    def __post_init__(self):
        if self.target_labels is None:
            self.target_labels = list(range(8, 27))  # T1-S1


class DRRGenerator:
    """
    DRR generator from CT Volume.
    
    Usage:
        generator = DRRGenerator()
        
        # Single file processing
        drr = generator.process_nifti("ct_scan.nii.gz", "segmentation.nii.gz")
        
        # Folder processing
        generator.process_folder(
            volume_dir="data/volumes",
            segmentation_dir="data/segmentations", 
            output_dir="data/drr_outputs"
        )
    """
    
    def __init__(self, config: DRRConfig = None):
        self.config = config or DRRConfig()
        
        if not HAS_NIBABEL:
            print("NIfTI support unavailable. Only numpy arrays can be processed.")
    
    def check_spine_coverage(self, mask_vol: np.ndarray) -> bool:
        """
        Checks if the spine coverage is sufficient.
        Only accepts volumes with enough vertebrae present.
        """
        unique_labels = np.unique(mask_vol)
        bones = [int(x) for x in unique_labels if x > 0]
        if len(bones) < 5:
            return False
        # Check for upper thoracic (T1-T5) and lower lumbar/sacral (L4-S1)
        has_upper = any(b in [8, 9, 10, 11, 12] for b in bones)
        has_lower = any(b in [23, 24, 25, 26] for b in bones)
        return has_upper and has_lower
    
    def get_spine_bbox(self, mask_vol: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Calculate spine bounding box."""
        target_indices = np.where(np.isin(mask_vol, self.config.target_labels))
        rows, cols, slices = target_indices
        
        if rows.size == 0:
            return None
        
        x_min, x_max = np.min(rows), np.max(rows)
        z_min, z_max = np.min(slices), np.max(slices)
        
        return x_min, x_max, z_min, z_max
    
    def resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target size and add noisy padding if needed.
        """
        h, w = img.shape
        target_h = self.config.target_height
        target_w = self.config.target_width
        
        # Resize preserving aspect ratio
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        delta_h = target_h - new_h
        delta_w = target_w - new_w
        
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        # Black padding
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Add noise (optional)
        if self.config.add_noise:
            mask = np.ones((target_h, target_w), dtype=np.uint8)
            mask[top:top+new_h, left:left+new_w] = 0
            
            noise = np.random.normal(
                self.config.noise_mean,
                self.config.noise_std,
                (target_h, target_w)
            ).astype(np.uint8)
            
            padded = np.where(mask == 1, noise, padded)
        
        return padded
    
    def create_drr(self, ct_vol: np.ndarray, mask_vol: np.ndarray) -> Optional[np.ndarray]:
        """
        Create DRR from CT and mask volumes.
        
        Args:
            ct_vol: CT volume (HU values)
            mask_vol: Segmentation mask
        
        Returns:
            DRR image or None (on error)
        """
        # Find bounding box
        bbox = self.get_spine_bbox(mask_vol)
        if bbox is None:
            return None
        
        x_min, x_max, z_min, z_max = bbox
        
        # Z-axis (less top padding, more bottom padding)
        spine_h = z_max - z_min
        z_start = z_min
        z_end = min(ct_vol.shape[2], z_max + int(spine_h * self.config.pad_ratio_z))
        
        # X-axis (horizontal padding)
        spine_w = x_max - x_min
        pad_w_amount = int(spine_w * self.config.pad_ratio_x)
        
        x_start = max(0, x_min - pad_w_amount)
        x_end = min(ct_vol.shape[0], x_max + pad_w_amount)
        
        # HU thresholding
        temp_ct = ct_vol.copy()
        temp_ct[temp_ct < self.config.hu_threshold] = -1000
        
        # Crop
        crop = temp_ct[x_start:x_end, :, z_start:z_end]
        if crop.size == 0:
            return None
        
        # Maximum Intensity Projection (MIP)
        drr = np.max(crop, axis=1)
        drr = np.rot90(drr, k=1)
        
        # Static normalization
        drr = np.clip(drr, self.config.hu_min, self.config.hu_max)
        drr = (drr - self.config.hu_min) / (self.config.hu_max - self.config.hu_min) * 255.0
        drr = drr.astype(np.uint8)
        
        # Resize and padding
        final_drr = self.resize_with_padding(drr)
        
        return final_drr
    
    def process_nifti(
        self, 
        volume_path: str, 
        segmentation_path: str
    ) -> Optional[np.ndarray]:
        """
        Create DRR from NIfTI files.
        
        Args:
            volume_path: CT volume file path (.nii or .nii.gz)
            segmentation_path: Segmentation mask path
        
        Returns:
            DRR image or None
        """
        if not HAS_NIBABEL:
            raise ImportError("nibabel required: pip install nibabel")
        
        # Load volumes
        ct = nib.as_closest_canonical(nib.load(volume_path)).get_fdata()
        mask = nib.as_closest_canonical(nib.load(segmentation_path)).get_fdata()
        
        # Spine coverage check
        if not self.check_spine_coverage(mask):
            return None
        
        return self.create_drr(ct, mask)
    
    def process_folder(
        self,
        volume_dir: str,
        segmentation_dir: str,
        output_dir: str,
        strict_filter: bool = True
    ) -> int:
        """
        Process all CT volumes in a folder.
        
        Args:
            volume_dir: CT volume folder
            segmentation_dir: Segmentation folder
            output_dir: Output folder
            strict_filter: Spine coverage check
        
        Returns:
            Number of processed files
        """
        if not HAS_NIBABEL:
            raise ImportError("nibabel required: pip install nibabel")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find segmentation files
        seg_files = sorted(
            glob.glob(os.path.join(segmentation_dir, "*.nii.gz")) +
            glob.glob(os.path.join(segmentation_dir, "*.nii"))
        )
        
        print(f"{len(seg_files)} files found")
        
        processed = 0
        skipped = 0
        
        for seg_path in tqdm(seg_files, desc="Generating DRR"):
            try:
                # Load mask
                mask = nib.as_closest_canonical(nib.load(seg_path)).get_fdata()
                
                # Strict filter
                if strict_filter and not self.check_spine_coverage(mask):
                    skipped += 1
                    continue
                
                # Find CT file
                filename = os.path.basename(seg_path)
                vol_path = os.path.join(volume_dir, filename)
                
                if not os.path.exists(vol_path):
                    # Try .nii / .nii.gz conversion
                    if filename.endswith(".nii"):
                        vol_path = vol_path + ".gz"
                    elif filename.endswith(".nii.gz"):
                        vol_path = vol_path[:-3]
                
                if not os.path.exists(vol_path):
                    continue
                
                # Load CT and process
                ct = nib.as_closest_canonical(nib.load(vol_path)).get_fdata()
                drr = self.create_drr(ct, mask)
                
                if drr is not None:
                    save_name = filename.replace(".nii.gz", ".png").replace(".nii", ".png")
                    cv2.imwrite(os.path.join(output_dir, save_name), drr)
                    processed += 1
                    
            except Exception as e:
                print(f"Error ({seg_path}): {e}")
                continue
        
        print(f"\nProcessed: {processed}")
        print(f"Skipped: {skipped}")
        
        return processed


if __name__ == "__main__":
    # Test
    generator = DRRGenerator()
    print("DRR Generator ready")
    print(f"Target size: {generator.config.target_width}x{generator.config.target_height}")
