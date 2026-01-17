
"""
Data Augmentation Module
========================
Data augmentation tools for spine segmentation training.
"""

import os
import glob
import cv2
import shutil
import numpy as np
import albumentations as A
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AugmentationConfig:
    """Augmentation settings."""
    # Combo 1: Geometry + Light
    shift_limit: float = 0.05
    brightness_limit: float = 0.25
    contrast_limit: float = 0.25
    
    # Combo 2: Quality degradation
    noise_var_limit: tuple = (10.0, 40.0)
    blur_limit: tuple = (3, 3)
    gamma_limit: tuple = (80, 120)
    
    # Combo 3: Hard augmentation
    dropout_max_holes: int = 4
    dropout_max_height: int = 25
    dropout_max_width: int = 25
    
    # General
    multiplier: int = 3  # Number of augmented versions per image


class SpineAugmentor:
    """
    Augmentation class for spine dataset.
    
    Usage:
        augmentor = SpineAugmentor()
        # Single image
        results = augmentor.augment_image(image, mask)
        # Folder processing
        augmentor.process_folder(
            input_img_dir="data/images",
            input_mask_dir="data/masks",
            output_img_dir="data/aug_images",
            output_mask_dir="data/aug_masks"
        )
    """
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.pipelines = self._create_pipelines()
    
    def _create_pipelines(self) -> Dict[str, A.Compose]:
        """Create augmentation pipelines."""
        cfg = self.config
        
        return {
            # Geometry + Light
            "geo_light": A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=cfg.shift_limit, 
                    scale_limit=0, 
                    rotate_limit=0,
                    border_mode=0, value=0, p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.contrast_limit, p=1.0
                )
            ]),
            
            # Quality degradation
            "quality": A.Compose([
                A.GaussNoise(var_limit=cfg.noise_var_limit, p=1.0),
                A.GaussianBlur(blur_limit=cfg.blur_limit, p=0.5),
                A.RandomGamma(gamma_limit=cfg.gamma_limit, p=0.8)
            ]),
            
            # Hard augmentation
            "hard": A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.03, scale_limit=0, rotate_limit=0,
                    border_mode=0, value=0, p=1.0
                ),
                A.CoarseDropout(
                    max_holes=cfg.dropout_max_holes,
                    max_height=cfg.dropout_max_height,
                    max_width=cfg.dropout_max_width,
                    min_holes=1, fill_value=0, p=1.0
                )
            ])
        }
    
    def augment_image(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """
        Apply all augmentations to a single image.
        
        Args:
            image: RGB image
            mask: Grayscale mask
        
        Returns:
            List of {"name": str, "image": array, "mask": array}
        """
        results = []
        
        for name, pipeline in self.pipelines.items():
            transformed = pipeline(image=image, mask=mask)
            results.append({
                "name": name,
                "image": transformed["image"],
                "mask": transformed["mask"]
            })
        
        return results
    
    def process_folder(
        self,
        input_img_dir: str,
        input_mask_dir: str,
        output_img_dir: str,
        output_mask_dir: str,
        prefix: str = "aug",
        include_original: bool = True
    ) -> int:
        """
        Augment all images in a folder.
        
        Args:
            input_img_dir: Source image folder
            input_mask_dir: Source mask folder
            output_img_dir: Target image folder
            output_mask_dir: Target mask folder
            prefix: Filename prefix
            include_original: Also copy originals
        
        Returns:
            Total number of created files
        """
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(input_img_dir, "*.png")))
        
        if not image_files:
            print(f"No images found in {input_img_dir}!")
            return 0
        
        total_count = 0
        
        for img_path in tqdm(image_files, desc="Augmentation"):
            filename = os.path.basename(img_path)
            mask_path = os.path.join(input_mask_dir, filename)
            
            if not os.path.exists(mask_path):
                continue
            
            # Read images
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                continue
            
            # Save original
            if include_original:
                base_name = f"{prefix}_orig_{filename}"
                cv2.imwrite(
                    os.path.join(output_img_dir, base_name),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(os.path.join(output_mask_dir, base_name), mask)
                total_count += 1
            
            # Save augmented versions
            for aug_result in self.augment_image(image, mask):
                aug_name = f"{prefix}_{aug_result['name']}_{filename}"
                cv2.imwrite(
                    os.path.join(output_img_dir, aug_name),
                    cv2.cvtColor(aug_result['image'], cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(output_mask_dir, aug_name),
                    aug_result['mask']
                )
                total_count += 1
        
        print(f"Total {total_count} images created")
        return total_count


class DatasetMerger:
    """
    Merges and balances datasets from different sources.
    
    Usage:
        merger = DatasetMerger()
        merger.add_source("hospital", img_dir, mask_dir, augment=False)
        merger.add_source("totalsegmentator", img_dir, mask_dir, augment=True, multiplier=4)
        merger.merge(output_dir)
    """
    
    def __init__(self):
        self.sources = []
    
    def add_source(
        self,
        name: str,
        img_dir: str,
        mask_dir: str,
        augment: bool = False,
        multiplier: int = 1
    ):
        """Add data source."""
        self.sources.append({
            "name": name,
            "img_dir": img_dir,
            "mask_dir": mask_dir,
            "augment": augment,
            "multiplier": multiplier
        })
    
    def merge(
        self,
        output_img_dir: str,
        output_mask_dir: str
    ) -> Dict[str, int]:
        """
        Merge all sources.
        
        Returns:
            Number of files from each source
        """
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        
        stats = {}
        augmentor = SpineAugmentor()
        
        for source in self.sources:
            name = source["name"]
            img_files = glob.glob(os.path.join(source["img_dir"], "*.png"))
            
            count = 0
            
            for img_path in tqdm(img_files, desc=f"{name}"):
                filename = os.path.basename(img_path)
                mask_path = os.path.join(source["mask_dir"], filename)
                
                if not os.path.exists(mask_path):
                    continue
                
                if source["augment"]:
                    # With augmentation
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Original
                    base_name = f"{name}_orig_{filename}"
                    cv2.imwrite(
                        os.path.join(output_img_dir, base_name),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    )
                    cv2.imwrite(os.path.join(output_mask_dir, base_name), mask)
                    count += 1
                    
                    # Augmented
                    for aug in augmentor.augment_image(image, mask):
                        aug_name = f"{name}_{aug['name']}_{filename}"
                        cv2.imwrite(
                            os.path.join(output_img_dir, aug_name),
                            cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR)
                        )
                        cv2.imwrite(
                            os.path.join(output_mask_dir, aug_name),
                            aug['mask']
                        )
                        count += 1
                else:
                    # Direct copy
                    shutil.copy(img_path, os.path.join(output_img_dir, filename))
                    shutil.copy(mask_path, os.path.join(output_mask_dir, filename))
                    count += 1
            
            stats[name] = count
        
        # Report
        total = sum(stats.values())
        print("\n" + "="*40)
        print("DATASET MERGE REPORT")
        for name, count in stats.items():
            print(f"   {name}: {count}")
        print(f"   TOTAL: {total}")
        print("="*40)
        
        return stats


if __name__ == "__main__":
    # Test
    augmentor = SpineAugmentor()
    print("SpineAugmentor ready")
    print(f"Number of pipelines: {len(augmentor.pipelines)}")
