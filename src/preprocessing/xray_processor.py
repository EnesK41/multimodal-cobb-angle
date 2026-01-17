
"""
X-Ray Preprocessing Module
==========================
X-ray image cleaning, cropping, and normalization operations.
"""

import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# Config import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    FILTER_XRAYS_DIR, FILTER_XRAYS_OUTPUT_DIR, TEST_DIR,
    TARGET_WIDTH, TARGET_HEIGHT, ASPECT_RATIO,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
)



# === PARAMETERS ===
SCALE_BRIGHTNESS_THRESHOLD = 220  # Bright scale threshold
SCALE_VERTICAL_KERNEL_HEIGHT = 50
SCALE_HORIZONTAL_KERNEL_WIDTH = 50
EDGE_MARGIN_PERCENT = 0.15
SAFE_CENTER_PERCENT = 0.60


def remove_scale_rulers(img: np.ndarray) -> np.ndarray:
    """
    Remove scale rulers from X-ray images.
    
    Only operates on edge regions - preserves center anatomy.
    
    Methods:
        1. Morphological vertical/horizontal line detection
        2. Mask bright objects in edge regions
        3. Clean small bright areas with connected components
    
    Args:
        img: Grayscale image (numpy array)
    
    Returns:
        np.ndarray: Cleaned image
    """
    h, w = img.shape
    result = img.copy()
    
    # Center protection mask
    center_margin_x = int(w * (1 - SAFE_CENTER_PERCENT) / 2)
    center_margin_y = int(h * 0.10)
    
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[:, :center_margin_x] = 255
    edge_mask[:, -center_margin_x:] = 255
    edge_mask[:center_margin_y, :] = 255
    edge_mask[-center_margin_y:, :] = 255
    
    # Remove vertical lines
    _, thresh_v = cv2.threshold(result, SCALE_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, SCALE_VERTICAL_KERNEL_HEIGHT))
    vertical_lines = cv2.morphologyEx(thresh_v, cv2.MORPH_OPEN, kernel_v)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    vertical_lines = cv2.dilate(vertical_lines, kernel_dilate, iterations=2)
    vertical_lines = cv2.bitwise_and(vertical_lines, edge_mask)
    result[vertical_lines == 255] = 0
    
    # Remove horizontal lines
    _, thresh_h = cv2.threshold(result, SCALE_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (SCALE_HORIZONTAL_KERNEL_WIDTH, 1))
    horizontal_lines = cv2.morphologyEx(thresh_h, cv2.MORPH_OPEN, kernel_h)
    kernel_dilate_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    horizontal_lines = cv2.dilate(horizontal_lines, kernel_dilate_h, iterations=2)
    horizontal_lines = cv2.bitwise_and(horizontal_lines, edge_mask)
    result[horizontal_lines == 255] = 0
    
    # Bright objects in edge regions
    edge_width = int(w * EDGE_MARGIN_PERCENT)
    left_region = result[:, :edge_width]
    _, left_thresh = cv2.threshold(left_region, 200, 255, cv2.THRESH_BINARY)
    result[:, :edge_width][left_thresh == 255] = 0
    
    right_region = result[:, -edge_width:]
    _, right_thresh = cv2.threshold(right_region, 200, 255, cv2.THRESH_BINARY)
    result[:, -edge_width:][right_thresh == 255] = 0
    
    # Small bright spots
    edge_region = cv2.bitwise_and(result, result, mask=edge_mask)
    _, small_bright = cv2.threshold(edge_region, 230, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(small_bright, connectivity=8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 500 or (width < 20 and height < 20):
            result[labels == i] = 0
    
    return result


def smart_spine_center_crop(img: np.ndarray, aspect_ratio: float = None) -> np.ndarray:
    """
    Smart spine-centered cropping.
    
    Finds the densest (bone) region using vertical projection,
    and crops according to aspect ratio.
    
    Args:
        img: Grayscale image
        aspect_ratio: Target aspect ratio (default: from config)
    
    Returns:
        np.ndarray: Cropped image
    """
    if aspect_ratio is None:
        aspect_ratio = ASPECT_RATIO
        
    h, w = img.shape
    
    # Mask text and rulers
    img_cleaned = img.copy()
    img_cleaned[0:int(h*0.15), :] = 0
    img_cleaned[:, 0:int(w*0.08)] = 0
    img_cleaned[:, int(w*0.92):w] = 0
    
    # Vertical projection
    col_sums = np.sum(img_cleaned, axis=0)
    kernel_size = int(w * 0.05)
    col_sums_smooth = np.convolve(col_sums, np.ones(kernel_size)/kernel_size, mode='same')
    spine_x = np.argmax(col_sums_smooth)
    
    # Crop calculation
    ideal_crop_w = int(h * aspect_ratio)
    if ideal_crop_w > w:
        ideal_crop_w = w
    
    start_x = max(0, spine_x - (ideal_crop_w // 2))
    end_x = start_x + ideal_crop_w
    
    if end_x > w:
        end_x = w
        start_x = max(0, end_x - ideal_crop_w)
    
    return img[0:h, start_x:end_x]


class XRayPreprocessor:
    """
    X-Ray image processing class.
    
    Usage:
        processor = XRayPreprocessor()
        result = processor.process(image_path)
        
        # Or folder processing
        processor.process_folder(input_folder, output_folder)
    """
    
    def __init__(
        self,
        target_width: int = None,
        target_height: int = None,
        clahe_clip: float = None,
        clahe_grid: tuple = None,
        remove_scales: bool = True,
        enable_domain_matching: bool = False
    ):
        self.target_width = target_width or TARGET_WIDTH
        self.target_height = target_height or TARGET_HEIGHT
        self.clahe_clip = clahe_clip or CLAHE_CLIP_LIMIT
        self.clahe_grid = clahe_grid or CLAHE_TILE_GRID_SIZE
        self.remove_scales = remove_scales
        self.enable_domain_matching = enable_domain_matching
        
        # Create CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, 
            tileGridSize=self.clahe_grid
        )
    
    def process(self, image_path: str, skip_scale_removal: bool = False) -> np.ndarray:
        """
        Process a single image.
        
        Args:
            image_path: Image path
            skip_scale_removal: Skip scale removal
        
        Returns:
            np.ndarray: Processed image or None (on error)
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Normalization
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Scale removal
            skip_folders = ["test_real_xray", "input"]
            should_skip = skip_scale_removal or any(f in image_path for f in skip_folders)
            
            if self.remove_scales and not should_skip:
                img = remove_scale_rulers(img)
            
            # Smart cropping
            img_cropped = smart_spine_center_crop(img)
            if img_cropped.size == 0:
                return None
            
            # Resize
            img_resized = cv2.resize(
                img_cropped, 
                (self.target_width, self.target_height), 
                interpolation=cv2.INTER_AREA
            )
            
            # CLAHE
            img_final = self.clahe.apply(img_resized)
            
            return img_final
            
        except Exception as e:
            print(f"Error ({image_path}): {e}")
            return None
    
    def process_folder(
        self, 
        input_folder: str, 
        output_folder: str,
        extensions: list = None
    ) -> int:
        """
        Process all images in a folder.
        
        Args:
            input_folder: Source folder
            output_folder: Target folder
            extensions: File extension list
        
        Returns:
            int: Number of processed images
        """
        os.makedirs(output_folder, exist_ok=True)
        
        if extensions is None:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not image_files:
            print(f"No images found in '{input_folder}'!")
            return 0
        
        print(f"Processing {len(image_files)} images...")
        
        processed = 0
        for img_path in tqdm(image_files):
            result = self.process(img_path)
            if result is not None:
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, result)
                processed += 1
        
        print(f"{processed} images processed -> {output_folder}")
        return processed


# Legacy API compatibility
def clean_xray(image_path, skip_scale_removal=False):
    """Legacy API - use XRayPreprocessor instead."""
    processor = XRayPreprocessor()
    return processor.process(image_path, skip_scale_removal)


if __name__ == "__main__":
    processor = XRayPreprocessor()
    processor.process_folder(FILTER_XRAYS_DIR, FILTER_XRAYS_OUTPUT_DIR)
