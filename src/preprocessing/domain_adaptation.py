"""
Domain Adaptation Module
========================
Tools for closing the domain gap between different datasets.
"""

import cv2
import os
import glob
import numpy as np
from typing import Optional, Tuple


# Config import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TEST_DIR


# Global cache
_reference_histogram = None


def compute_reference_histogram(reference_folder: str = None, max_samples: int = 50) -> Optional[np.ndarray]:
    """
    Computes mean histogram from reference folder.
    
    Args:
        reference_folder: Folder containing reference images
        max_samples: Maximum number of samples (for speed)
    
    Returns:
        Normalized histogram or None
    """
    global _reference_histogram
    
    if _reference_histogram is not None:
        return _reference_histogram
    
    ref_folder = reference_folder or TEST_DIR
    ref_files = glob.glob(os.path.join(ref_folder, "*.jpg")) + \
                glob.glob(os.path.join(ref_folder, "*.png"))
    
    if not ref_files:
        print(f"⚠️ Reference images not found in {ref_folder}!")
        return None
    
    sample_files = ref_files[:max_samples] if len(ref_files) > max_samples else ref_files
    
    all_histograms = []
    for f in sample_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
            all_histograms.append(hist)
    
    if not all_histograms:
        return None
    
    _reference_histogram = np.mean(all_histograms, axis=0)
    _reference_histogram = _reference_histogram / _reference_histogram.sum()
    
    return _reference_histogram


def histogram_matching(source_img: np.ndarray, reference_hist: np.ndarray) -> np.ndarray:
    """
    Matches image histogram to reference histogram.
    
    Args:
        source_img: Source grayscale image
        reference_hist: Target histogram (normalized)
    
    Returns:
        Histogram matched image
    """
    if reference_hist is None:
        return source_img
    
    # Source CDF
    src_hist, _ = np.histogram(source_img.flatten(), bins=256, range=(0, 256))
    src_cdf = src_hist.cumsum()
    src_cdf = src_cdf / src_cdf[-1]
    
    # Reference CDF
    ref_cdf = reference_hist.cumsum()
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    # Lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        lookup_table[src_val] = np.argmin(diff)
    
    return lookup_table[source_img]


class DomainAdapter:
    """
    Tools for domain adaptation.
    
    Analyzes and reduces domain gap before CycleGAN training.
    
    Usage:
        adapter = DomainAdapter(reference_folder="data/test_real_xray")
        adapted_img = adapter.adapt(image)
        
        # or domain gap analysis
        score = adapter.compute_gap_score(image)
    """
    
    def __init__(self, reference_folder: str = None):
        """
        Args:
            reference_folder: Image folder for reference domain
        """
        self.reference_folder = reference_folder or TEST_DIR
        self._ref_hist = None
        self._ref_stats = None
    
    @property
    def reference_histogram(self) -> np.ndarray:
        if self._ref_hist is None:
            self._ref_hist = compute_reference_histogram(self.reference_folder)
        return self._ref_hist
    
    def adapt(self, img: np.ndarray, apply_blur: bool = True) -> np.ndarray:
        """
        Adapts image to reference domain.
        
        Args:
            img: Grayscale image
            apply_blur: Apply slight blur (soften edges)
        
        Returns:
            Adapted image
        """
        result = histogram_matching(img, self.reference_histogram)
        
        if apply_blur:
            result = cv2.GaussianBlur(result, (3, 3), 0.5)
        
        return result
    
    def compute_gap_score(self, img: np.ndarray) -> float:
        """
        Calculates distance of image to reference domain.
        
        Lower score = closer domain.
        
        Args:
            img: Grayscale image
        
        Returns:
            Domain gap score (0-1, lower is better)
        """
        if self.reference_histogram is None:
            return 1.0
        
        # Image histogram
        img_hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        img_hist = img_hist / img_hist.sum()
        
        # Bhattacharyya distance
        bc = np.sum(np.sqrt(img_hist * self.reference_histogram))
        distance = -np.log(bc + 1e-10)
        
        # Normalize (0-1)
        score = 1 - np.exp(-distance)
        
        return float(score)
    
    def analyze_dataset(self, folder: str) -> dict:
        """
        Performs domain analysis for all images in a folder.
        
        Args:
            folder: Folder to analyze
        
        Returns:
            dict: Statistics (mean_gap, std_gap, min_gap, max_gap)
        """
        image_files = glob.glob(os.path.join(folder, "*.jpg")) + \
                      glob.glob(os.path.join(folder, "*.png"))
        
        if not image_files:
            return {"error": "No images found"}
        
        scores = []
        for f in image_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                scores.append(self.compute_gap_score(img))
        
        return {
            "mean_gap": float(np.mean(scores)),
            "std_gap": float(np.std(scores)),
            "min_gap": float(np.min(scores)),
            "max_gap": float(np.max(scores)),
            "num_images": len(scores)
        }



if __name__ == "__main__":
    # Test
    adapter = DomainAdapter()
    print("Domain Adapter initialized")
    print(f"Reference folder: {adapter.reference_folder}")
