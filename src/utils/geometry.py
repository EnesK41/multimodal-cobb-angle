"""
Geometry Module
===============
Cobb angle and vertebral geometry calculations.
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Dict, List


# TotalSegmentator vertebrae label IDs (T1=8 to L5=24)
VERTEBRAE_LABELS = list(range(8, 25))


def get_vertebra_angle(mask: np.ndarray, label_id: int) -> Optional[float]:
    """
    Calculates the orientation angle of a single vertebra using PCA.
    
    Args:
        mask: Multi-class segmentation mask
        label_id: Vertebra label to analyze
    
    Returns:
        Angle in degrees or None (if vertebra not found)
    """
    y, x = np.where(mask == label_id)
    
    if len(y) < 50:
        return None

    points = np.column_stack((x, y))
    
    pca = PCA(n_components=2)
    pca.fit(points)
    
    v1 = pca.components_[0]
    angle = np.arctan2(v1[1], v1[0]) * 180 / np.pi
    
    return angle


def label_to_name(label_id: int) -> str:
    """
    Converts vertebra label ID to name.
    
    Args:
        label_id: TotalSegmentator label ID
    
    Returns:
        Vertebra name (e.g. T1, L5)
    """
    if label_id < 20:
        return f"T{label_id - 7}"
    else:
        return f"L{label_id - 19}"


def calculate_cobb_angle_multiclass(multiclass_mask: np.ndarray) -> Tuple[float, Optional[Dict]]:
    """
    Calculates Cobb angle from multi-class vertebra segmentation mask.
    
    Measures orientation of each vertebra and finds the pair with maximum angular difference.
    
    Args:
        multiclass_mask: Segmentation mask with T1-L5 labels
    
    Returns:
        tuple: (cobb_angle, debug_info_dict)
    """
    angles = {}
    
    for label_id in VERTEBRAE_LABELS:
        angle = get_vertebra_angle(multiclass_mask, label_id)
        if angle is not None:
            name = label_to_name(label_id)
            angles[name] = angle

    if len(angles) < 2:
        return 0.0, None

    max_cobb = 0.0
    best_pair = (None, None)
    
    keys = list(angles.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            name1 = keys[i]
            name2 = keys[j]
            
            diff = abs(angles[name1] - angles[name2])
            
            if diff > 90:
                diff = 180 - diff
            
            if diff > max_cobb:
                max_cobb = diff
                best_pair = (name1, name2)
    
    debug_data = {
        "all_angles": angles,
        "upper_vertebra": best_pair[0],
        "lower_vertebra": best_pair[1],
        "num_detected": len(angles)
    }
    
    return max_cobb, debug_data


def draw_vertebra_overlay(
    image: np.ndarray, 
    mask: np.ndarray, 
    angles: Dict[str, float],
    colors: Dict[str, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draws vertebra segmentation overlay on image.
    
    Args:
        image: BGR image
        mask: Segmentation mask
        angles: Vertebra angles {name: angle}
        colors: Custom colors {name: (B, G, R)}
    
    Returns:
        Image with overlay
    """
    overlay = image.copy()
    
    if colors is None:
        colors = {}
    
    for label_id in VERTEBRAE_LABELS:
        y, x = np.where(mask == label_id)
        if len(y) == 0:
            continue
        
        name = label_to_name(label_id)
        color = colors.get(name, (0, 255, 0))  # Default green
        
        # Color specific region
        for yi, xi in zip(y, x):
            overlay[yi, xi] = color
        
        # Center and text
        cx, cy = int(np.mean(x)), int(np.mean(y))
        if name in angles:
            text = f"{name}: {angles[name]:.1f}°"
            cv2.putText(overlay, text, (cx-30, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Blend with original
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    return result


def calculate_lordosis_kyphosis(angles: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates thoracic kyphosis and lumbar lordosis angles.
    
    Thoracic kyphosis: T1-T12 curvature
    Lumbar lordosis: L1-L5 curvature
    
    Args:
        angles: Vertebra angles dictionary
    
    Returns:
        dict: {"thoracic_kyphosis": float, "lumbar_lordosis": float}
    """
    thoracic = [angles.get(f"T{i}") for i in range(1, 13) if f"T{i}" in angles]
    lumbar = [angles.get(f"L{i}") for i in range(1, 6) if f"L{i}" in angles]
    
    result = {}
    
    if len(thoracic) >= 2:
        result["thoracic_kyphosis"] = abs(max(thoracic) - min(thoracic))
    
    if len(lumbar) >= 2:
        result["lumbar_lordosis"] = abs(max(lumbar) - min(lumbar))
    
    return result
