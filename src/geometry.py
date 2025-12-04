"""
Geometric analysis functions for Cobb angle calculation.
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize


def calculate_cobb_angle_from_mask(mask_image):
    """
    Calculates Cobb angle from spine mask.
    
    Args:
        mask_image: Grayscale mask image (numpy array)
        
    Returns:
        tuple: (cobb_angle, curve_data) - Angle value and curve data for visualization
    """
    binary_mask = (mask_image > 127).astype(np.uint8)
    
    if np.sum(binary_mask) < 100:
        return 0.0, None

    # Select largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = (labels == largest_label).astype(np.uint8)

    # Skeletonization
    skeleton = skeletonize(binary_mask)
    y_coords, x_coords = np.where(skeleton > 0)
    
    if len(y_coords) < 10:
        return 0.0, None

    try:
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        x_sorted = x_coords[sort_idx]

        z = np.polyfit(y_sorted, x_sorted, 5)
        p = np.poly1d(z)
        p_deriv = np.polyder(p)
        
        y_range = np.linspace(min(y_sorted), max(y_sorted), 100)
        slopes = p_deriv(y_range)
        
        max_slope = np.max(slopes)
        min_slope = np.min(slopes)
        
        angle_top = np.degrees(np.arctan(max_slope))
        angle_bottom = np.degrees(np.arctan(min_slope))
        
        cobb_angle = abs(angle_top - angle_bottom)
        return cobb_angle, (p, y_range)

    except Exception as e:
        print(f"Geometry error: {e}")
        return 0.0, None


# Backward compatibility alias
calculate_cobb_angle = calculate_cobb_angle_from_mask