import numpy as np
import cv2
from sklearn.decomposition import PCA

# VerSe / TotalSegmentator vertebrae label IDs (T1=8 to L5=24)
VERTEBRAE_LABELS = list(range(8, 25)) 

def get_vertebra_angle(mask, label_id):
    """
    Calculate orientation angle of a single vertebra using PCA.
    
    Args:
        mask: Multi-class segmentation mask
        label_id: Vertebra label ID to analyze
    
    Returns:
        Angle in degrees or None if vertebra not found
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

def calculate_cobb_angle_multiclass(multiclass_mask):
    """
    Calculate Cobb angle from multi-class vertebrae segmentation mask.
    
    Measures each vertebra's orientation and finds the pair with maximum angular difference.
    
    Args:
        multiclass_mask: Segmentation mask with vertebrae labels (T1-L5)
    
    Returns:
        tuple: (cobb_angle, debug_info_dict)
    """
    angles = {}
    
    for label_id in VERTEBRAE_LABELS:
        angle = get_vertebra_angle(multiclass_mask, label_id)
        if angle is not None:
            if label_id < 20:
                name = f"T{label_id - 7}"
            else:
                name = f"L{label_id - 19}"
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
        "lower_vertebra": best_pair[1]
    }
    
    return max_cobb, debug_data