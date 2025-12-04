import numpy as np
import cv2
from sklearn.decomposition import PCA

# VerSe / TotalSegmentator ID Haritası
# Bizim için önemli olan T1(8) ile L5(24) arasıdır.
VERTEBRAE_LABELS = list(range(8, 25)) 

def get_vertebra_angle(mask, label_id):
    """
    Tek bir omurun (örn: sadece L1) maskesini alır,
    PCA (Principal Component Analysis) ile duruş açısını hesaplar.
    """
    # 1. Sadece ilgili omurun piksellerini seç
    y, x = np.where(mask == label_id)
    
    # Eğer omur yoksa veya çok küçükse (gürültü) atla
    if len(y) < 50: return None

    # 2. Koordinatları (x, y) formatında listele
    # Resim koordinatlarında y aşağı doğru artar, bunu düzelteceğiz.
    points = np.column_stack((x, y))
    
    # 3. PCA ile Ana Ekseni (Orientation) Bul
    # PCA bize verinin en çok yayıldığı yönü (omurun uzun ekseni) verir.
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # İlk bileşen (v1) omurun ana yönüdür
    v1 = pca.components_[0]
    
    # 4. Açıyı Hesapla (Derece)
    angle = np.arctan2(v1[1], v1[0]) * 180 / np.pi
    
    # Açıyı dikeye göre değil yataya göre normalize et (Cobb standardı)
    return angle

def calculate_cobb_angle_multiclass(multiclass_mask):
    """
    Multi-Class Maskeden Cobb Açısı Hesabı.
    Her omurun açısını ölçer, farkı en büyük olan ikiliyi bulur.
    """
    angles = {}
    
    # 1. Her omurun açısını ölç
    for label_id in VERTEBRAE_LABELS:
        angle = get_vertebra_angle(multiclass_mask, label_id)
        if angle is not None:
            # Omur isimlerini anlaşılır yap (T1, T2... L1...)
            if label_id < 20: name = f"T{label_id - 7}"
            else: name = f"L{label_id - 19}"
            
            angles[name] = angle

    # Yeterli omur yoksa çık
    if len(angles) < 2:
        return 0.0, None

    # 2. Cobb Açısını Bul (Brute Force)
    # Herhangi iki omur arasındaki farkın maksimum olduğu değeri bul.
    max_cobb = 0.0
    best_pair = (None, None)
    
    keys = list(angles.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            name1 = keys[i]
            name2 = keys[j]
            
            # Açı farkı
            diff = abs(angles[name1] - angles[name2])
            
            # Bazen 180 derece ters vektör çıkabilir, onu düzelt
            if diff > 90: diff = 180 - diff
            
            if diff > max_cobb:
                max_cobb = diff
                best_pair = (name1, name2)
    
    # Görselleştirme verisi
    debug_data = {
        "all_angles": angles,
        "upper_vertebra": best_pair[0],
        "lower_vertebra": best_pair[1]
    }
    
    return max_cobb, debug_data