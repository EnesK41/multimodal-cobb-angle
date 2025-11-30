import numpy as np
import cv2

def calculate_cobb_angle(mask_image):
    """
    Maskeden Cobb Açısı hesaplar.
    YÖNTEM: 'Center of Mass' (Ağırlık Merkezi).
    Skeletonize yerine, her satırdaki piksellerin ortalamasını alır.
    Bu yöntem zikzak oluşumunu %100 engeller.
    """
    # 1. Maske Temizliği
    binary_mask = (mask_image > 127).astype(np.uint8)
    
    # Maske boşsa dön
    if np.sum(binary_mask) < 100: return 0.0, None

    # En büyük parçayı al (Gürültü temizliği)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = (labels == largest_label).astype(np.uint8)

    # 2. AĞIRLIK MERKEZİ (Center of Mass) ÇIKARMA
    # Her satır (y) için, o satırdaki beyaz piksellerin ortalama x konumunu buluyoruz.
    # Bu bize tek ve pürüzsüz bir omurga hattı verir.
    
    y_coords, x_coords = np.nonzero(binary_mask)
    
    # Y ekseninde benzersiz satırları bul
    unique_ys = np.unique(y_coords)
    
    if len(unique_ys) < 20: return 0.0, None
    
    mid_x = []
    mid_y = []
    
    for y in unique_ys:
        # O satırdaki (y) tüm x indekslerini bul
        xs_in_row = x_coords[y_coords == y]
        # Ortalamasını al (Merkez nokta)
        avg_x = np.mean(xs_in_row)
        
        mid_x.append(avg_x)
        mid_y.append(y)
        
    mid_x = np.array(mid_x)
    mid_y = np.array(mid_y)

    try:
        # --- KRİTİK: UÇLARI KESME (TRIMMING) ---
        # Kanca etkisini önlemek için %10 alttan ve üstten kırpıyoruz
        min_y = np.min(mid_y)
        max_y = np.max(mid_y)
        height = max_y - min_y
        margin = int(height * 0.10)
        
        valid_indices = (mid_y > (min_y + margin)) & (mid_y < (max_y - margin))
        
        if np.sum(valid_indices) > 10:
            y_trimmed = mid_y[valid_indices]
            x_trimmed = mid_x[valid_indices]
        else:
            y_trimmed = mid_y
            x_trimmed = mid_x

        # 3. Polinom Uydurma (Smooth Curve)
        # Bulduğumuz orta noktalara 5. dereceden polinom uyduruyoruz
        z = np.polyfit(y_trimmed, x_trimmed, 5) 
        p = np.poly1d(z)
        p_deriv = np.polyder(p)
        
        # 4. Açı Hesabı
        y_range = np.linspace(np.min(y_trimmed), np.max(y_trimmed), 100)
        slopes = p_deriv(y_range)
        
        max_slope = np.max(slopes)
        min_slope = np.min(slopes)
        
        angle_top = np.degrees(np.arctan(max_slope))
        angle_bottom = np.degrees(np.arctan(min_slope))
        
        cobb_angle = abs(angle_top - angle_bottom)
        
        # Görselleştirme verisi (Polinom eğrisi)
        # Çizim için tüm aralığı (trimmed dahil) kapsayan noktalar üret
        full_y_range = np.linspace(min_y, max_y, 100)
        full_x_vals = p(full_y_range)
        
        return cobb_angle, (full_x_vals, full_y_range)

    except Exception as e:
        print(f"Geometri Hatası: {e}")
        return 0.0, None