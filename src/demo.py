import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "augmented_dataset")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODEL_PATH = "best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- YENİ & SAĞLAM GEOMETRİ FONKSİYONU ---
def calculate_cobb_angle_robust(mask_image):
    """
    Polinom yerine 'Kayar Pencere' (Sliding Window) yöntemi kullanır.
    Çizginin resim dışına taşmasını engeller ve daha doğru ölçer.
    """
    # 1. Maske Temizliği
    binary_mask = (mask_image > 127).astype(np.uint8)
    if np.sum(binary_mask) < 100: return 0.0, None

    # En büyük parçayı al
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = (labels == largest_label).astype(np.uint8)

    # 2. İskelet Çıkarma
    skeleton = skeletonize(binary_mask)
    y_coords, x_coords = np.where(skeleton > 0)
    
    if len(y_coords) < 20: return 0.0, None

    # Koordinatları Y eksenine göre sırala (Yukarıdan aşağıya)
    sorted_indices = np.argsort(y_coords)
    y_sorted = y_coords[sorted_indices]
    x_sorted = x_coords[sorted_indices]

    # --- KAYAR PENCERE ALGORİTMASI ---
    # Omurga üzerinde 50 piksellik pencerelerle gezip yerel eğimleri ölçüyoruz.
    window_size = 40  # Pencere boyutu (Piksel)
    step = 5          # Adım aralığı
    
    slopes = []
    
    # Görselleştirme için orta noktaları saklayalım
    mid_points_y = []
    mid_points_x = []

    for i in range(0, len(y_sorted) - window_size, step):
        # Pencere içindeki noktaları al
        y_window = y_sorted[i : i + window_size]
        x_window = x_sorted[i : i + window_size]
        
        # Bu küçük parçaya DÜZ ÇİZGİ (1. derece polinom) uydur
        # Bu işlem çok kararlıdır, saçmalamaz.
        if len(y_window) < 10: continue
            
        z = np.polyfit(y_window, x_window, 1) # x = ay + b
        slope = z[0] # Eğim (a)
        
        slopes.append(slope)
        
        # Görselleştirme için pencerenin orta noktasını kaydet
        mid_idx = i + window_size // 2
        mid_points_y.append(y_sorted[mid_idx])
        mid_points_x.append(x_sorted[mid_idx])

    if not slopes: return 0.0, None

    # --- AÇI HESABI ---
    # En sağa yatık ve en sola yatık yerel eğimleri bul
    # Gürültüden kaçmak için en uç tekil değeri değil, %5'lik dilimi alıyoruz
    slopes = np.array(slopes)
    max_slope = np.percentile(slopes, 95) # En pozitif eğim
    min_slope = np.percentile(slopes, 5)  # En negatif eğim

    angle_top = np.degrees(np.arctan(max_slope))
    angle_bottom = np.degrees(np.arctan(min_slope))
    
    cobb_angle = abs(angle_top - angle_bottom)
    
    return cobb_angle, (mid_points_x, mid_points_y)

# --- ANA DEMO FONKSİYONU ---
def run_demo():
    # Hata veren dosya üzerinde deneyelim (Rotated file)
    all_files = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    if len(all_files) == 0: return

    # Rastgele veya spesifik bir dosya seç
    # image_path = all_files[0] 
    # Hata veren spesifik dosya varsa adını buraya yazabilirsin test için
    # Örn: image_path = os.path.join(IMAGES_DIR, "AO_aug0_rot9.png")
    image_path = all_files[0] 

    filename = os.path.basename(image_path)
    print(f"📂 Seçilen Dosya: {filename}")

    # Model Yükle
    model = smp.Unet(encoder_name='resnet18', in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Resim İşle
    original_img = cv2.imread(image_path)
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (512, 512))
    
    x = img_resized.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Tahmin
    with torch.no_grad():
        pred_mask = model(x)
        pred_mask = pred_mask.cpu().numpy()[0, 0]
    
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # --- YENİ SAĞLAM HESAPLAMA ---
    print("📐 Açı Hesaplanıyor (Robust Metod)...")
    angle, curve_data = calculate_cobb_angle_robust(binary_mask)
    
    # Görselleştirme
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.imshow(binary_mask, alpha=0.4, cmap='jet')
    plt.title(f"Model Tahmini: {filename}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    
    if curve_data:
        x_pts, y_pts = curve_data
        # Artık eğri değil, hesaplanan orta noktaları çiziyoruz (Daha temiz görünür)
        plt.plot(x_pts, y_pts, color='red', linewidth=3, label='Omurga Hattı')
        
        plt.text(50, 50, f"Cobb: {angle:.1f}°", color='yellow', fontsize=16, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.7))
        plt.legend()
    
    plt.title("Geometrik Analiz (Robust)")
    plt.axis('off')
    
    print(f"✅ SONUÇ: Cobb Açısı = {angle:.2f} derece")
    plt.show()

if __name__ == "__main__":
    run_demo()