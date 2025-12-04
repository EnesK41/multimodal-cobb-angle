import os
import glob
import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import random
import re

# --- MODÜLLER ---
from geometry import calculate_cobb_angle_multiclass
from cyclegan_model import Generator # CycleGAN mimarisi

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")  # Öncelikli klasör
TEST_DIR = os.path.join(BASE_DIR, "data", "test_real_xray")  # Yedek klasör
MODEL_UNET_PATH = "best_multiclass_model.pth"
MODEL_CYCLE_PATH = "generator_Xray2DRR.pth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 27 

def extract_ground_truth(filename):
    match = re.search(r'_gt(\d+)', filename)
    return float(match.group(1)) if match else None

def run_ultimate_demo():
    # 1. Dosya Seçimi - Önce input klasörüne bak, boşsa test_real_xray'e bak
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    
    if input_files:
        all_files = input_files
        print("📁 Kaynak: data/input/")
    else:
        all_files = glob.glob(os.path.join(TEST_DIR, "*.*"))
        print("📁 Kaynak: data/test_real_xray/")
    
    if not all_files:
        print("❌ HATA: Hiçbir klasörde görüntü bulunamadı!")
        return
    
    image_path = random.choice(all_files)
    filename = os.path.basename(image_path)
    true_angle_val = extract_ground_truth(filename)

    print(f"📂 Dosya: {filename}")

    # 2. Modelleri Yükle
    # A) CycleGAN (Temizleyici)
    print("🧹 CycleGAN Yükleniyor...")
    cycle_net = Generator().to(DEVICE)
    if os.path.exists(MODEL_CYCLE_PATH):
        cycle_net.load_state_dict(torch.load(MODEL_CYCLE_PATH, map_location=DEVICE))
        cycle_net.eval()
        USE_CYCLEGAN = True
    else:
        print("⚠️ CycleGAN modeli bulunamadı! Sadece U-Net kullanılacak.")
        USE_CYCLEGAN = False

    # B) U-Net (Segmentasyon)
    print("🧠 U-Net Yükleniyor...")
    unet = smp.Unet(encoder_name='resnet18', in_channels=3, classes=NUM_CLASSES, activation=None)
    unet.load_state_dict(torch.load(MODEL_UNET_PATH, map_location=DEVICE))
    unet.to(DEVICE)
    unet.eval()

    # 3. Görüntü İşleme
    img_raw = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    
    # Tensör Hazırlığı (-1 ile 1 arası normalizasyon CycleGAN için)
    x_tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).float().unsqueeze(0).to(DEVICE)
    x_cycle_in = (x_tensor / 127.5) - 1.0 

    start_time = time.time()
    
    # --- ADIM 1: CycleGAN (Temizleme) - KARŞILAŞTIRMA İÇİN HER İKİSİNİ DE DENE ---
    if USE_CYCLEGAN:
        with torch.no_grad():
            fake_drr = cycle_net(x_cycle_in)
            
        x_unet_cyclegan = (fake_drr + 1.0) / 2.0
        drr_vis = fake_drr.cpu().squeeze().permute(1, 2, 0).numpy()
        drr_vis = (drr_vis * 0.5 + 0.5)
    else:
        x_unet_cyclegan = x_tensor / 255.0
        drr_vis = img_resized / 255.0

    # --- ADIM 2: U-Net (Segmentasyon) - HER İKİ YOLLA DA TEST ---
    # A) CycleGAN ile
    with torch.no_grad():
        output_cycle = unet(x_unet_cyclegan)
        pred_mask_cycle = torch.argmax(output_cycle, dim=1).cpu().numpy()[0]
    
    # B) CycleGAN OLMADAN (Direkt X-ray)
    x_unet_direct = x_tensor / 255.0
    with torch.no_grad():
        output_direct = unet(x_unet_direct)
        pred_mask_direct = torch.argmax(output_direct, dim=1).cpu().numpy()[0]
    
    # --- ADIM 3: Geometri (Ölçüm) - İKİ SONUÇ ---
    pred_angle_cycle, pred_info_cycle = calculate_cobb_angle_multiclass(pred_mask_cycle)
    pred_angle_direct, pred_info_direct = calculate_cobb_angle_multiclass(pred_mask_direct)
    
    inference_time = time.time() - start_time

    # 4. Sonuçları Hazırla
    error_cycle = abs(true_angle_val - pred_angle_cycle) if true_angle_val and pred_angle_cycle else None
    error_direct = abs(true_angle_val - pred_angle_direct) if true_angle_val and pred_angle_direct else None

    # 5. Görselleştirme (4 Panel: Karşılaştırma)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 1: Orijinal X-ray + Doktor Ölçümü
    ax[0].imshow(img_resized)
    title_input = "INPUT: Real X-ray"
    if true_angle_val:
        title_input += f"\n🩺 Doktor: {true_angle_val}°"
    ax[0].set_title(title_input, fontsize=11, fontweight='bold')
    ax[0].axis('off')

    # Panel 2: CycleGAN Çıktısı
    ax[1].imshow(drr_vis, cmap='gray')
    ax[1].set_title("CycleGAN Output", fontsize=11, fontweight='bold', color='purple')
    ax[1].axis('off')

    # Panel 3: CycleGAN + U-Net Sonucu
    ax[2].imshow(img_resized, cmap='gray')
    masked_cycle = np.ma.masked_where(pred_mask_cycle == 0, pred_mask_cycle)
    ax[2].imshow(masked_cycle, alpha=0.6, cmap='tab20')
    title_cycle = f"WITH CycleGAN\n{pred_angle_cycle:.1f}°" if pred_angle_cycle else "WITH CycleGAN\nFailed"
    if error_cycle: title_cycle += f" (Err: {error_cycle:.1f}°)"
    ax[2].set_title(title_cycle, fontsize=11, fontweight='bold', color='red')
    ax[2].axis('off')

    # Panel 4: Direkt U-Net Sonucu (CycleGAN YOK)
    ax[3].imshow(img_resized, cmap='gray')
    masked_direct = np.ma.masked_where(pred_mask_direct == 0, pred_mask_direct)
    ax[3].imshow(masked_direct, alpha=0.6, cmap='tab20')
    title_direct = f"WITHOUT CycleGAN\n{pred_angle_direct:.1f}°" if pred_angle_direct else "WITHOUT CycleGAN\nFailed"
    if error_direct: title_direct += f" (Err: {error_direct:.1f}°)"
    ax[3].set_title(title_direct, fontsize=11, fontweight='bold', color='green')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 KARŞILAŞTIRMA:")
    print(f"   Ground Truth: {true_angle_val}°")
    print(f"   CycleGAN ile: {pred_angle_cycle}° (Hata: {error_cycle:.1f}°)" if error_cycle else f"   CycleGAN ile: {pred_angle_cycle}")
    print(f"   CycleGAN'sız: {pred_angle_direct}° (Hata: {error_direct:.1f}°)" if error_direct else f"   CycleGAN'sız: {pred_angle_direct}")

if __name__ == "__main__":
    run_ultimate_demo()