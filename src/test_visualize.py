import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# --- AYARLAR ---
DATA_DIR = "data/augmented_dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
MODEL_PATH = "best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize_prediction():
    # 1. Modeli Yükle (Eğitimdeki ayarların aynısı olmalı)
    print(f"Loading model from {MODEL_PATH}...")
    model = smp.Unet(
        encoder_name='resnet18', 
        encoder_weights=None, # Ağırlıkları dosyadan yükleyeceğiz
        in_channels=3, 
        classes=1, 
        activation='sigmoid'
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    # 2. Rastgele 5 Görüntü Seç
    all_files = os.listdir(IMAGES_DIR)
    test_files = np.random.choice(all_files, 5, replace=False)

    # 3. Görselleştirme Döngüsü
    plt.figure(figsize=(15, 10))
    
    for idx, filename in enumerate(test_files):
        # Görüntü Hazırlığı
        img_path = os.path.join(IMAGES_DIR, filename)
        mask_path = os.path.join(MASKS_DIR, filename)
        
        # Oku
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_mask = cv2.imread(mask_path, 0)
        
        # Modele Uygun Hale Getir (Preprocessing)
        img_input = cv2.resize(image, (512, 512))
        img_input = img_input.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # Tahmin Yap
        with torch.no_grad():
            pred_mask = model(img_tensor)
            pred_mask = pred_mask.cpu().numpy()[0, 0]
        
        # Binary Maskeye Çevir (Siyah-Beyaz Keskinleştir)
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # --- ÇİZİM ---
        # 1. Satır: Orijinal
        plt.subplot(5, 3, idx*3 + 1)
        plt.imshow(image)
        plt.title(f"Input: {filename}", fontsize=8)
        plt.axis('off')
        
        # 2. Satır: Gerçek Maske (Cevap Anahtarı)
        plt.subplot(5, 3, idx*3 + 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("Ground Truth", fontsize=8)
        plt.axis('off')

        # 3. Satır: Model Tahmini
        plt.subplot(5, 3, idx*3 + 3)
        plt.imshow(pred_mask_binary, cmap='gray')
        plt.title("AI Prediction", fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("test_results.png")
    print("✅ Sonuçlar 'test_results.png' dosyasına kaydedildi. Açıp bakabilirsin!")
    plt.show()

if __name__ == "__main__":
    visualize_prediction()