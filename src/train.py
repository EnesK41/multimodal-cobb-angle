import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- AYARLAR ---
DATA_DIR = "data/augmented_dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
MODEL_SAVE_PATH = "best_model.pth"

# Hiperparametreler
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
BATCH_SIZE = 8 # 4070 ile 16 veya 32 de yapabilirsin, VRAM yeter
EPOCHS = 50 
IMG_SIZE = 512

print(f"🔥 Eğitim Cihazı: {DEVICE} (4070 Ti Hazır mı?)")

# --- DATASET SINIFI ---
class SpineDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.ids = os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __getitem__(self, i):
        # Dosya İsimleri
        id_ = self.ids[i]
        
        # Görüntü Oku (Grayscale değil RGB okuyoruz çünkü ResNet RGB bekler)
        image = cv2.imread(os.path.join(self.images_dir, id_))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Maske Oku
        mask = cv2.imread(os.path.join(self.masks_dir, id_), 0) # Grayscale oku
        # Maskeyi 0 ve 1 yap (Binary)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0) # (1, 512, 512) formatına getir

        # Resize (Emin olmak için)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask_resized = cv2.resize(mask[0], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask_resized, axis=0)

        # Normalizasyon (0-1 arası)
        image = image.astype(np.float32) / 255.0
        
        # Torch Tensor'a çevir (Channel First: C, H, W)
        image = np.transpose(image, (2, 0, 1)) # (3, 512, 512)
        
        return torch.from_numpy(image), torch.from_numpy(mask)

    def __len__(self):
        return len(self.ids)

# --- EĞİTİM FONKSİYONU ---
def train():
    # 1. Veri Setini Hazırla
    full_dataset = SpineDataset(IMAGES_DIR, MASKS_DIR)
    
    # Train/Val Split (%90 Train, %10 Val)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Eğitim Verisi: {len(train_dataset)} | Doğrulama Verisi: {len(val_dataset)}")

    # 2. Modeli Kur (U-Net)
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        in_channels=3, 
        classes=1, 
        activation='sigmoid'
    )
    model.to(DEVICE)

    # 3. Loss ve Optimizer
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Eğitim Döngüsü
    best_score = float('inf')

    for i in range(EPOCHS):
        print(f"\nEpoch: {i+1}/{EPOCHS}")
        
        # -- TRAIN --
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            mask_pred = model(x)
            loss = loss_fn(mask_pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # -- VALIDATION --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                mask_pred = model(x)
                loss = loss_fn(mask_pred, y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # En iyi modeli kaydet
        if avg_val_loss < best_score:
            best_score = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("💾 Model Kaydedildi (New Best Score)!")

    print(f"\n✅ Eğitim Tamamlandı! En iyi model: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    if not os.path.exists(IMAGES_DIR):
        print("❌ HATA: Veri klasörü bulunamadı. Lütfen önce augmented_dataset oluştur.")
    else:
        train()