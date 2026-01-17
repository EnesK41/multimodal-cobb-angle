"""
KAGGLE ARCHIVE - U-Net Spine Segmentation Training
==================================================
This file is adapted from the original Kaggle notebook.
For local use: see src/training/train_unet.py

U-Net training on a hybrid dataset (Hospital + TotalSegmentator).
"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. HYBRID DATASET CLASS
# ==========================================
class SpineDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_src = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        final_mask = np.zeros_like(mask_src, dtype=np.uint8)
        file_name = os.path.basename(mask_path).lower()

        # Hybrid logic: if 'totalsegmentator' or 'verse' in filename/path
        if "verse" in file_name or "totalsegmentator" in file_name:
            # TotalSegmentator: Cervical and Sacrum -> 255 (Ignore), T1-L5 -> 1-17
            ignore_condition = ((mask_src >= 1) & (mask_src <= 7)) | (mask_src == 25)
            final_mask[ignore_condition] = 255
            valid_indices = (mask_src >= 8) & (mask_src <= 24)
            final_mask[valid_indices] = mask_src[valid_indices] - 7
        else:
            # Hospital: Flip upside-down images
            image = cv2.flip(image, 0)
            final_mask = cv2.flip(mask_src, 0)

        if self.transform:
            augmented = self.transform(image=image, mask=final_mask)
            image = augmented['image']
            final_mask = augmented['mask']
            
        return image, final_mask.long()

# ==========================================
# 2. DATA PREPARATION
# ==========================================
IMG_DIR = "/kaggle/working/Final_Training_Set/images"
MSK_DIR = "/kaggle/working/Final_Training_Set/masks"

all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
all_masks = sorted(glob.glob(os.path.join(MSK_DIR, "*.png")))

X_train, X_val, y_train, y_val = train_test_split(all_images, all_masks, test_size=0.15, random_state=42)

 # Augmentation (on-the-fly data diversification)
train_transform = A.Compose([
    A.Resize(1024, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(1024, 512),
    A.Normalize(),
    ToTensorV2()
])

train_loader = DataLoader(SpineDataset(X_train, y_train, train_transform), batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(SpineDataset(X_val, y_val, val_transform), batch_size=4, shuffle=False, num_workers=2)

# ==========================================
# 3. MODEL, LOSS, AND TRAINING LOOP
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet("resnet34", encoder_weights="imagenet", classes=18, activation=None).to(DEVICE)

# Losses: Dice + CrossEntropy (with ignore index 255)
dice_loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=255)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_val_loss = float("inf")
EPOCHS = 70

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, msks in pbar:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Hybrid loss calculation
            loss = dice_loss_fn(outputs, msks) + ce_loss_fn(outputs, msks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                outputs = model(imgs)
                v_loss = dice_loss_fn(outputs, msks) + ce_loss_fn(outputs, msks)
                val_loss += v_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_spine_unet.pth")
            print(f"Model improved (Val Loss: {best_val_loss:.4f}), saved!")

    print("\nTraining completed. The best model is saved as 'best_spine_unet.pth'.")
