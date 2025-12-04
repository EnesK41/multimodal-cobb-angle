"""
U-Net model training script.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from tqdm import tqdm

from config import (
    IMAGES_DIR, MASKS_DIR, MODEL_PATH, ENCODER_NAME, ENCODER_WEIGHTS,
    DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS, IMG_SIZE, TRAIN_SPLIT
)


class SpineDataset(Dataset):
    """Spine segmentation dataset."""
    
    def __init__(self, images_dir, masks_dir):
        self.ids = os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __getitem__(self, i):
        id_ = self.ids[i]
        
        image = cv2.imread(os.path.join(self.images_dir, id_))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(self.masks_dir, id_), 0)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask_resized = cv2.resize(mask[0], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask_resized, axis=0)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image), torch.from_numpy(mask)

    def __len__(self):
        return len(self.ids)


def train():
    """Starts model training."""
    print(f"Training device: {DEVICE}")
    
    full_dataset = SpineDataset(IMAGES_DIR, MASKS_DIR)
    
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")

    model = smp.Unet(
        encoder_name=ENCODER_NAME, 
        encoder_weights=ENCODER_WEIGHTS, 
        in_channels=3, 
        classes=1, 
        activation="sigmoid"
    )
    model.to(DEVICE)

    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_score = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch+1}/{EPOCHS}")
        
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

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model saved!")

    print(f"\nTraining completed! Model: {MODEL_PATH}")


if __name__ == "__main__":
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Data folder not found: {IMAGES_DIR}")
    else:
        train()