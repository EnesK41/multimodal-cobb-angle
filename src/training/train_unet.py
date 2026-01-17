"""
U-Net Training Module
=====================
U-Net training for spine segmentation.
Hybrid dataset (from different sources) support.
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
from typing import List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEVICE, IMAGES_DIR, MASKS_DIR, MODEL_UNET_PATH


@dataclass
class UNetTrainConfig:
    """U-Net training configuration."""
    # Model
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    num_classes: int = 18  # T1-L5 (17) + background
    
    # Image
    img_height: int = 1024
    img_width: int = 512
    
    # Training
    batch_size: int = 4
    num_epochs: int = 70
    learning_rate: float = 1e-4
    test_split: float = 0.15
    num_workers: int = 2
    
    # Scheduler
    patience: int = 5
    factor: float = 0.5
    
    # Saving
    save_path: str = "best_spine_unet.pth"
    
    # Ignore index (for cervical vertebrae and sacrum)
    ignore_index: int = 255


class HybridSpineDataset(Dataset):
    """
    Hybrid dataset class.
    Automatically processes data from different sources.
    
    TotalSegmentator data: Cervical (C1-C7) and Sacrum are ignored
    Hospital data: Flip correction applied (optional)
    """
    
    def __init__(
        self, 
        img_paths: List[str], 
        mask_paths: List[str], 
        transform=None,
        flip_hospital: bool = True,
        total_label_offset: int = 7  # T1=8 -> 1 conversion
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.flip_hospital = flip_hospital
        self.total_label_offset = total_label_offset
    
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
        
        # Hybrid logic: process based on path (TotalSegmentator vs Hospital)
        if "totalsegmentator" in mask_path.lower() or "verse" in mask_path.lower():
            # TotalSegmentator: Cervical (1-7) and Sacrum (25) -> ignore (255)
            # T1-L5 (8-24) -> 1-17
            ignore_condition = ((mask_src >= 1) & (mask_src <= 7)) | (mask_src == 25)
            final_mask[ignore_condition] = 255
            
            valid_indices = (mask_src >= 8) & (mask_src <= 24)
            final_mask[valid_indices] = mask_src[valid_indices] - self.total_label_offset
        else:
            # Hospital data
            if self.flip_hospital:
                image = cv2.flip(image, 0)
                final_mask = cv2.flip(mask_src, 0)
            else:
                final_mask = mask_src
        
        if self.transform:
            augmented = self.transform(image=image, mask=final_mask)
            image = augmented['image']
            final_mask = augmented['mask']
        
        return image, final_mask.long()


class UNetTrainer:
    """
    U-Net training class.
    
    Usage:
        trainer = UNetTrainer()
        trainer.train(img_dir, mask_dir)
        
        # Or with custom config
        config = UNetTrainConfig(num_epochs=100)
        trainer = UNetTrainer(config)
        trainer.train(img_dir, mask_dir)
    """
    
    def __init__(self, config: UNetTrainConfig = None):
        self.config = config or UNetTrainConfig()
        self.device = DEVICE
        self.model = None
        self.best_val_loss = float("inf")
    
    def _create_model(self):
        """Create model."""
        self.model = smp.Unet(
            encoder_name=self.config.encoder_name,
            encoder_weights=self.config.encoder_weights,
            classes=self.config.num_classes,
            activation=None
        ).to(self.device)
    
    def _get_transforms(self) -> Tuple[A.Compose, A.Compose]:
        """Training and validation transforms."""
        train_transform = A.Compose([
            A.Resize(self.config.img_height, self.config.img_width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(self.config.img_height, self.config.img_width),
            A.Normalize(),
            ToTensorV2()
        ])
        
        return train_transform, val_transform
    
    def _prepare_data(
        self, 
        img_dir: str, 
        mask_dir: str
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders."""
        all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        all_masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        
        X_train, X_val, y_train, y_val = train_test_split(
            all_images, all_masks,
            test_size=self.config.test_split,
            random_state=42
        )
        
        train_transform, val_transform = self._get_transforms()
        
        train_dataset = HybridSpineDataset(X_train, y_train, train_transform)
        val_dataset = HybridSpineDataset(X_val, y_val, val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader
    
    def train(
        self, 
        img_dir: str = None, 
        mask_dir: str = None,
        save_path: str = None
    ):
        """
        Start U-Net training.
        
        Args:
            img_dir: Image folder (default: from config)
            mask_dir: Mask folder (default: from config)
            save_path: Model save path
        """
        img_dir = img_dir or IMAGES_DIR
        mask_dir = mask_dir or MASKS_DIR
        save_path = save_path or self.config.save_path
        
        # Create model
        self._create_model()
        
        # Prepare data loaders
        train_loader, val_loader = self._prepare_data(img_dir, mask_dir)
        
        print(f"Dataset: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")
        
        # Loss functions
        dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True,
            ignore_index=self.config.ignore_index
        )
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config.patience,
            factor=self.config.factor
        )
        
        print(f"\nU-Net Training Started ({self.config.num_epochs} epochs)")
        print(f"   Device: {self.device}")
        print(f"   Encoder: {self.config.encoder_name}")
        print(f"   Classes: {self.config.num_classes}\n")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for imgs, masks in pbar:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(imgs)
                
                loss = dice_loss(outputs, masks) + ce_loss(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = self.model(imgs)
                    loss = dice_loss(outputs, masks) + ce_loss(outputs, masks)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
            
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"⭐ Model saved (Val Loss: {self.best_val_loss:.4f})")
        
        print(f"\n✅ Training completed!")
        print(f"📂 Model: {save_path}")


if __name__ == "__main__":
    trainer = UNetTrainer()
    print("UNetTrainer ready")
    print(f"Config: {trainer.config}")
