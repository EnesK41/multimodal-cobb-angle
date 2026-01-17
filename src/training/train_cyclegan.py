"""
CycleGAN Training Module
========================
CycleGAN training for X-Ray to DRR domain transfer.

Usage:
    # From terminal
    python -m src.training.train_cyclegan
    
    # Or from Python
    from src.training import CycleGANTrainer
    trainer = CycleGANTrainer()
    trainer.train()
"""

import os
import sys
import glob
import itertools

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Config and Model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import BASE_DIR, DATA_DIR, DEVICE
from src.models.cyclegan import Generator, Discriminator
from src.models.losses import GradientConsistencyLoss


class TrainConfig:
    """CycleGAN training configuration."""
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data Paths
    DRR_PATH = os.path.join(DATA_DIR, "trainCT")
    XRAY_PATH = os.path.join(DATA_DIR, "trainX", "train")
    OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "cyclegan_outputs")
    
    # Training
    NUM_EPOCHS = 310
    DECAY_START_EPOCH = 155
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-4
    NUM_WORKERS = 0  # Windows
    
    # Image
    IMG_HEIGHT = 1024
    IMG_WIDTH = 512
    
    # Loss Weights
    LAMBDA_CYCLE = 10.0
    LAMBDA_IDENTITY = 5.0
    LAMBDA_GRAD = 2.0
    
    # Saving
    SAVE_IMAGE_EVERY = 5
    SAVE_MODEL_EVERY = 20
    
    # Checkpoint
    RESUME_FROM = None
    START_EPOCH = 0


class CycleGANDataset(Dataset):
    """Unpaired dataset for CycleGAN."""
    
    def __init__(self, drr_dir: str, xray_dir: str, config: TrainConfig):
        self.config = config
        
        all_drrs = sorted(
            glob.glob(os.path.join(drr_dir, "*.png")) + 
            glob.glob(os.path.join(drr_dir, "*.jpg"))
        )
        all_xrays = sorted(
            glob.glob(os.path.join(xray_dir, "*.png")) + 
            glob.glob(os.path.join(xray_dir, "*.jpg"))
        )
        
        self.files_A = [f for f in all_drrs if "maybe_" not in os.path.basename(f)]
        self.files_B = [f for f in all_xrays if "maybe_" not in os.path.basename(f)]
        
        print(f"Dataset Loaded: DRR={len(self.files_A)}, XRAY={len(self.files_B)}")
        
        self.len_data = max(len(self.files_A), len(self.files_B))
        
        self.transform = A.Compose([
            A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH),
            A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        img_A_path = self.files_A[index % len(self.files_A)]
        img_B_path = self.files_B[index % len(self.files_B)]
        
        try:
            img_A = cv2.cvtColor(cv2.imread(img_A_path), cv2.COLOR_BGR2RGB)
            img_B = cv2.cvtColor(cv2.imread(img_B_path), cv2.COLOR_BGR2RGB)
            
            return (
                self.transform(image=img_A)["image"],
                self.transform(image=img_B)["image"]
            )
        except Exception as e:
            print(f"File error: {e}")
            return (
                torch.zeros((3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH)),
                torch.zeros((3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            )


class CycleGANTrainer:
    """
    CycleGAN training class.
    
    Usage:
        trainer = CycleGANTrainer()
        trainer.train()
        
        # Or with custom config
        config = TrainConfig()
        config.NUM_EPOCHS = 100
        trainer = CycleGANTrainer(config)
        trainer.train()
    """
    
    def __init__(self, config: TrainConfig = None):
        self.config = config or TrainConfig()
        self._setup_models()
        self._setup_optimizers()
        self._setup_losses()
    
    def _setup_models(self):
        """Initialize models."""
        input_shape = (3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        
        self.gen_AB = Generator(img_channels=3, num_residuals=9).to(self.config.DEVICE)
        self.gen_BA = Generator(img_channels=3, num_residuals=9).to(self.config.DEVICE)
        self.disc_A = Discriminator(input_shape).to(self.config.DEVICE)
        self.disc_B = Discriminator(input_shape).to(self.config.DEVICE)
        
        if self.config.RESUME_FROM and os.path.exists(self.config.RESUME_FROM):
            self._load_checkpoint()
    
    def _setup_optimizers(self):
        """Create optimizer and scheduler."""
        self.opt_gen = optim.Adam(
            itertools.chain(self.gen_AB.parameters(), self.gen_BA.parameters()),
            lr=self.config.LEARNING_RATE, betas=(0.5, 0.999)
        )
        self.opt_disc = optim.Adam(
            itertools.chain(self.disc_A.parameters(), self.disc_B.parameters()),
            lr=self.config.LEARNING_RATE, betas=(0.5, 0.999)
        )
        
        def lr_lambda(epoch):
            return 1.0 - max(0, epoch + 1 - self.config.DECAY_START_EPOCH) / \
                   float(self.config.NUM_EPOCHS - self.config.DECAY_START_EPOCH + 1)
        
        self.scheduler_gen = optim.lr_scheduler.LambdaLR(self.opt_gen, lr_lambda)
        self.scheduler_disc = optim.lr_scheduler.LambdaLR(self.opt_disc, lr_lambda)
    
    def _setup_losses(self):
        """Loss functions."""
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.grad_loss = GradientConsistencyLoss().to(self.config.DEVICE)
    
    def _load_checkpoint(self):
        """Load checkpoint."""
        print(f"Loading checkpoint: {self.config.RESUME_FROM}")
        self.gen_AB.load_state_dict(torch.load(
            self.config.RESUME_FROM.replace("gen_BA", "gen_AB"),
            map_location=self.config.DEVICE
        ))
        self.gen_BA.load_state_dict(torch.load(
            self.config.RESUME_FROM.replace("gen_AB", "gen_BA"),
            map_location=self.config.DEVICE
        ))
        print("Checkpoint loaded!")
    
    def train(self):
        """Main training loop."""
        os.makedirs(self.config.OUTPUT_PATH, exist_ok=True)
        
        # Dataset
        dataset = CycleGANDataset(
            self.config.DRR_PATH, 
            self.config.XRAY_PATH, 
            self.config
        )
        
        if len(dataset.files_A) == 0 or len(dataset.files_B) == 0:
            print("Dataset is empty!")
            return
        
        loader = DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.DEVICE == "cuda"
        )
        
        scaler = torch.cuda.amp.GradScaler() if self.config.DEVICE == "cuda" else None
        
        print(f"\nCycleGAN Training - {self.config.NUM_EPOCHS} epochs\n")
        
        for epoch in range(self.config.START_EPOCH, self.config.NUM_EPOCHS):
            self._train_epoch(epoch, loader, scaler)
            
            self.scheduler_gen.step()
            self.scheduler_disc.step()
            
            # Save
            if (epoch + 1) % self.config.SAVE_MODEL_EVERY == 0:
                self._save_checkpoint(epoch + 1)
        
        # Final
        self._save_checkpoint("final")
        print("\nTraining completed!")
    
    def _train_epoch(self, epoch: int, loader: DataLoader, scaler):
        """Single epoch training."""
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for real_A, real_B in loop:
            real_A = real_A.to(self.config.DEVICE)
            real_B = real_B.to(self.config.DEVICE)
            
            # Generator
            if scaler:
                with torch.cuda.amp.autocast():
                    loss_G, fake_A, fake_B = self._generator_step(real_A, real_B)
                self.opt_gen.zero_grad()
                scaler.scale(loss_G).backward()
                scaler.step(self.opt_gen)
                
                with torch.cuda.amp.autocast():
                    loss_D = self._discriminator_step(real_A, real_B, fake_A, fake_B)
                self.opt_disc.zero_grad()
                scaler.scale(loss_D).backward()
                scaler.step(self.opt_disc)
                scaler.update()
            else:
                loss_G, fake_A, fake_B = self._generator_step(real_A, real_B)
                self.opt_gen.zero_grad()
                loss_G.backward()
                self.opt_gen.step()
                
                loss_D = self._discriminator_step(real_A, real_B, fake_A, fake_B)
                self.opt_disc.zero_grad()
                loss_D.backward()
                self.opt_disc.step()
            
            loop.set_postfix(G=loss_G.item(), D=loss_D.item())
        
        # Save sample image
        if (epoch + 1) % self.config.SAVE_IMAGE_EVERY == 0:
            with torch.no_grad():
                grid = torch.cat((real_A, fake_B, real_B, fake_A), 3) * 0.5 + 0.5
                save_image(grid, os.path.join(self.config.OUTPUT_PATH, f"epoch_{epoch+1}.png"))
    
    def _generator_step(self, real_A, real_B):
        """Generator forward pass and loss."""
        fake_B = self.gen_AB(real_A)
        fake_A = self.gen_BA(real_B)
        
        # Identity
        loss_id = (self.L1(self.gen_BA(real_A), real_A) + 
                   self.L1(self.gen_AB(real_B), real_B)) / 2
        
        # GAN
        loss_gan = (self.mse(self.disc_B(fake_B), torch.ones_like(self.disc_B(fake_B))) +
                    self.mse(self.disc_A(fake_A), torch.ones_like(self.disc_A(fake_A)))) / 2
        
        # Cycle
        recov_A = self.gen_BA(fake_B)
        recov_B = self.gen_AB(fake_A)
        loss_cycle = (self.L1(real_A, recov_A) + self.L1(real_B, recov_B)) / 2
        
        # Gradient
        loss_grad = (self.grad_loss(real_A, recov_A) + self.grad_loss(real_B, recov_B)) / 2
        
        loss_G = (loss_gan + 
                  self.config.LAMBDA_CYCLE * loss_cycle + 
                  self.config.LAMBDA_IDENTITY * loss_id +
                  self.config.LAMBDA_GRAD * loss_grad)
        
        return loss_G, fake_A, fake_B
    
    def _discriminator_step(self, real_A, real_B, fake_A, fake_B):
        """Discriminator forward pass and loss."""
        loss_real = (self.mse(self.disc_A(real_A), torch.ones_like(self.disc_A(real_A))) +
                     self.mse(self.disc_B(real_B), torch.ones_like(self.disc_B(real_B))))
        
        loss_fake = (self.mse(self.disc_A(fake_A.detach()), torch.zeros_like(self.disc_A(fake_A))) +
                     self.mse(self.disc_B(fake_B.detach()), torch.zeros_like(self.disc_B(fake_B))))
        
        return (loss_real + loss_fake) / 4
    
    def _save_checkpoint(self, tag):
        """Save model."""
        torch.save(self.gen_AB.state_dict(), 
                   os.path.join(self.config.OUTPUT_PATH, f"gen_AB_{tag}.pth"))
        torch.save(self.gen_BA.state_dict(), 
                   os.path.join(self.config.OUTPUT_PATH, f"gen_BA_{tag}.pth"))
        print(f"Checkpoint saved: {tag}")


if __name__ == "__main__":
    trainer = CycleGANTrainer()
    trainer.train()
