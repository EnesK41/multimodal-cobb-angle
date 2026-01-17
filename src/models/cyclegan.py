"""
CycleGAN Model Architectures
============================
Generator and Discriminator models.
For X-Ray to DRR domain transfer.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """ResNet-style residual block."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-based Generator.
    
    Architecture:
        Encoder (downsampling) -> Residual Blocks -> Decoder (upsampling)
    
    Args:
        img_channels: Input/output channel count (default: 3 RGB)
        num_residuals: Number of residual blocks (default: 9)
    """
    def __init__(self, img_channels=3, num_residuals=9):
        super().__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling (Encoder)
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features *= 2

        # Residual blocks
        for _ in range(num_residuals):
            model += [ResidualBlock(in_features)]

        # Upsampling (Decoder)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features //= 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, img_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Makes real/fake decisions on 70x70 patches.
    
    Args:
        input_shape: Input size (channels, height, width)
    """
    def __init__(self, input_shape=(3, 256, 256)):
        super().__init__()
        channels, height, width = input_shape
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
