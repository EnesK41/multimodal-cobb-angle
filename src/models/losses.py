"""
Custom Loss Functions
=====================
Custom loss functions for CycleGAN and segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientConsistencyLoss(nn.Module):
    """
    Measures edge consistency with Sobel filter.
    Ensures preservation of anatomical structures.
    
    Usage:
        loss_fn = GradientConsistencyLoss().to(device)
        loss = loss_fn(real_image, generated_image)
    """
    def __init__(self):
        super(GradientConsistencyLoss, self).__init__()
        # Sobel kernels
        kernel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, real, generated):
        # Grayscale conversion
        real_gray = torch.mean(real, dim=1, keepdim=True)
        gen_gray = torch.mean(generated, dim=1, keepdim=True)
        
        # Sobel gradients
        grad_real_x = F.conv2d(real_gray, self.kernel_x, padding=1)
        grad_real_y = F.conv2d(real_gray, self.kernel_y, padding=1)
        grad_gen_x = F.conv2d(gen_gray, self.kernel_x, padding=1)
        grad_gen_y = F.conv2d(gen_gray, self.kernel_y, padding=1)
        
        # L1 loss
        return F.l1_loss(grad_real_x, grad_gen_x) + F.l1_loss(grad_real_y, grad_gen_y)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    1 - (2 * intersection / union)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    BCE + Dice Loss combination.
    More stable training for segmentation.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
