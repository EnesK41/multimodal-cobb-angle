"""
Central configuration for all paths and hyperparameters.
"""
import os
import torch

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Dataset Paths
INPUT_DIR = os.path.join(DATA_DIR, "input")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented_dataset")
IMAGES_DIR = os.path.join(AUGMENTED_DIR, "images")
MASKS_DIR = os.path.join(AUGMENTED_DIR, "masks")
SEGMENTED_MASKS_DIR = os.path.join(DATA_DIR, "segmented_masks")
TOBESEGMENTED_DIR = os.path.join(DATA_DIR, "tobesegmented")

# Model Settings
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
NUM_CLASSES = 1
IMG_SIZE = 512

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 50
TRAIN_SPLIT = 0.9

# Device Selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DRR Generation Settings
AUGMENTATION_COUNT = 20

# Spine labels for TotalSegmentator
CT_SPINE_LABELS = [
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", 
    "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", 
    "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", 
    "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "sacrum"
]
