"""
Central configuration file for the multimodal Cobb angle measurement project.
All paths, hyperparameters, and settings are managed here.
"""
import os
import torch

# ==========================================
# BASE DIRECTORIES
# ==========================================
# Directory containing this file (multimodal-cobb-angle)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Base directory (bitirme - Kopya)
BASE_DIR = os.path.dirname(PROJECT_DIR)
# Data directory within the project
DATA_DIR = os.path.join(PROJECT_DIR, "data")


# ==========================================
# DATA PATHS
# ==========================================
# Input directories
INPUT_DIR = os.path.join(DATA_DIR, "input")
TEST_DIR = os.path.join(DATA_DIR, "test_real_xray")

# Filter X-Ray preprocessing paths
FILTER_XRAYS_DIR = os.path.join(DATA_DIR, "filter_xrays")
FILTER_XRAYS_OUTPUT_DIR = os.path.join(DATA_DIR, "filter_xrays_outputs")

# CT segmentation paths
TOBESEGMENTED_DIR = os.path.join(DATA_DIR, "tobesegmented")
SEGMENTED_MASKS_DIR = os.path.join(DATA_DIR, "segmented_masks")

# Augmented dataset paths (for training)
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented_dataset")
IMAGES_DIR = os.path.join(AUGMENTED_DIR, "images")
MASKS_DIR = os.path.join(AUGMENTED_DIR, "masks")

# CycleGAN training paths
CYCLEGAN_DRR_DIR = os.path.join(DATA_DIR, "trainCT")  # DRR images
CYCLEGAN_XRAY_DIR = os.path.join(DATA_DIR, "trainX", "train")  # X-Ray images
CYCLEGAN_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cyclegan_outputs")

# ==========================================
# MODEL PATHS
# ==========================================
MODEL_DIR = PROJECT_DIR
MODEL_UNET_PATH = os.path.join(MODEL_DIR, "cobb_unet_resnet34.pth")
MODEL_CYCLEGAN_PATH = os.path.join(MODEL_DIR, "cyclegan.pth")
MODEL_CYCLEGAN2_PATH = os.path.join(MODEL_DIR, "generator_Xray2DRR.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# ==========================================
# DEVICE CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()

# ==========================================
# IMAGE PREPROCESSING SETTINGS
# ==========================================
# Image dimensions (height x width)
TARGET_HEIGHT = 1024
TARGET_WIDTH = 512
ASPECT_RATIO = TARGET_HEIGHT / TARGET_WIDTH  # 2:1 aspect ratio

# For model input
IMG_SIZE = (1024, 512)  # (height, width)
IN_CHANNELS = 3

# ==========================================
# MODEL ARCHITECTURE SETTINGS
# ==========================================
# U-Net configuration
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 18  # 17 vertebrae + background
NUM_CLASSES_BINARY = 1  # For binary segmentation

# CycleGAN configuration
NUM_RESIDUALS = 9
IMG_CHANNELS = 3

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 50
TRAIN_SPLIT = 0.9
WEIGHT_DECAY = 1e-5

# ==========================================
# DATA AUGMENTATION SETTINGS
# ==========================================
AUGMENTATION_COUNT = 20  # Number of augmented samples per CT volume

# Augmentation parameters for training
AUG_ROTATION_LIMIT = 15
AUG_SCALE_LIMIT = 0.1
AUG_SHIFT_LIMIT = 0.1

# ==========================================
# CT SPINE LABELS (for TotalSegmentator)
# ==========================================
CT_SPINE_LABELS = [
    # Cervical vertebrae (C1-C7)
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", 
    "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    
    # Thoracic vertebrae (T1-T12)
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", 
    "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", 
    "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    
    # Lumbar vertebrae (L1-L5)
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    
    # Sacrum
    "sacrum"
]

# ==========================================
# IMAGE PROCESSING PARAMETERS
# ==========================================
# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Normalization parameters
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
MAX_PIXEL_VALUE = 255.0

# ==========================================
# EVALUATION SETTINGS
# ==========================================
# For calculate_mae.py
CONFIDENCE_THRESHOLD = 0.5
MIN_MASK_AREA = 100  # Minimum mask area to consider for angle calculation

# ==========================================
# VISUALIZATION SETTINGS
# ==========================================
FIG_SIZE = (18, 6)
FIG_DPI = 100
CMAP_MASK = 'jet'
CMAP_GRAY = 'gray'

# ==========================================
# LOGGING SETTINGS
# ==========================================
VERBOSE = True
LOG_INTERVAL = 10  # Log every N batches during training

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        INPUT_DIR,
        TEST_DIR,
        FILTER_XRAYS_DIR,
        FILTER_XRAYS_OUTPUT_DIR,
        TOBESEGMENTED_DIR,
        SEGMENTED_MASKS_DIR,
        AUGMENTED_DIR,
        IMAGES_DIR,
        MASKS_DIR,
        MODEL_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    if VERBOSE:
        print("All directories created/verified")

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Encoder: {ENCODER_NAME}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print("=" * 60)

# Auto-create directories on import
if __name__ != "__main__":
    try:
        ensure_directories()
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")
