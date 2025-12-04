# Automatic Cobb Angle Measurement

A deep learning-based system for automatic Cobb angle measurement from medical images, designed for scoliosis diagnosis.

## 🎯 About

This project automatically measures the Cobb angle (spinal curvature) from CT and X-ray images using deep learning-based segmentation. Unlike traditional manual measurement methods, it provides fast and consistent results.

### Key Features

- **Automatic Segmentation**: 3D spine segmentation using TotalSegmentator
- **DRR Generation**: Creates 2D radiograph-like images from CT volumes
- **Deep Learning Model**: U-Net architecture with ResNet-18 encoder
- **Multi-class Segmentation**: Individual vertebra labeling for accurate angle measurement

## 📁 Project Structure

```
multimodal-cobb-angle/
├── best_model.pth          # Trained model weights
├── requirements.txt        # Python dependencies
├── data/
│   ├── tobesegmented/      # Input CT files
│   │   └── CT/
│   ├── segmented_masks/    # TotalSegmentator outputs
│   └── augmented_dataset/  # Training data (2D)
│       ├── images/
│       └── masks/
└── src/
    ├── config.py           # Central configuration
    ├── auto_segmentation.py # TotalSegmentator integration
    ├── drr_utils.py        # DRR generation and augmentation
    ├── train.py            # Model training
    ├── demo.py             # Single image demo
    ├── test_visualize.py   # Test visualization
    └── calculate_mae.py    # Model evaluation (MAE)
```

## 🚀 Installation

### Requirements

- Python 3.10+
- CUDA-enabled GPU (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Automatic Segmentation (3D CT)

Generate spine masks from CT volumes:

```bash
# Place CT files in data/tobesegmented/CT/
python src/auto_segmentation.py
```

### 2. DRR Generation and Data Augmentation

Create 2D training data from 3D CT:

```bash
python src/drr_utils.py
```

### 3. Model Training

Train the segmentation model:

```bash
python src/train.py
```

### 4. Demo

Run inference on a single image:

```bash
# Use first available image
python src/demo.py

# Specify an image
python src/demo.py --image path/to/image.jpg
```

### 5. Model Evaluation

Calculate Mean Absolute Error:

```bash
python src/calculate_mae.py
```

## 📊 Configuration

All settings are managed in `src/config.py`:

```python
# Model Settings
ENCODER_NAME = "resnet18"
IMG_SIZE = 512

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 50
```

## 📝 License

This project is developed for academic purposes.

## 👤 Developer

**Enes K.**
- GitHub: [@EnesK41](https://github.com/EnesK41)
