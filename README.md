# Multimodal Cobb Angle Measurement System

Automatic Cobb angle measurement from spine X-Ray and DRR images using deep learning.

## Overview

This system uses a two-stage pipeline:
1. **CycleGAN**: Translates X-Ray images to DRR (Digitally Reconstructed Radiograph) domain
2. **U-Net**: Segments vertebrae from DRR images
3. **PCA-based Cobb Angle**: Calculates Cobb angle from vertebrae orientations

## Features

- **Automatic Source Detection**: Recognizes Hospital DRR, TotalSegmentator DRR, labeled X-Ray, and unknown inputs
- **Ground Truth Comparison**: Compares predictions against GT masks when available
- **Web Interface**: User-friendly Gradio interface for image analysis

## Project Structure

```
multimodal-cobb-angle/
├── app.py                    # Main Gradio application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── best_spine_unet.pth       # U-Net model weights (not in git)
├── cyclegan_xray_to_drr.pth  # CycleGAN model weights (not in git)
├── data/                     # Data directory (not in git)
│   ├── hospital_drr/         # Hospital DRR images
│   │   ├── trainCT/          # DRR images
│   │   └── masks/            # Ground truth masks
│   ├── totalsegmentator_drr/ # TotalSegmentator DRR images
│   │   ├── trainCT/          # DRR images
│   │   └── masks/            # Ground truth masks
│   └── labeled_xray/         # Labeled X-Ray images
├── src/                      # Source modules
│   ├── models/               # Model definitions
│   ├── preprocessing/        # Image preprocessing
│   ├── training/             # Training scripts
│   └── utils/                # Utility functions
└── notebooks/                # Development notebooks
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/multimodal-cobb-angle.git
cd multimodal-cobb-angle

# Install dependencies
pip install -r requirements.txt

# Download model weights (not included in git)
# Place best_spine_unet.pth and cyclegan_xray_to_drr.pth in project root
```

## Usage

### Run Web Interface
```bash
python app.py
```

The interface will open at `http://127.0.0.1:7861`

### Input Types

| Source | Path Pattern | Processing |
|--------|--------------|------------|
| Hospital DRR | `data/hospital_drr/trainCT/*.png` | Direct U-Net segmentation |
| TotalSegmentator DRR | `data/totalsegmentator_drr/trainCT/*.png` | U-Net with label offset (7) |
| X-Ray | `data/labeled_xray/*_gt*.jpg` | CycleGAN → U-Net |
| Unknown | Any other path | CycleGAN → U-Net |

### Example Paths
- Hospital: `data/hospital_drr/trainCT/AA.png`
- TotalSegmentator: `data/totalsegmentator_drr/trainCT/case_0000.png`
- X-Ray: `data/labeled_xray/sunhl-1th-06-Jan-2017-187 A AP_gt11.5.jpg`

## Model Details

### CycleGAN (X-Ray to DRR)
- Architecture: ResNet Generator with 9 blocks
- Input: Grayscale X-Ray (1024x512)
- Output: Synthetic DRR (1024x512)

### U-Net (Vertebrae Segmentation)
- Architecture: ResNet34 encoder
- Input: DRR image (1024x512)
- Output: 18-class segmentation (background + 17 vertebrae)

### Cobb Angle Calculation
- Method: PCA-based orientation estimation
- Uses top 10 largest vertebrae by pixel count
- Angle = max(slope) - min(slope)

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Difference between predicted and GT Cobb angles
- **Dice Score**: Overlap between predicted and GT segmentation masks

## Technical Implementation Details

### Label Handling & Normalization
The system handles different dataset standards through normalization:

- **Hospital Data**: Use `label_offset=0`. Images are vertically flipped during processing to match the U-Net training alignment.
- **TotalSegmentator Data**: Uses `label_offset=7`. The TotalSegmentator dataset labels vertebrae starting from C1 (1). Our model is trained on T1-L5 (17 classes).
  - T1 in TotalSegmentator is label 8.
  - T1 in our Model is label 1.
  - Offset: 8 - 1 = 7.

### Metric Calculation

#### Cobb Angle
Calculated using PCA (Principal Component Analysis) on the segmented masks:
1. Identify each vertebra blob.
2. Fit a line using PCA to determine the orientation slope.
3. Find the pair of vertebrae with the maximum angular difference.
4. `Cobb Angle = |max_slope - min_slope|`

#### Dice Score (Multi-class)
Segmentation quality is measured using the Dice Similarity Coefficient (DSC):
- Calculated individually for each vertebra class present in the Ground Truth.
- **Formula**: `2 * |Intersection| / (|GT_Area| + |Pred_Area|)`
- The final score is the mean of all individual vertebra Dice scores.

## License

This project is for academic research purposes.