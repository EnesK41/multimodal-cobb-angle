# Notebooks - Archive

This folder contains original training code from Colab/Kaggle.
**For reference and documentation only** - not meant to be run directly.

## Contents

| File | Description | Modular Version |
|------|-------------|-----------------|
| `kaggle_drr_generation.py` | DRR generation from CT | `src/preprocessing/drr_generator.py` |
| `kaggle_augmentation.py` | Data augmentation and merging | `src/preprocessing/augmentation.py` |
| `kaggle_unet_training.py` | U-Net training | `src/training/train_unet.py` |

## Notes

1. These codes are **not optimized for local execution**
2. For modular versions, see the `src/` folder
3. Kaggle/Colab paths are hardcoded - config required for local use

## Modular Usage

### DRR Generation
```python
from src.preprocessing import DRRGenerator, DRRConfig

config = DRRConfig(target_height=1024, target_width=512)
generator = DRRGenerator(config)
generator.process_folder(volume_dir, seg_dir, output_dir)
```

### Data Augmentation
```python
from src.preprocessing import SpineAugmentor, DatasetMerger

# Single source augmentation
augmentor = SpineAugmentor()
augmentor.process_folder(img_dir, mask_dir, out_img, out_mask)

# Multi-source merging
merger = DatasetMerger()
merger.add_source("hospital", hospital_img, hospital_mask, augment=False)
merger.add_source("verse", verse_img, verse_mask, augment=True)
merger.merge(output_img, output_mask)
```

### U-Net Training
```python
from src.training import UNetTrainer, UNetTrainConfig

config = UNetTrainConfig(num_epochs=70, batch_size=4)
trainer = UNetTrainer(config)
trainer.train(img_dir, mask_dir)
```

## Conversion Notes

### Kaggle to Local Changes:
- `/kaggle/input/...` to `config.py` paths
- `/kaggle/working/...` to `data/` folder
- `!pip install` to `requirements.txt`
- Notebook cells to Python classes
- Hardcoded values to dataclass configs

---
*Last updated: Archived during project modularization*
