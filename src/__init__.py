"""
Multimodal Cobb Angle - Source Package
======================================

Modüller:
    - models: CycleGAN, U-Net model tanımlamaları
    - preprocessing: X-ray görüntü işleme
    - training: Model eğitim araçları
    - evaluation: Test ve karşılaştırma
    - utils: Yardımcı fonksiyonlar (geometri vb.)

Kullanım:
    from src.models import Generator, Discriminator
    from src.preprocessing import XRayPreprocessor
    from src.training import CycleGANTrainer
    from src.evaluation import analyze_domain_gap
    from src.utils import calculate_cobb_angle_multiclass
"""

__version__ = "1.0.0"
__author__ = "Multimodal Cobb Angle Project"
