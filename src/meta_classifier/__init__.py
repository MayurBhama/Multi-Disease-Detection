# src/inference/__init__.py
"""
Meta Classifier Inference Module
================================
Production-ready inference engine for multi-disease detection.

Supports:
- Brain MRI tumor classification (EfficientNetB0)
- Pneumonia classification (Xception)

Usage:
    from src.inference import MetaClassifier
    
    classifier = MetaClassifier()
    result = classifier.predict("path/to/image.png", disease_type="brain_mri")
"""

from .predictor import MetaClassifier
from .loader import ModelLoader

__all__ = ["MetaClassifier", "ModelLoader"]
__version__ = "1.0.0"
