# src/inference/__init__.py
"""
Meta Classifier Inference Module
================================
Production-ready inference engine for multi-disease detection.

Supports:
- Brain MRI tumor classification (EfficientNetB0)
- Pneumonia classification (Xception)
- Retina diabetic retinopathy (EfficientNet Ensemble)

Usage:
    from src.inference import MetaClassifier
    
    classifier = MetaClassifier()
    result = classifier.predict("path/to/image.png", disease_type="brain_mri")
"""

from .predictor import MetaClassifier
from .loader import ModelLoader
from .retina_ensemble import RetinaEnsemble

__all__ = ["MetaClassifier", "ModelLoader", "RetinaEnsemble"]
__version__ = "1.0.0"
