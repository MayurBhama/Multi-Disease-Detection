# src/inference/predictor.py
"""
Meta Classifier - Central Prediction Engine
============================================
Routes inference requests to appropriate disease-specific models.

Supports:
- brain_mri: Brain tumor classification (EfficientNetB0)
- pneumonia: Chest X-ray classification (Xception)
- retina: Diabetic retinopathy classification (EfficientNet Ensemble)

Usage:
    from src.inference import MetaClassifier
    
    classifier = MetaClassifier()
    
    # Brain MRI prediction
    result = classifier.predict("path/to/mri.png", disease_type="brain_mri")
    
    # Pneumonia prediction
    result = classifier.predict("path/to/xray.png", disease_type="pneumonia")
    
    # Retina prediction (ensemble)
    result = classifier.predict("path/to/fundus.png", disease_type="retina")
"""

import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import numpy as np
import tensorflow as tf

from .loader import ModelLoader, MODEL_CONFIGS
from .retina_ensemble import RetinaEnsemble
from .utils import (
    preprocess_brain_mri,
    preprocess_pneumonia,
    format_probabilities,
    get_prediction_result,
    sigmoid_to_binary
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetaClassifier:
    """
    Central meta-classifier for multi-disease detection.
    
    Routes predictions to disease-specific models:
    - brain_mri: EfficientNetB0 for brain tumor classification
    - pneumonia: Xception for chest X-ray classification
    - retina: Ensemble of 3 EfficientNet models for DR
    
    Attributes:
        supported_diseases: List of supported disease types
        
    Example:
        >>> classifier = MetaClassifier()
        >>> result = classifier.predict("image.png", "brain_mri")
        >>> print(result["predicted_class"])
        'glioma'
        >>> print(result["confidence"])
        0.95
    """
    
    SUPPORTED_DISEASES = ["brain_mri", "pneumonia", "retina"]
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize meta-classifier.
        
        Args:
            models_dir: Directory containing model weights
        """
        self.models_dir = models_dir
        self.loader = ModelLoader(models_dir)
        
        # Disease-specific components
        self._brain_mri_model: Optional[tf.keras.Model] = None
        self._pneumonia_model: Optional[tf.keras.Model] = None
        self._retina_ensemble: Optional[RetinaEnsemble] = None
        
        # Grad-CAM explainers (lazy loaded)
        self._brain_mri_gradcam = None
        self._pneumonia_gradcam = None
        self._retina_gradcam = None
        
        # Cached configs
        self._configs: Dict[str, Dict] = {}
        
        logger.info("MetaClassifier initialized")
        logger.info(f"  Supported diseases: {self.SUPPORTED_DISEASES}")
        logger.info(f"  Models directory: {models_dir}")
    
    def _validate_disease_type(self, disease_type: str) -> None:
        """Validate disease type is supported."""
        if disease_type not in self.SUPPORTED_DISEASES:
            raise ValueError(
                f"Unknown disease type: '{disease_type}'. "
                f"Supported types: {self.SUPPORTED_DISEASES}"
            )
    
    def _validate_image_path(self, image_path: str) -> None:
        """Validate image file exists."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
    
    def _get_class_labels(self, disease_type: str) -> Dict:
        """Get class labels for disease type."""
        if disease_type not in self._configs:
            self._configs[disease_type] = {
                "labels": self.loader.load_class_labels(disease_type),
                "preprocessing": self.loader.load_preprocessing_config(disease_type)
            }
        return self._configs[disease_type]["labels"]
    
    def _get_preprocessing_config(self, disease_type: str) -> Dict:
        """Get preprocessing config for disease type."""
        if disease_type not in self._configs:
            self._configs[disease_type] = {
                "labels": self.loader.load_class_labels(disease_type),
                "preprocessing": self.loader.load_preprocessing_config(disease_type)
            }
        return self._configs[disease_type]["preprocessing"]
    
    def _predict_brain_mri(self, image_path: str) -> Dict[str, Any]:
        """
        Predict brain tumor type from MRI image.
        
        Classes: glioma, meningioma, notumor, pituitary
        """
        # Load model if not cached
        if self._brain_mri_model is None:
            self._brain_mri_model = self.loader.load_model("brain_mri")
        
        # Get configs
        class_labels = self._get_class_labels("brain_mri")
        preprocess_config = self._get_preprocessing_config("brain_mri")
        
        # Preprocess
        img = preprocess_brain_mri(image_path)
        
        # Predict
        predictions = self._brain_mri_model.predict(img, verbose=0)[0]
        
        # Format result
        class_names = class_labels["classes"]
        predicted_class, class_id, confidence = get_prediction_result(predictions, class_names)
        
        return {
            "disease_type": "brain_mri",
            "predicted_class": predicted_class,
            "class_id": class_id,
            "confidence": round(confidence, 6),
            "probabilities": format_probabilities(predictions, class_names),
            "preprocessing": preprocess_config,
            "model_info": {
                "architecture": "EfficientNetB0",
                "num_classes": len(class_names)
            }
        }
    
    def _predict_pneumonia(self, image_path: str) -> Dict[str, Any]:
        """
        Predict pneumonia from chest X-ray.
        
        Classes: NORMAL, PNEUMONIA
        """
        # Load model if not cached
        if self._pneumonia_model is None:
            self._pneumonia_model = self.loader.load_model("pneumonia")
        
        # Get configs
        class_labels = self._get_class_labels("pneumonia")
        preprocess_config = self._get_preprocessing_config("pneumonia")
        
        # Preprocess
        img = preprocess_pneumonia(image_path)
        
        # Predict (sigmoid output for binary)
        prediction = self._pneumonia_model.predict(img, verbose=0)[0][0]
        
        # Convert sigmoid to binary result
        class_names = class_labels["classes"]
        predicted_class, class_id, confidence, probabilities = sigmoid_to_binary(
            prediction, threshold=0.5, class_names=class_names
        )
        
        return {
            "disease_type": "pneumonia",
            "predicted_class": predicted_class,
            "class_id": class_id,
            "confidence": round(confidence, 6),
            "probabilities": probabilities,
            "preprocessing": preprocess_config,
            "model_info": {
                "architecture": "Xception",
                "num_classes": len(class_names),
                "output_type": "sigmoid"
            }
        }
    
    def _predict_retina(self, image_path: str, return_individual: bool = False) -> Dict[str, Any]:
        """
        Predict diabetic retinopathy severity from fundus image.
        
        Classes: No DR, Mild, Moderate, Severe, Proliferative
        
        Uses ensemble of 3 EfficientNet models with weighted averaging.
        """
        # Initialize ensemble if not loaded
        if self._retina_ensemble is None:
            self._retina_ensemble = RetinaEnsemble()
        
        # Get prediction from ensemble
        result = self._retina_ensemble.predict(image_path, return_individual=return_individual)
        
        # Add preprocessing config
        result["preprocessing"] = self._get_preprocessing_config("retina")
        result["model_info"] = {
            "architecture": "EfficientNet Ensemble",
            "num_classes": 5,
            "ensemble_models": ["EfficientNetV2-S", "EfficientNetB2", "EfficientNetB0"]
        }
        
        return result
    
    def predict(
        self,
        image_path: str,
        disease_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main prediction entry point.
        
        Args:
            image_path: Path to input image
            disease_type: One of "brain_mri", "pneumonia", "retina"
            **kwargs: Additional arguments (e.g., return_individual for retina)
            
        Returns:
            Prediction result dictionary with schema:
            {
                "disease_type": str,
                "predicted_class": str,
                "class_id": int,
                "confidence": float,
                "probabilities": {class_name: probability},
                "preprocessing": dict,
                "model_info": dict,
                "timestamp": str
            }
            
        Raises:
            ValueError: If disease_type is not supported
            FileNotFoundError: If image file doesn't exist
        """
        # Validate inputs
        self._validate_disease_type(disease_type)
        self._validate_image_path(image_path)
        
        logger.info(f"Predicting: {disease_type} | Image: {os.path.basename(image_path)}")
        
        # Route to appropriate predictor
        if disease_type == "brain_mri":
            result = self._predict_brain_mri(image_path)
        elif disease_type == "pneumonia":
            result = self._predict_pneumonia(image_path)
        elif disease_type == "retina":
            return_individual = kwargs.get("return_individual", False)
            result = self._predict_retina(image_path, return_individual=return_individual)
        else:
            raise ValueError(f"Unknown disease type: {disease_type}")
        
        # Add metadata
        result["image_path"] = image_path
        result["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"  Result: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        disease_type: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Predict for multiple images.
        
        Args:
            image_paths: List of image paths
            disease_type: Disease type for all images
            **kwargs: Additional arguments
            
        Returns:
            List of prediction results
        """
        return [
            self.predict(path, disease_type, **kwargs)
            for path in image_paths
        ]
    
    def get_supported_diseases(self) -> List[str]:
        """Get list of supported disease types."""
        return self.SUPPORTED_DISEASES.copy()
    
    def get_disease_info(self, disease_type: str) -> Dict[str, Any]:
        """
        Get information about a disease type.
        
        Args:
            disease_type: Disease type to get info for
            
        Returns:
            Dictionary with class labels, preprocessing config, etc.
        """
        self._validate_disease_type(disease_type)
        
        return {
            "disease_type": disease_type,
            "class_labels": self._get_class_labels(disease_type),
            "preprocessing": self._get_preprocessing_config(disease_type),
            "model_config": MODEL_CONFIGS.get(disease_type, {})
        }
    
    def explain(
        self,
        image_path: str,
        disease_type: str,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prediction with Grad-CAM explanation.
        
        Args:
            image_path: Path to input image
            disease_type: One of "brain_mri", "pneumonia", "retina"
            save_dir: Directory to save Grad-CAM outputs (default: outputs/gradcam/{disease_type})
            **kwargs: Additional args (e.g., return_individual for retina)
            
        Returns:
            Dictionary with prediction + Grad-CAM paths:
            {
                "prediction": {...},
                "gradcam": {
                    "heatmap_path": str,
                    "overlay_path": str
                }
            }
        """
        # Validate inputs
        self._validate_disease_type(disease_type)
        self._validate_image_path(image_path)
        
        # Default save directory
        if save_dir is None:
            save_dir = f"outputs/gradcam/{disease_type}"
        
        logger.info(f"Explaining: {disease_type} | Image: {os.path.basename(image_path)}")
        
        # Late import to avoid circular dependencies
        from .inference.brain_mri_gradcam import BrainMRIGradCAM
        from .inference.pneumonia_gradcam import PneumoniaGradCAM
        from .inference.retina_gradcam import RetinaGradCAM
        
        if disease_type == "brain_mri":
            if self._brain_mri_gradcam is None:
                self._brain_mri_gradcam = BrainMRIGradCAM(self.models_dir)
            result = self._brain_mri_gradcam.explain(image_path, save_dir=save_dir)
            
        elif disease_type == "pneumonia":
            if self._pneumonia_gradcam is None:
                self._pneumonia_gradcam = PneumoniaGradCAM(self.models_dir)
            result = self._pneumonia_gradcam.explain(image_path, save_dir=save_dir)
            
        elif disease_type == "retina":
            if self._retina_gradcam is None:
                self._retina_gradcam = RetinaGradCAM(self.models_dir)
            return_individual = kwargs.get("return_individual", True)
            result = self._retina_gradcam.explain(
                image_path, save_dir=save_dir, return_individual=return_individual
            )
        else:
            raise ValueError(f"Unknown disease type: {disease_type}")
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"  Grad-CAM saved to: {save_dir}")
        
        return result
    
    def clear_cache(self) -> None:
        """Clear all cached models and reload from disk."""
        self._brain_mri_model = None
        self._pneumonia_model = None
        self._retina_ensemble = None
        self._brain_mri_gradcam = None
        self._pneumonia_gradcam = None
        self._retina_gradcam = None
        self._configs.clear()
        self.loader.clear_cache()
        logger.info("MetaClassifier cache cleared")


# =====================================================
# CONVENIENCE FUNCTION
# =====================================================
def predict(
    image_path: str,
    disease_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for single predictions.
    
    Creates a MetaClassifier instance and runs prediction.
    For repeated predictions, use MetaClassifier class directly
    to benefit from model caching.
    
    Args:
        image_path: Path to input image
        disease_type: One of "brain_mri", "pneumonia", "retina"
        
    Returns:
        Prediction result dictionary
    """
    classifier = MetaClassifier()
    return classifier.predict(image_path, disease_type, **kwargs)
