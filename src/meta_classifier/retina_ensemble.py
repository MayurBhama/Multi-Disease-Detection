# src/inference/retina_ensemble.py
"""
Retina Ensemble Classifier
==========================
Ensemble of 3 EfficientNet models for diabetic retinopathy classification.
Uses weighted softmax averaging based on individual model QWK scores.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import tensorflow as tf

from .loader import ModelLoader, MODEL_CONFIGS
from .utils import preprocess_retina, format_probabilities, get_prediction_result
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetinaEnsemble:
    """
    Ensemble classifier for diabetic retinopathy.
    
    Combines predictions from 3 EfficientNet models:
    - EfficientNetV2-S
    - EfficientNetB2
    - EfficientNetB0
    
    Uses weighted averaging of softmax probabilities.
    
    Usage:
        ensemble = RetinaEnsemble()
        result = ensemble.predict("path/to/retina_image.png")
    """
    
    def __init__(self):
        self.loader = ModelLoader()
        self.config = MODEL_CONFIGS["retina"]
        
        self.model_keys = ["efficientnetv2s", "efficientnetb2", "efficientnetb0"]
        self.weights = self.config["ensemble_weights"]
        self.input_shape = self.config["input_shape"]
        self.num_classes = self.config["num_classes"]
        
        self.models: Dict[str, tf.keras.Model] = {}
        self.class_labels: Optional[Dict] = None
        
        self._loaded = False
    
    def load(self) -> None:
        """Load all ensemble models and class labels."""
        if self._loaded:
            logger.debug("Retina ensemble already loaded")
            return
        
        logger.info("Loading Retina Ensemble models...")
        
        # Load all 3 models
        for key in self.model_keys:
            self.models[key] = self.loader.load_model("retina", model_key=key)
        
        # Load class labels
        self.class_labels = self.loader.load_class_labels("retina")
        
        self._loaded = True
        logger.info(f"Retina Ensemble loaded: {len(self.models)} models")
        logger.info(f"  Weights: {dict(zip(self.model_keys, self.weights))}")
    
    def _preprocess(self, image_path: str, model_key: str) -> np.ndarray:
        """
        Preprocess image for specific model.
        
        V2-S uses v2 preprocessing, B-series use v1.
        """
        version = "v2" if "v2" in model_key else "v1"
        target_size = (self.input_shape[0], self.input_shape[1])
        return preprocess_retina(image_path, target_size=target_size, version=version)
    
    def predict(
        self,
        image_path: str,
        return_individual: bool = False
    ) -> Dict[str, Any]:
        """
        Predict diabetic retinopathy severity.
        
        Args:
            image_path: Path to fundus image
            return_individual: If True, include individual model predictions
            
        Returns:
            Prediction result dictionary
        """
        # Ensure models are loaded
        if not self._loaded:
            self.load()
        
        individual_preds = {}
        ensemble_probs = np.zeros(self.num_classes)
        
        # Get predictions from each model
        for model_key, weight in zip(self.model_keys, self.weights):
            # Preprocess for this specific model
            img = self._preprocess(image_path, model_key)
            
            # Predict
            model = self.models[model_key]
            probs = model.predict(img, verbose=0)[0]
            
            # Store individual prediction
            individual_preds[model_key] = {
                "probabilities": probs.tolist(),
                "predicted_class_id": int(np.argmax(probs)),
                "confidence": float(np.max(probs)),
                "weight": weight
            }
            
            # Add to weighted ensemble
            ensemble_probs += probs * weight
        
        # Normalize ensemble probabilities (weights should sum to 1, but just in case)
        ensemble_probs = ensemble_probs / sum(self.weights)
        
        # Get final prediction
        class_names = self.class_labels["classes"]
        predicted_class, class_id, confidence = get_prediction_result(ensemble_probs, class_names)
        
        result = {
            "disease_type": "retina",
            "predicted_class": predicted_class,
            "class_id": class_id,
            "confidence": round(confidence, 6),
            "probabilities": format_probabilities(ensemble_probs, class_names),
            "ensemble_info": {
                "num_models": len(self.models),
                "weights": dict(zip(self.model_keys, self.weights))
            }
        }
        
        if return_individual:
            # Add individual model predictions
            for key, pred in individual_preds.items():
                pred["predicted_class"] = class_names[pred["predicted_class_id"]]
                pred["probabilities"] = format_probabilities(
                    np.array(pred["probabilities"]), class_names
                )
            result["individual_predictions"] = individual_preds
        
        return result
    
    def predict_batch(
        self, 
        image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Predict for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        return [self.predict(path) for path in image_paths]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about ensemble models."""
        if not self._loaded:
            self.load()
        
        return {
            "num_models": len(self.models),
            "models": {
                key: {
                    "parameters": model.count_params(),
                    "weight": self.weights[i]
                }
                for i, (key, model) in enumerate(self.models.items())
            },
            "class_labels": self.class_labels,
            "input_shape": self.input_shape
        }
