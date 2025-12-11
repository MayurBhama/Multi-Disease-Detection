# src/meta_classifier/inference/retina_gradcam.py
"""
Retina Ensemble Grad-CAM (Production-Ready)
===========================================
Grad-CAM for 3-model ensemble diabetic retinopathy classifier.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import cv2
import tensorflow as tf

from .gradcam import GradCAM
from .overlay_utils import create_professional_overlay, create_side_by_side, apply_colormap
from ..loader import ModelLoader, MODEL_CONFIGS
from ..utils import load_image, preprocess_retina, crop_black_borders, resize_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetinaGradCAM:
    """
    Production-ready Grad-CAM for retina diabetic retinopathy ensemble.
    
    Features:
    - Individual and ensemble Grad-CAM
    - Professional overlays with labels
    - Weighted heatmap combination
    
    Classes: No DR, Mild, Moderate, Severe, Proliferative
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loader = ModelLoader(models_dir)
        self.config = MODEL_CONFIGS["retina"]
        
        self.model_keys = ["efficientnetv2s", "efficientnetb2", "efficientnetb0"]
        self.weights = self.config["ensemble_weights"]
        
        self.models: Dict[str, tf.keras.Model] = {}
        self.gradcams: Dict[str, GradCAM] = {}
        self.class_labels = None
        
        self._loaded = False
    
    def load(self):
        """Load all models and initialize Grad-CAM."""
        if self._loaded:
            return
        
        logger.info("Loading Retina ensemble for Grad-CAM...")
        try:
            self.class_labels = self.loader.load_class_labels("retina")
            
            for key in self.model_keys:
                logger.info(f"  Loading {key}...")
                model = self.loader.load_model("retina", model_key=key)
                self.models[key] = model
                self.gradcams[key] = GradCAM(model)
            
            self._loaded = True
            logger.info("Retina ensemble Grad-CAM ready")
        except Exception as e:
            logger.error(f"Failed to load Retina ensemble: {e}")
            raise
    
    def _preprocess_for_model(self, image_path: str, model_key: str) -> np.ndarray:
        """Preprocess image for specific model."""
        version = "v2" if "v2" in model_key else "v1"
        return preprocess_retina(image_path, target_size=(224, 224), version=version)
    
    def explain(
        self,
        image_path: str,
        class_idx: Optional[int] = None,
        save_dir: Optional[str] = None,
        return_individual: bool = True,
        colormap: str = "turbo",
        add_labels: bool = True
    ) -> Dict[str, Any]:
        """
        Generate production-ready ensemble Grad-CAM.
        """
        if not self._loaded:
            self.load()
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load at original resolution
            original = load_image(image_path)
            original_cropped = crop_black_borders(original)
            original_h, original_w = original_cropped.shape[:2]
            
            # For model input
            original_resized = resize_image(original_cropped, (224, 224))
            
            # Ensemble prediction and heatmaps
            ensemble_probs = np.zeros(5)
            individual_heatmaps = []
            individual_results = {}
            
            for model_key, weight in zip(self.model_keys, self.weights):
                preprocessed = self._preprocess_for_model(image_path, model_key)
                predictions = self.models[model_key].predict(preprocessed, verbose=0)[0]
                ensemble_probs += predictions * weight
                
                heatmap = self.gradcams[model_key].generate_heatmap(
                    preprocessed,
                    class_idx=class_idx if class_idx else int(np.argmax(predictions))
                )
                individual_heatmaps.append((heatmap, weight))
                
                if return_individual:
                    individual_results[model_key] = {
                        "prediction": {
                            "class_id": int(np.argmax(predictions)),
                            "class": self.class_labels["classes"][int(np.argmax(predictions))],
                            "confidence": float(np.max(predictions))
                        },
                        "heatmap": heatmap
                    }
            
            # Normalize ensemble
            ensemble_probs /= sum(self.weights)
            
            if class_idx is None:
                class_idx = int(np.argmax(ensemble_probs))
            
            predicted_class = self.class_labels["classes"][class_idx]
            confidence = float(ensemble_probs[class_idx])
            
            # Weighted ensemble heatmap
            ensemble_heatmap = np.zeros_like(individual_heatmaps[0][0])
            for heatmap, weight in individual_heatmaps:
                heatmap_resized = cv2.resize(heatmap, ensemble_heatmap.shape[::-1])
                ensemble_heatmap += heatmap_resized * weight
            
            if ensemble_heatmap.max() > 0:
                ensemble_heatmap /= ensemble_heatmap.max()
            
            # Professional overlay at original resolution
            overlay = create_professional_overlay(
                original_cropped, ensemble_heatmap, predicted_class, confidence,
                colormap=colormap, add_label=add_labels, add_bar=add_labels
            )
            
            # Colored heatmap at original resolution
            heatmap_resized = cv2.resize(ensemble_heatmap, (original_w, original_h))
            heatmap_colored = apply_colormap(heatmap_resized, colormap)
            
            # Side-by-side
            side_by_side = create_side_by_side(
                original_cropped, heatmap_colored, overlay,
                predicted_class, confidence
            )
            
            result = {
                "disease_type": "retina",
                "image_path": image_path,
                "prediction": {
                    "class": predicted_class,
                    "class_id": class_idx,
                    "confidence": confidence,
                    "probabilities": {
                        name: float(prob)
                        for name, prob in zip(self.class_labels["classes"], ensemble_probs)
                    }
                },
                "gradcam": {
                    "ensemble_weights": dict(zip(self.model_keys, self.weights)),
                    "heatmap": ensemble_heatmap,
                    "overlay": overlay,
                    "colormap": colormap
                }
            }
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save ensemble outputs
                overlay_path = os.path.join(save_dir, f"{basename}_ensemble_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                heatmap_path = os.path.join(save_dir, f"{basename}_ensemble_heatmap.png")
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
                
                comparison_path = os.path.join(save_dir, f"{basename}_ensemble_comparison.png")
                cv2.imwrite(comparison_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
                
                result["gradcam"]["overlay_path"] = overlay_path
                result["gradcam"]["heatmap_path"] = heatmap_path
                result["gradcam"]["comparison_path"] = comparison_path
                
                # Save individual heatmaps
                if return_individual:
                    for model_key in self.model_keys:
                        ind_heatmap = individual_results[model_key]["heatmap"]
                        ind_resized = cv2.resize(ind_heatmap, (original_w, original_h))
                        ind_colored = apply_colormap(ind_resized, colormap)
                        
                        ind_path = os.path.join(save_dir, f"{basename}_{model_key}_heatmap.png")
                        cv2.imwrite(ind_path, cv2.cvtColor(ind_colored, cv2.COLOR_RGB2BGR))
                        individual_results[model_key]["heatmap_path"] = ind_path
                
                logger.info(f"Retina Grad-CAM saved: {predicted_class} ({confidence:.1%})")
            
            if return_individual:
                result["individual_models"] = individual_results
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {e}")
            raise RuntimeError(f"Grad-CAM generation failed: {e}") from e
