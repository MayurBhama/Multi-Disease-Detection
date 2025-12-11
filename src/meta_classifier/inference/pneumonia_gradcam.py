# src/meta_classifier/inference/pneumonia_gradcam.py
"""
Pneumonia Grad-CAM Wrapper (Production-Ready)
=============================================
Grad-CAM for Xception pneumonia classifier.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import cv2
import tensorflow as tf

from .overlay_utils import create_professional_overlay, create_side_by_side, apply_colormap
from ..loader import ModelLoader
from ..utils import load_image, preprocess_pneumonia
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PneumoniaGradCAM:
    """
    Production-ready Grad-CAM for pneumonia classification.
    
    Features:
    - Professional overlays with labels
    - Full resolution output
    - Error handling for deployment
    
    Classes: NORMAL, PNEUMONIA
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loader = ModelLoader(models_dir)
        
        self.model = None
        self.xception_base = None
        self.class_labels = None
        self.target_layer = "block14_sepconv2_act"
        
        self._loaded = False
    
    def load(self):
        """Load model and initialize Grad-CAM."""
        if self._loaded:
            return
        
        logger.info("Loading Pneumonia model for Grad-CAM...")
        try:
            self.model = self.loader.load_model("pneumonia")
            self.class_labels = self.loader.load_class_labels("pneumonia")
            self.xception_base = self.model.layers[0]
            self._loaded = True
            logger.info(f"Pneumonia Grad-CAM ready. Target layer: {self.target_layer}")
        except Exception as e:
            logger.error(f"Failed to load Pneumonia model: {e}")
            raise
    
    def _generate_heatmap(self, preprocessed: np.ndarray, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        try:
            target_layer = self.xception_base.get_layer(self.target_layer)
        except ValueError:
            for layer in reversed(self.xception_base.layers):
                if 'conv' in layer.name.lower() or 'act' in layer.name.lower():
                    target_layer = layer
                    self.target_layer = layer.name
                    break
        
        intermediate_model = tf.keras.Model(
            inputs=self.xception_base.input,
            outputs=[target_layer.output, self.xception_base.output]
        )
        
        image_tensor = tf.cast(preprocessed, tf.float32)
        
        with tf.GradientTape() as tape:
            conv_outputs, xception_output = intermediate_model(image_tensor)
            tape.watch(conv_outputs)
            
            x = xception_output
            for layer in self.model.layers[1:]:
                x = layer(x)
            
            loss = x[0, 0] if class_idx == 1 else 1 - x[0, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            return np.ones((8, 8))
        
        weights = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(weights[0] * conv_outputs[0], axis=-1)
        heatmap = tf.maximum(heatmap, 0).numpy()
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def explain(
        self,
        image_path: str,
        class_idx: Optional[int] = None,
        save_dir: Optional[str] = None,
        colormap: str = "turbo",
        add_labels: bool = True,
        output_size: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Generate production-ready Grad-CAM explanation.
        """
        if not self._loaded:
            self.load()
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load at full resolution
            original_image = load_image(image_path)
            original_h, original_w = original_image.shape[:2]
            
            # Preprocess
            preprocessed = preprocess_pneumonia(image_path)
            
            # Predict
            prediction = self.model.predict(preprocessed, verbose=0)[0][0]
            
            if class_idx is None:
                class_idx = 1 if prediction >= 0.5 else 0
            
            predicted_class = self.class_labels["classes"][class_idx]
            confidence = float(prediction if class_idx == 1 else 1 - prediction)
            
            # Generate heatmap
            heatmap = self._generate_heatmap(preprocessed, class_idx)
            
            # Create professional overlay at original resolution
            overlay = create_professional_overlay(
                original_image, heatmap, predicted_class, confidence,
                colormap=colormap, add_label=add_labels, add_bar=add_labels,
                output_size=output_size
            )
            
            # Colored heatmap at full resolution
            heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
            heatmap_colored = apply_colormap(heatmap_resized, colormap)
            
            # Side-by-side
            side_by_side = create_side_by_side(
                original_image, heatmap_colored, overlay,
                predicted_class, confidence
            )
            
            result = {
                "disease_type": "pneumonia",
                "image_path": image_path,
                "prediction": {
                    "class": predicted_class,
                    "class_id": class_idx,
                    "confidence": confidence,
                    "probabilities": {
                        "NORMAL": float(1 - prediction),
                        "PNEUMONIA": float(prediction)
                    }
                },
                "gradcam": {
                    "target_layer": self.target_layer,
                    "heatmap": heatmap,
                    "overlay": overlay,
                    "colormap": colormap
                }
            }
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(image_path))[0]
                
                overlay_path = os.path.join(save_dir, f"{basename}_{predicted_class}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                heatmap_path = os.path.join(save_dir, f"{basename}_{predicted_class}_heatmap.png")
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
                
                comparison_path = os.path.join(save_dir, f"{basename}_{predicted_class}_comparison.png")
                cv2.imwrite(comparison_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
                
                result["gradcam"]["overlay_path"] = overlay_path
                result["gradcam"]["heatmap_path"] = heatmap_path
                result["gradcam"]["comparison_path"] = comparison_path
                
                logger.info(f"Pneumonia Grad-CAM saved: {predicted_class} ({confidence:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {e}")
            raise RuntimeError(f"Grad-CAM generation failed: {e}") from e
