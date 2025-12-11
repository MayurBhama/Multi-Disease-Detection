# src/meta_classifier/inference/brain_mri_gradcam.py
"""
Brain MRI Grad-CAM Wrapper (Production-Ready)
=============================================
Grad-CAM for EfficientNetB0 brain tumor classifier.
"""

import os
from typing import Dict, Any, Optional, Union
import numpy as np
import cv2
import tensorflow as tf

from .gradcam import GradCAM
from .overlay_utils import create_professional_overlay, create_side_by_side, apply_colormap
from ..loader import ModelLoader
from ..utils import load_image, preprocess_brain_mri
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrainMRIGradCAM:
    """
    Production-ready Grad-CAM for brain MRI tumor classification.
    
    Features:
    - Professional overlays with prediction labels
    - Multiple colormap options
    - Error handling for production deployment
    
    Classes: glioma, meningioma, notumor, pituitary
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loader = ModelLoader(models_dir)
        
        self.model = None
        self.gradcam = None
        self.class_labels = None
        
        self._loaded = False
    
    def load(self):
        """Load model and initialize Grad-CAM."""
        if self._loaded:
            return
        
        logger.info("Loading Brain MRI model for Grad-CAM...")
        try:
            self.model = self.loader.load_model("brain_mri")
            self.class_labels = self.loader.load_class_labels("brain_mri")
            self.gradcam = GradCAM(self.model)
            self._loaded = True
            logger.info(f"Brain MRI Grad-CAM ready. Target layer: {self.gradcam.target_layer}")
        except Exception as e:
            logger.error(f"Failed to load Brain MRI model: {e}")
            raise
    
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
        
        Args:
            image_path: Path to MRI image
            class_idx: Target class (auto-detected if None)
            save_dir: Directory to save outputs
            colormap: Colormap ('turbo', 'jet', 'viridis', etc.)
            add_labels: Add prediction labels to overlay
            output_size: Optional output dimensions (width, height)
            
        Returns:
            Dictionary with prediction and Grad-CAM outputs
        """
        if not self._loaded:
            self.load()
        
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load original image at full resolution
            original_image = load_image(image_path)
            original_h, original_w = original_image.shape[:2]
            
            # Preprocess for model (224x224)
            preprocessed = preprocess_brain_mri(image_path)
            
            # Get prediction
            predictions = self.model.predict(preprocessed, verbose=0)[0]
            if class_idx is None:
                class_idx = int(np.argmax(predictions))
            
            predicted_class = self.class_labels["classes"][class_idx]
            confidence = float(predictions[class_idx])
            
            # Generate heatmap
            heatmap = self.gradcam.generate_heatmap(preprocessed, class_idx)
            
            # Create professional overlay at original resolution
            overlay = create_professional_overlay(
                original_image,
                heatmap,
                predicted_class,
                confidence,
                colormap=colormap,
                add_label=add_labels,
                add_bar=add_labels,
                output_size=output_size
            )
            
            # Resize heatmap for output
            heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
            heatmap_colored = apply_colormap(heatmap_resized, colormap)
            
            # Create side-by-side comparison
            side_by_side = create_side_by_side(
                original_image, heatmap_colored, overlay,
                predicted_class, confidence
            )
            
            result = {
                "disease_type": "brain_mri",
                "image_path": image_path,
                "prediction": {
                    "class": predicted_class,
                    "class_id": class_idx,
                    "confidence": confidence,
                    "probabilities": {
                        name: float(prob) 
                        for name, prob in zip(self.class_labels["classes"], predictions)
                    }
                },
                "gradcam": {
                    "target_layer": self.gradcam.target_layer,
                    "heatmap": heatmap,
                    "overlay": overlay,
                    "colormap": colormap
                }
            }
            
            # Save outputs
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save professional overlay
                overlay_path = os.path.join(save_dir, f"{basename}_{predicted_class}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                # Save heatmap
                heatmap_path = os.path.join(save_dir, f"{basename}_{predicted_class}_heatmap.png")
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
                
                # Save side-by-side
                comparison_path = os.path.join(save_dir, f"{basename}_{predicted_class}_comparison.png")
                cv2.imwrite(comparison_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
                
                result["gradcam"]["overlay_path"] = overlay_path
                result["gradcam"]["heatmap_path"] = heatmap_path
                result["gradcam"]["comparison_path"] = comparison_path
                
                logger.info(f"Brain MRI Grad-CAM saved: {predicted_class} ({confidence:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {e}")
            raise RuntimeError(f"Grad-CAM generation failed: {e}") from e
