# src/meta_classifier/inference/gradcam.py
"""
Universal Grad-CAM Engine
=========================
Works with any Keras/TensorFlow model to generate visual explanations.

Features:
- Auto-detect last convolutional layer
- Generate raw heatmaps
- Overlay heatmaps on original images
- Batch processing support
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import cv2
import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """
    Universal Grad-CAM implementation for Keras models.
    
    Example:
        model = load_model(...)
        gradcam = GradCAM(model)
        heatmap = gradcam.generate_heatmap(image, class_idx=0)
        overlay = gradcam.overlay_heatmap(image, heatmap)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        target_layer: Optional[str] = None,
        use_guided_gradients: bool = False
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            target_layer: Name of target conv layer (auto-detected if None)
            use_guided_gradients: Use guided backpropagation
        """
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.use_guided_gradients = use_guided_gradients
        
        # Create gradient model
        self._build_gradient_model()
        
        logger.debug(f"GradCAM initialized with target layer: {self.target_layer}")
    
    def _find_target_layer(self) -> str:
        """
        Auto-detect the last convolutional layer.
        
        Searches for Conv2D layers and returns the last one found.
        Handles nested models (e.g., functional API with base model).
        """
        conv_layers = []
        
        def find_conv_layers(layer):
            """Recursively find conv layers."""
            if hasattr(layer, 'layers'):
                # This is a nested model/layer
                for sub_layer in layer.layers:
                    find_conv_layers(sub_layer)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer.name)
        
        # Search through all layers
        for layer in self.model.layers:
            find_conv_layers(layer)
        
        if not conv_layers:
            raise ValueError("No Conv2D layers found in model")
        
        target = conv_layers[-1]
        logger.debug(f"Auto-detected target layer: {target}")
        return target
    
    def _get_layer(self, layer_name: str):
        """Get layer by name, handling nested models."""
        def search_layer(model, name):
            try:
                return model.get_layer(name)
            except ValueError:
                # Search in nested models
                for layer in model.layers:
                    if hasattr(layer, 'layers'):
                        try:
                            return search_layer(layer, name)
                        except ValueError:
                            continue
                raise ValueError(f"Layer {name} not found")
        
        return search_layer(self.model, layer_name)
    
    def _build_gradient_model(self):
        """Build model for gradient computation."""
        # For Sequential models, we need to ensure it's built
        if isinstance(self.model, tf.keras.Sequential):
            if not self.model.built:
                # Build the model with a dummy input
                # Get input shape from first layer
                first_layer = self.model.layers[0]
                if hasattr(first_layer, 'input_shape'):
                    input_shape = first_layer.input_shape
                    if input_shape and input_shape[0] is None:
                        input_shape = input_shape[1:]
                else:
                    # Default shape for Xception
                    input_shape = (256, 256, 3)
                
                dummy_input = tf.zeros((1, *input_shape))
                _ = self.model(dummy_input)
        
        target_layer = self._get_layer(self.target_layer)
        
        # For Sequential models, create a new functional model
        if isinstance(self.model, tf.keras.Sequential):
            # Get the input shape from the first layer
            input_shape = self.model.layers[0].input_shape[1:]
            inputs = tf.keras.Input(shape=input_shape)
            
            # Build forward through all layers, capturing target layer output
            x = inputs
            target_output = None
            for layer in self.model.layers:
                x = layer(x)
                if layer.name == self.target_layer:
                    target_output = x
                # For nested models (like Xception base), search inside
                if hasattr(layer, 'layers'):
                    for sub_layer in layer.layers:
                        if sub_layer.name == self.target_layer:
                            # Get output from nested model at this layer
                            target_output = layer.get_layer(self.target_layer).output
            
            if target_output is None:
                # Target layer is inside a nested model
                # Rebuild with proper tracking
                x = inputs
                for layer in self.model.layers:
                    if hasattr(layer, 'layers') and any(l.name == self.target_layer for l in layer.layers):
                        # This is the nested model containing our target
                        nested_target = layer.get_layer(self.target_layer)
                        # Create intermediate model
                        nested_model = tf.keras.Model(
                            inputs=layer.input,
                            outputs=[nested_target.output, layer.output]
                        )
                        target_output, x = nested_model(x)
                    else:
                        x = layer(x)
            
            self.gradient_model = tf.keras.Model(
                inputs=inputs,
                outputs=[target_output, x]
            )
        else:
            # Standard functional model
            self.gradient_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
    
    @tf.function
    def _compute_gradients(
        self,
        image: tf.Tensor,
        class_idx: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute gradients of class output w.r.t. conv layer activations.
        
        Args:
            image: Preprocessed input image (1, H, W, C)
            class_idx: Target class index
            
        Returns:
            Tuple of (conv_outputs, gradients)
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.gradient_model(image)
            
            # Handle both softmax and sigmoid outputs
            if predictions.shape[-1] == 1:
                # Sigmoid (binary classification)
                if class_idx == 1:
                    loss = predictions[0, 0]
                else:
                    loss = 1 - predictions[0, 0]
            else:
                # Softmax (multi-class)
                loss = predictions[0, class_idx]
        
        gradients = tape.gradient(loss, conv_outputs)
        return conv_outputs, gradients
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Preprocessed image (1, H, W, C) or (H, W, C)
            class_idx: Target class (uses predicted class if None)
            normalize: Normalize heatmap to [0, 1]
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        image_tensor = tf.cast(image, tf.float32)
        
        # Get predicted class if not specified
        if class_idx is None:
            predictions = self.model.predict(image, verbose=0)
            if predictions.shape[-1] == 1:
                class_idx = 1 if predictions[0, 0] >= 0.5 else 0
            else:
                class_idx = np.argmax(predictions[0])
        
        # Compute gradients
        conv_outputs, gradients = self._compute_gradients(image_tensor, class_idx)
        
        # Global average pooling of gradients
        weights = tf.reduce_mean(gradients, axis=(1, 2))
        
        # Weighted combination of conv outputs
        conv_outputs = conv_outputs[0]
        weights = weights[0]
        
        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
        
        # ReLU to keep only positive contributions
        heatmap = tf.maximum(heatmap, 0)
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Normalize
        if normalize and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: Original image (H, W, C) in RGB, 0-255 range
            heatmap: Grad-CAM heatmap (h, w)
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Overlay image (H, W, C) in RGB, 0-255 range
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(
            heatmap, 
            (original_image.shape[1], original_image.shape[0])
        )
        
        # Convert to uint8
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap (outputs BGR)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Convert to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure original is uint8
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(
            original_image, 1 - alpha,
            heatmap_colored, alpha,
            0
        )
        
        return overlay
    
    def explain(
        self,
        image: np.ndarray,
        original_image: np.ndarray,
        class_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete explanation with heatmap and overlay.
        
        Args:
            image: Preprocessed image for model
            original_image: Original image for overlay
            class_idx: Target class
            save_path: Path to save overlay (without extension)
            
        Returns:
            Dictionary with heatmap, overlay, and paths
        """
        heatmap = self.generate_heatmap(image, class_idx)
        overlay = self.overlay_heatmap(original_image, heatmap)
        
        result = {
            "heatmap": heatmap,
            "overlay": overlay,
            "target_layer": self.target_layer
        }
        
        if save_path:
            # Save heatmap
            heatmap_path = f"{save_path}_heatmap.png"
            heatmap_img = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(
                cv2.resize(heatmap_img, (original_image.shape[1], original_image.shape[0])),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(heatmap_path, heatmap_colored)
            
            # Save overlay
            overlay_path = f"{save_path}_overlay.png"
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            result["heatmap_path"] = heatmap_path
            result["overlay_path"] = overlay_path
            
            logger.debug(f"Saved Grad-CAM to {save_path}")
        
        return result


def find_target_layer(model: tf.keras.Model) -> str:
    """Utility function to find target conv layer."""
    gradcam = GradCAM(model)
    return gradcam.target_layer
