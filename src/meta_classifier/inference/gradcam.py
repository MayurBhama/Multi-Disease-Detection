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
from src.utils.exception import PredictionError

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
        try:
            target_layer = self._get_layer(self.target_layer)
            
            # For functional models, try direct approach first
            if hasattr(self.model, 'input') and hasattr(target_layer, 'output'):
                try:
                    self.gradient_model = tf.keras.Model(
                        inputs=self.model.input,
                        outputs=[target_layer.output, self.model.output]
                    )
                    logger.debug("Built gradient model using direct layer access")
                    return
                except ValueError as e:
                    logger.debug(f"Direct approach failed: {e}, trying alternative method")
            
            # Alternative: Build a new model that traces through layers
            # This is needed for models with preprocessing layers or complex nesting
            input_shape = None
            if hasattr(self.model, 'input_shape'):
                input_shape = self.model.input_shape[1:]  # Remove batch dim
            elif hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                first_layer = self.model.layers[0]
                if hasattr(first_layer, 'input_shape'):
                    input_shape = first_layer.input_shape[1:]
            
            if input_shape is None:
                input_shape = (224, 224, 3)  # Default
            
            # Create a fresh forward pass to get intermediate outputs
            inputs = tf.keras.Input(shape=input_shape)
            target_output = None
            x = inputs
            
            for layer in self.model.layers:
                # Apply the layer
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue  # Skip input layers
                
                try:
                    x = layer(x)
                except Exception:
                    # Some layers may need special handling
                    continue
                
                # Check if this layer or any nested layer is our target
                if layer.name == self.target_layer:
                    target_output = x
                elif hasattr(layer, 'layers'):
                    # Search in nested models (e.g., EfficientNet base)
                    for sub_layer in layer.layers:
                        if sub_layer.name == self.target_layer:
                            # Get the output at this layer from the nested model
                            try:
                                intermediate_model = tf.keras.Model(
                                    inputs=layer.input,
                                    outputs=layer.get_layer(self.target_layer).output
                                )
                                target_output = intermediate_model(layer.input)
                            except Exception:
                                pass
            
            if target_output is None:
                # Last resort: use the last conv layer output we can find
                for layer in reversed(self.model.layers):
                    if hasattr(layer, 'layers'):
                        for sub_layer in reversed(layer.layers):
                            if isinstance(sub_layer, tf.keras.layers.Conv2D):
                                self.target_layer = sub_layer.name
                                try:
                                    intermediate_model = tf.keras.Model(
                                        inputs=layer.input,
                                        outputs=sub_layer.output
                                    )
                                    # Rebuild with correct layer
                                    self.gradient_model = tf.keras.Model(
                                        inputs=self.model.input,
                                        outputs=[sub_layer.output, self.model.output]
                                    )
                                    logger.debug(f"Using fallback target layer: {self.target_layer}")
                                    return
                                except Exception:
                                    continue
                
                raise ValueError(f"Could not find target layer output for: {self.target_layer}")
            
            self.gradient_model = tf.keras.Model(
                inputs=inputs,
                outputs=[target_output, x]
            )
            logger.debug("Built gradient model using layer tracing")
            
        except Exception as e:
            logger.warning(f"Could not build gradient model: {e}")
            # Create a simple fallback that just returns model output
            self.gradient_model = None
    
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
            
        Raises:
            PredictionError: If heatmap generation fails
        """
        try:
            # Check if gradient model was built successfully
            if self.gradient_model is None:
                logger.warning("Gradient model not available, returning uniform heatmap")
                # Return a uniform heatmap as fallback
                if len(image.shape) == 3:
                    h, w = image.shape[:2]
                else:
                    h, w = image.shape[1:3]
                return np.ones((7, 7)) * 0.5  # Default 7x7 uniform heatmap
            
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
            
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
            raise PredictionError(f"Grad-CAM heatmap generation failed: {e}")
    
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
