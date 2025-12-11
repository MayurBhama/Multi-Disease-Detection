# src/meta_classifier/utils.py
"""
Image Preprocessing Utilities
=============================
Disease-specific image preprocessing for inference.
"""

import os
from typing import Dict, Any, Tuple, Union
import numpy as np
import cv2
import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.exception import PreprocessingError

logger = get_logger(__name__)


# =====================================================
# IMAGE LOADING
# =====================================================
def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB image as numpy array (H, W, 3)
        
    Raises:
        PreprocessingError: If image loading fails
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
        
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise PreprocessingError(f"Failed to load image: {e}")


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image from bytes."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image from bytes")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Failed to decode image from bytes: {e}")
        raise PreprocessingError(f"Failed to decode image: {e}")


# =====================================================
# PREPROCESSING FUNCTIONS
# =====================================================
def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    try:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        raise PreprocessingError(f"Failed to resize image: {e}")


def crop_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop black borders from retina images.
    
    Args:
        img: Input RGB image
        threshold: Pixel value threshold for black detection
        
    Returns:
        Cropped image
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 5
            x, y = max(0, x - margin), max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            img = img[y:y+h, x:x+w]
        
        return img
        
    except Exception as e:
        logger.warning(f"Failed to crop borders, returning original: {e}")
        return img  # Return original if cropping fails


def apply_efficientnet_preprocessing(img: np.ndarray, version: str = "v1") -> np.ndarray:
    """
    Apply EfficientNet-specific preprocessing.
    
    Args:
        img: Input image (0-255 range)
        version: "v1" for B-series, "v2" for V2 series
        
    Returns:
        Preprocessed image
    """
    try:
        img = img.astype(np.float32)
        
        if version == "v2":
            img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        else:
            img = tf.keras.applications.efficientnet.preprocess_input(img)
        
        return img
        
    except Exception as e:
        logger.error(f"EfficientNet preprocessing failed: {e}")
        raise PreprocessingError(f"Preprocessing failed: {e}")


def apply_xception_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    Apply Xception-specific preprocessing.
    
    Args:
        img: Input image (0-255 range)
        
    Returns:
        Preprocessed image
    """
    try:
        img = img.astype(np.float32)
        img = tf.keras.applications.xception.preprocess_input(img)
        return img
    except Exception as e:
        logger.error(f"Xception preprocessing failed: {e}")
        raise PreprocessingError(f"Preprocessing failed: {e}")


# =====================================================
# DISEASE-SPECIFIC PREPROCESSING
# =====================================================
def preprocess_brain_mri(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess Brain MRI image for EfficientNetB0.
    
    Args:
        image_path: Path to image file
        target_size: Target image dimensions
        
    Returns:
        Preprocessed image batch (1, H, W, 3)
        
    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        img = load_image(image_path)
        img = resize_image(img, target_size)
        img = apply_efficientnet_preprocessing(img, version="v1")
        
        # Add batch dimension
        return np.expand_dims(img, axis=0)
        
    except PreprocessingError:
        raise
    except Exception as e:
        logger.error(f"Brain MRI preprocessing failed: {e}")
        raise PreprocessingError(f"Brain MRI preprocessing failed: {e}")


def preprocess_pneumonia(
    image_path: str,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Preprocess chest X-ray image for Xception.
    
    Args:
        image_path: Path to image file
        target_size: Target image dimensions
        
    Returns:
        Preprocessed image batch (1, H, W, 3)
        
    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        img = load_image(image_path)
        img = resize_image(img, target_size)
        img = apply_xception_preprocessing(img)
        
        # Add batch dimension
        return np.expand_dims(img, axis=0)
        
    except PreprocessingError:
        raise
    except Exception as e:
        logger.error(f"Pneumonia preprocessing failed: {e}")
        raise PreprocessingError(f"Pneumonia preprocessing failed: {e}")


def preprocess_retina(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    version: str = "v1"
) -> np.ndarray:
    """
    Preprocess retina fundus image for EfficientNet.
    
    Args:
        image_path: Path to image file
        target_size: Target image dimensions
        version: "v1" for B-series, "v2" for V2-S
        
    Returns:
        Preprocessed image batch (1, H, W, 3)
        
    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        img = load_image(image_path)
        
        # Crop black borders (common in fundus images)
        img = crop_black_borders(img)
        
        # Resize
        img = resize_image(img, target_size)
        
        # Apply preprocessing
        img = apply_efficientnet_preprocessing(img, version=version)
        
        # Add batch dimension
        return np.expand_dims(img, axis=0)
        
    except PreprocessingError:
        raise
    except Exception as e:
        logger.error(f"Retina preprocessing failed: {e}")
        raise PreprocessingError(f"Retina preprocessing failed: {e}")


# =====================================================
# OUTPUT FORMATTING
# =====================================================
def format_probabilities(
    probabilities: np.ndarray,
    class_names: list
) -> Dict[str, float]:
    """Format probabilities as class_name: probability dict."""
    try:
        if len(probabilities.shape) > 1:
            probabilities = probabilities.flatten()
        
        return {
            name: round(float(prob), 6)
            for name, prob in zip(class_names, probabilities)
        }
    except Exception as e:
        logger.error(f"Failed to format probabilities: {e}")
        return {}


def get_prediction_result(
    probabilities: np.ndarray,
    class_names: list
) -> Tuple[str, int, float]:
    """Get prediction result from probabilities."""
    try:
        if len(probabilities.shape) > 1:
            probabilities = probabilities.flatten()
        
        class_id = int(np.argmax(probabilities))
        predicted_class = class_names[class_id]
        confidence = float(probabilities[class_id])
        
        return predicted_class, class_id, confidence
        
    except Exception as e:
        logger.error(f"Failed to get prediction result: {e}")
        raise PreprocessingError(f"Failed to process prediction: {e}")


def sigmoid_to_binary(
    probability: float,
    threshold: float = 0.5,
    class_names: list = ["NORMAL", "PNEUMONIA"]
) -> Tuple[str, int, float, Dict[str, float]]:
    """Convert sigmoid output to binary classification result."""
    try:
        class_id = 1 if probability >= threshold else 0
        predicted_class = class_names[class_id]
        
        confidence = probability if class_id == 1 else (1 - probability)
        
        probabilities_dict = {
            class_names[0]: round(1 - float(probability), 6),
            class_names[1]: round(float(probability), 6)
        }
        
        return predicted_class, class_id, float(confidence), probabilities_dict
        
    except Exception as e:
        logger.error(f"Failed to convert sigmoid: {e}")
        raise PreprocessingError(f"Failed to process binary prediction: {e}")
