# src/inference/utils.py
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
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image from bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =====================================================
# PREPROCESSING FUNCTIONS
# =====================================================
def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def crop_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop black borders from retina images.
    
    Args:
        img: Input RGB image
        threshold: Pixel value threshold for black detection
        
    Returns:
        Cropped image
    """
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


def apply_efficientnet_preprocessing(img: np.ndarray, version: str = "v1") -> np.ndarray:
    """
    Apply EfficientNet-specific preprocessing.
    
    Args:
        img: Input image (0-255 range)
        version: "v1" for B-series, "v2" for V2 series
        
    Returns:
        Preprocessed image
    """
    img = img.astype(np.float32)
    
    if version == "v2":
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    else:
        img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    return img


def apply_xception_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    Apply Xception-specific preprocessing.
    
    Args:
        img: Input image (0-255 range)
        
    Returns:
        Preprocessed image
    """
    img = img.astype(np.float32)
    img = tf.keras.applications.xception.preprocess_input(img)
    return img


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
    """
    img = load_image(image_path)
    img = resize_image(img, target_size)
    img = apply_efficientnet_preprocessing(img, version="v1")
    
    # Add batch dimension
    return np.expand_dims(img, axis=0)


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
    """
    img = load_image(image_path)
    img = resize_image(img, target_size)
    img = apply_xception_preprocessing(img)
    
    # Add batch dimension
    return np.expand_dims(img, axis=0)


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
    """
    img = load_image(image_path)
    
    # Crop black borders (common in fundus images)
    img = crop_black_borders(img)
    
    # Resize
    img = resize_image(img, target_size)
    
    # Apply preprocessing
    img = apply_efficientnet_preprocessing(img, version=version)
    
    # Add batch dimension
    return np.expand_dims(img, axis=0)


# =====================================================
# OUTPUT FORMATTING
# =====================================================
def format_probabilities(
    probabilities: np.ndarray,
    class_names: list
) -> Dict[str, float]:
    """
    Format probabilities as class_name: probability dict.
    
    Args:
        probabilities: Array of probabilities
        class_names: List of class names
        
    Returns:
        Dictionary mapping class names to probabilities
    """
    if len(probabilities.shape) > 1:
        probabilities = probabilities.flatten()
    
    return {
        name: round(float(prob), 6)
        for name, prob in zip(class_names, probabilities)
    }


def get_prediction_result(
    probabilities: np.ndarray,
    class_names: list
) -> Tuple[str, int, float]:
    """
    Get prediction result from probabilities.
    
    Args:
        probabilities: Array of probabilities
        class_names: List of class names
        
    Returns:
        Tuple of (predicted_class, class_id, confidence)
    """
    if len(probabilities.shape) > 1:
        probabilities = probabilities.flatten()
    
    class_id = int(np.argmax(probabilities))
    predicted_class = class_names[class_id]
    confidence = float(probabilities[class_id])
    
    return predicted_class, class_id, confidence


def sigmoid_to_binary(
    probability: float,
    threshold: float = 0.5,
    class_names: list = ["NORMAL", "PNEUMONIA"]
) -> Tuple[str, int, float, Dict[str, float]]:
    """
    Convert sigmoid output to binary classification result.
    
    Args:
        probability: Sigmoid output (probability of positive class)
        threshold: Classification threshold
        class_names: [negative_class, positive_class]
        
    Returns:
        Tuple of (predicted_class, class_id, confidence, probabilities_dict)
    """
    class_id = 1 if probability >= threshold else 0
    predicted_class = class_names[class_id]
    
    # For binary, confidence is the probability of the predicted class
    confidence = probability if class_id == 1 else (1 - probability)
    
    probabilities_dict = {
        class_names[0]: round(1 - float(probability), 6),
        class_names[1]: round(float(probability), 6)
    }
    
    return predicted_class, class_id, float(confidence), probabilities_dict
