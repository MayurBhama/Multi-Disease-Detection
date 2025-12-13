# src/meta_classifier/loader.py
"""
Model and Configuration Loaders
===============================
Handles loading of trained models and their configuration files.
Implements caching to ensure models load only once.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.exception import ModelLoadError, ConfigurationError

logger = get_logger(__name__)


# =====================================================
# CONFIGURATION
# =====================================================
MODELS_DIR = "models"

MODEL_CONFIGS = {
    "brain_mri": {
        "model_file": "brain_mri_efficientnetb0.weights.h5",
        "architecture": "efficientnetb0",
        "input_shape": (224, 224, 3),
        "num_classes": 4,  # glioma, meningioma, notumor, pituitary
    },
    "pneumonia": {
        "model_file": "pneumonia_xception.weights.h5",
        "architecture": "xception",
        "input_shape": (256, 256, 3),
        "num_classes": 2,
    },
    "retina": {
        "models": [
            {"file": "efficientnet_v2s.weights.h5", "architecture": "efficientnetv2s"},
            {"file": "efficientnet_b2.weights.h5", "architecture": "efficientnetb2"},
            {"file": "efficientnet_b0.weights.h5", "architecture": "efficientnetb0"},
        ],
        "input_shape": (224, 224, 3),
        "num_classes": 5,
        "ensemble_weights": [0.333, 0.329, 0.338],
    },
}


# =====================================================
# MODEL CACHE (SINGLETON)
# =====================================================
class ModelCache:
    """Singleton cache for loaded models."""
    
    _instance = None
    _models: Dict[str, tf.keras.Model] = {}
    _configs: Dict[str, Dict] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._models = {}
            cls._configs = {}
        return cls._instance
    
    def get_model(self, key: str) -> Optional[tf.keras.Model]:
        return self._models.get(key)
    
    def set_model(self, key: str, model: tf.keras.Model) -> None:
        self._models[key] = model
    
    def has_model(self, key: str) -> bool:
        return key in self._models
    
    def get_config(self, key: str) -> Optional[Dict]:
        return self._configs.get(key)
    
    def set_config(self, key: str, config: Dict) -> None:
        self._configs[key] = config
    
    def clear(self) -> None:
        """Clear all cached models and configs."""
        try:
            self._models.clear()
            self._configs.clear()
            tf.keras.backend.clear_session()
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")


# Global cache instance
_cache = ModelCache()


# =====================================================
# ARCHITECTURE BUILDERS
# =====================================================
def _build_brain_mri_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Build Brain MRI model architecture (EfficientNetB0 + Conv2D)."""
    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        
        inputs = tf.keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        
        return tf.keras.Model(inputs, outputs, name="brain_mri_efficientnetb0")
    except Exception as e:
        logger.error(f"Failed to build brain_mri model: {e}")
        raise ModelLoadError(f"Failed to build brain_mri architecture: {e}")


def _build_pneumonia_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Build Pneumonia model architecture (Xception Sequential)."""
    try:
        base_model = tf.keras.applications.Xception(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=input_shape
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ], name="pneumonia_xception")
        
        return model
    except Exception as e:
        logger.error(f"Failed to build pneumonia model: {e}")
        raise ModelLoadError(f"Failed to build pneumonia architecture: {e}")


def _build_efficientnet_model(
    architecture: str, 
    input_shape: Tuple[int, int, int], 
    num_classes: int
) -> tf.keras.Model:
    """Build EfficientNet model for retina ensemble."""
    
    arch_map = {
        "efficientnetv2s": tf.keras.applications.EfficientNetV2S,
        "efficientnetb2": tf.keras.applications.EfficientNetB2,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
    }
    
    if architecture not in arch_map:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    try:
        inputs = tf.keras.Input(shape=input_shape)
        base = arch_map[architecture](
            include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
        )
        
        x = tf.keras.layers.Dropout(0.3)(base.output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        
        return tf.keras.Model(inputs, outputs, name=f"retina_{architecture}")
    except Exception as e:
        logger.error(f"Failed to build {architecture} model: {e}")
        raise ModelLoadError(f"Failed to build {architecture} architecture: {e}")


# =====================================================
# MODEL LOADER CLASS
# =====================================================
class ModelLoader:
    """
    Model and configuration loader with caching.
    
    Usage:
        loader = ModelLoader()
        model = loader.load_model("brain_mri")
        labels = loader.load_class_labels("brain_mri")
    """
    
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.cache = _cache
    
    def load_model(self, disease_type: str, model_key: Optional[str] = None) -> tf.keras.Model:
        """
        Load a trained model with caching.
        
        Args:
            disease_type: One of "brain_mri", "pneumonia", "retina"
            model_key: For retina ensemble, specify which model
        
        Returns:
            Loaded Keras model with weights
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if disease_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown disease type: {disease_type}. Must be one of {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[disease_type]
        cache_key = f"{disease_type}_{model_key}" if model_key else disease_type
        
        # Check cache
        if self.cache.has_model(cache_key):
            logger.debug(f"Using cached model: {cache_key}")
            return self.cache.get_model(cache_key)
        
        logger.info(f"Loading model: {cache_key}")
        
        try:
            # Build architecture
            if disease_type == "brain_mri":
                model = _build_brain_mri_model(config["input_shape"], config["num_classes"])
                weights_path = os.path.join(self.models_dir, disease_type, config["model_file"])
            
            elif disease_type == "pneumonia":
                model = _build_pneumonia_model(config["input_shape"], config["num_classes"])
                weights_path = os.path.join(self.models_dir, disease_type, config["model_file"])
            
            elif disease_type == "retina":
                if model_key is None:
                    raise ValueError("For retina, specify model_key (efficientnetv2s, efficientnetb2, efficientnetb0)")
                
                model_info = next((m for m in config["models"] if m["architecture"] == model_key), None)
                if not model_info:
                    raise ValueError(f"Unknown retina model: {model_key}")
                
                model = _build_efficientnet_model(model_key, config["input_shape"], config["num_classes"])
                weights_path = os.path.join(self.models_dir, disease_type, model_info["file"])
            
            else:
                raise ValueError(f"Unknown disease type: {disease_type}")
            
            # Load weights
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found: {weights_path}")
            
            model.load_weights(weights_path)
            logger.info(f"  Loaded weights: {weights_path}")
            logger.info(f"  Parameters: {model.count_params():,}")
            
            # Cache model
            self.cache.set_model(cache_key, model)
            
            return model
            
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model {cache_key}: {e}")
            raise ModelLoadError(f"Failed to load model {cache_key}: {e}")
    
    def load_class_labels(self, disease_type: str) -> Dict[str, Any]:
        """Load class labels from class_labels.json."""
        cache_key = f"{disease_type}_labels"
        
        if self.cache.get_config(cache_key):
            return self.cache.get_config(cache_key)
        
        labels_path = os.path.join(self.models_dir, disease_type, "class_labels.json")
        
        try:
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Class labels not found: {labels_path}")
            
            with open(labels_path, "r") as f:
                labels = json.load(f)
            
            self.cache.set_config(cache_key, labels)
            logger.debug(f"Loaded class labels: {labels_path}")
            
            return labels
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {labels_path}: {e}")
            raise ConfigurationError(f"Invalid class labels file: {e}")
        except Exception as e:
            logger.error(f"Failed to load class labels: {e}")
            raise ConfigurationError(f"Failed to load class labels: {e}")
    
    def load_preprocessing_config(self, disease_type: str) -> Dict[str, Any]:
        """Load preprocessing config from preprocessing.json."""
        cache_key = f"{disease_type}_preprocessing"
        
        if self.cache.get_config(cache_key):
            return self.cache.get_config(cache_key)
        
        config_path = os.path.join(self.models_dir, disease_type, "preprocessing.json")
        
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Preprocessing config not found: {config_path}")
            
            with open(config_path, "r") as f:
                config = json.load(f)
            
            self.cache.set_config(cache_key, config)
            logger.debug(f"Loaded preprocessing config: {config_path}")
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_path}: {e}")
            raise ConfigurationError(f"Invalid preprocessing config: {e}")
        except Exception as e:
            logger.error(f"Failed to load preprocessing config: {e}")
            raise ConfigurationError(f"Failed to load preprocessing config: {e}")
    
    def get_model_config(self, disease_type: str) -> Dict[str, Any]:
        """Get model configuration."""
        if disease_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown disease type: {disease_type}")
        return MODEL_CONFIGS[disease_type]
    
    def clear_cache(self) -> None:
        """Clear all cached models and configs."""
        self.cache.clear()
        logger.info("Model cache cleared")
