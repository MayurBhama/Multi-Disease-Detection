# verify_models.py
"""Verify all trained models are correctly saved and loadable."""

import os
import sys
import json
import tensorflow as tf
from src.utils.logger import get_logger

logger = get_logger(__name__)

def verify_brain_mri():
    """Verify brain MRI model - uses EfficientNetB0 with Conv2D head."""
    logger.info("=" * 50)
    logger.info("VERIFYING: Brain MRI Model")
    logger.info("=" * 50)
    
    model_path = "models/brain_mri/brain_mri_xception.weights.h5"
    labels_path = "models/brain_mri/class_labels.json"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return False
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    logger.info(f"  Model file: {model_path}")
    logger.info(f"  Size: {os.path.getsize(model_path) / 1e6:.2f} MB")
    logger.info(f"  Classes: {labels['classes']}")
    
    # Build model matching train_brain.py architecture
    try:
        tf.keras.backend.clear_session()
        IMAGE_SIZE = (224, 224)
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3)
        )
        
        inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.load_weights(model_path)
        logger.info(f"  ✓ Model loaded successfully!")
        logger.info(f"  Parameters: {model.count_params():,}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed to load: {e}")
        return False


def verify_pneumonia():
    """Verify pneumonia model - uses Xception Sequential with 256x256 input."""
    logger.info("=" * 50)
    logger.info("VERIFYING: Pneumonia Model")
    logger.info("=" * 50)
    
    model_path = "models/pneumonia/pneumonia_xception.weights.h5"
    labels_path = "models/pneumonia/class_labels.json"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return False
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    logger.info(f"  Model file: {model_path}")
    logger.info(f"  Size: {os.path.getsize(model_path) / 1e6:.2f} MB")
    logger.info(f"  Classes: {labels['classes']}")
    
    # Build model matching train_pneumonia.py architecture
    try:
        tf.keras.backend.clear_session()
        base_model = tf.keras.applications.Xception(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(256, 256, 3)
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        
        model.load_weights(model_path)
        logger.info(f"  ✓ Model loaded successfully!")
        logger.info(f"  Parameters: {model.count_params():,}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed to load: {e}")
        return False


def verify_retina():
    """Verify retina ensemble models."""
    logger.info("=" * 50)
    logger.info("VERIFYING: Retina Ensemble Models")
    logger.info("=" * 50)
    
    labels_path = "models/retina/class_labels.json"
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    logger.info(f"  Classes: {labels['classes']}")
    
    models_info = [
        ("efficientnet_v2s.weights.h5", tf.keras.applications.EfficientNetV2S, 5),
        ("efficientnet_b2.weights.h5", tf.keras.applications.EfficientNetB2, 5),
        ("efficientnet_b0.weights.h5", tf.keras.applications.EfficientNetB0, 5),
    ]
    
    all_success = True
    for model_file, base_class, num_classes in models_info:
        model_path = f"models/retina/{model_file}"
        
        if not os.path.exists(model_path):
            logger.error(f"  ✗ Model not found: {model_path}")
            all_success = False
            continue
        
        logger.info(f"\n  Model: {model_file}")
        logger.info(f"  Size: {os.path.getsize(model_path) / 1e6:.2f} MB")
        
        try:
            tf.keras.backend.clear_session()
            inputs = tf.keras.Input(shape=(224, 224, 3))
            base = base_class(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
            x = tf.keras.layers.Dropout(0.3)(base.output)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)
            model.load_weights(model_path)
            logger.info(f"  ✓ Model loaded successfully!")
            logger.info(f"  Parameters: {model.count_params():,}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load: {e}")
            all_success = False
    
    return all_success


def main():
    logger.info("=" * 60)
    logger.info("MODEL VERIFICATION")
    logger.info("=" * 60)
    
    results = {}
    results['brain_mri'] = verify_brain_mri()
    results['pneumonia'] = verify_pneumonia()
    results['retina'] = verify_retina()
    
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    for name, status in results.items():
        icon = "✓" if status else "✗"
        logger.info(f"  {icon} {name}: {'PASSED' if status else 'FAILED'}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n  *** ALL MODELS VERIFIED SUCCESSFULLY! ***")
    else:
        logger.info("\n  *** SOME MODELS FAILED VERIFICATION ***")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
