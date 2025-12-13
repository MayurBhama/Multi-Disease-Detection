"""
Multi-Dataset Brain MRI Training Script
========================================
Trains EfficientNetB0 on combined 5-class brain tumor dataset.

Classes: glioma, meningioma, notumor, pituitary, other_tumor

Run: python scripts/train_multi_dataset.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
from pathlib import Path
import json
from datetime import datetime

# Configuration
CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "num_classes": 5,
    "classes": ["glioma", "meningioma", "notumor", "pituitary", "other_tumor"]
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "combined_brain"
MODEL_DIR = PROJECT_ROOT / "models" / "brain_mri_v2"
LOG_DIR = PROJECT_ROOT / "logs" / "brain_mri_v2"


def create_data_augmentation():
    """Create data augmentation pipeline for robust training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.3),  # ±30 degrees
        layers.RandomZoom(0.2),      # ±20% zoom
        layers.RandomContrast(0.2),  # ±20% contrast
        layers.RandomBrightness(0.2),  # ±20% brightness
    ], name="data_augmentation")


def load_datasets():
    """Load train and test datasets."""
    print("[1/4] Loading datasets...")
    
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"
    
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_dir}\n"
            "Run: python scripts/prepare_multi_dataset.py first"
        )
    
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(CONFIG["img_size"], CONFIG["img_size"]),
        batch_size=CONFIG["batch_size"],
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(CONFIG["img_size"], CONFIG["img_size"]),
        batch_size=CONFIG["batch_size"],
        label_mode="categorical",
        shuffle=False
    )
    
    # Get class names from directory
    class_names = train_ds.class_names
    print(f"  Classes: {class_names}")
    print(f"  Train batches: {len(train_ds)}")
    print(f"  Test batches: {len(test_ds)}")
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, test_ds, class_names


def build_model():
    """Build EfficientNetB0 model with custom head."""
    print("[2/4] Building model...")
    
    # Base model
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(CONFIG["img_size"], CONFIG["img_size"], 3),
        pooling=None
    )
    
    # Unfreeze top layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    # Build model
    inputs = keras.Input(shape=(CONFIG["img_size"], CONFIG["img_size"], 3))
    
    # Preprocessing
    x = data_augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    
    # Feature extraction
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(CONFIG["num_classes"], activation="softmax")(x)
    
    model = keras.Model(inputs, outputs, name="brain_mri_v2")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Trainable parameters: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    
    return model


def train_model(model, train_ds, test_ds):
    """Train the model with callbacks."""
    print("[3/4] Training model...")
    
    # Create output directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_DIR / "best_model.weights.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(LOG_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def save_model_artifacts(model, class_names):
    """Save model weights and configuration."""
    print("[4/4] Saving model artifacts...")
    
    # Save final weights
    model.save_weights(MODEL_DIR / "brain_mri_efficientnetb0.weights.h5")
    
    # Save class labels
    class_labels = {
        "classes": class_names,
        "num_classes": len(class_names),
        "version": "2.0"
    }
    with open(MODEL_DIR / "class_labels.json", "w") as f:
        json.dump(class_labels, f, indent=2)
    
    # Save preprocessing config
    preprocess_config = {
        "input_size": [CONFIG["img_size"], CONFIG["img_size"]],
        "normalization": "efficientnet",
        "color_mode": "rgb"
    }
    with open(MODEL_DIR / "preprocessing.json", "w") as f:
        json.dump(preprocess_config, f, indent=2)
    
    print(f"  Model saved to: {MODEL_DIR}")


def main():
    print("=" * 60)
    print("Multi-Dataset Brain MRI Training")
    print("=" * 60)
    print(f"Config: {CONFIG}")
    print()
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus[0].name}")
        # Memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found. Training on CPU (will be slower).")
    print()
    
    # Load data
    train_ds, test_ds, class_names = load_datasets()
    
    # Build model
    model = build_model()
    
    # Train
    history = train_model(model, train_ds, test_ds)
    
    # Save
    save_model_artifacts(model, class_names)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy*100:.2f}%")
    print("=" * 60)
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
