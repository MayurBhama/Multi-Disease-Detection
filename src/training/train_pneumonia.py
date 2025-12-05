import os
import yaml
import mlflow
import mlflow.keras
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.data_loader.pneumonia_loader import load_pneumonia_dataset


# ---------------------------------------------------------
# Load Config
# ---------------------------------------------------------
with open("configs/pneumonia.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DIR = config["dataset"]["raw_dir"]
TEST_DIR = config["dataset"]["test_dir"]
BATCH_SIZE = config["dataset"]["batch_size"]
SEED = config["dataset"]["seed"]
EPOCHS = config["training"]["epochs"]

MODEL_DIR = config["paths"]["model_dir"]
MODEL_PATH = os.path.join(MODEL_DIR, config["paths"]["model_filename"])

os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (256, 256)   # best for pneumonia dataset


# ---------------------------------------------------------
# Light Augmentation (best for X-rays)
# ---------------------------------------------------------
def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),   # 3% rotation only
    ])


augment_layer = get_augmentation_layer()


# ---------------------------------------------------------
# Build Model (stable)
# ---------------------------------------------------------
def build_model(trainable_layers=0):

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(256, 256, 3)
    )

    # Freeze majority of layers first
    if trainable_layers == 0:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    model = Sequential([
        base_model,
        BatchNormalization(),
        Dropout(0.25),
        Dense(256, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------
# Training Pipeline: Progressive Fine-Tuning
# ---------------------------------------------------------
def train():

    train_ds, val_ds, test_ds = load_pneumonia_dataset(
        raw_dir=RAW_DIR,
        test_dir=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        augment_layer=augment_layer
    )

    # -----------------------------------------------------
    # PHASE 1: Train top classifier only
    # -----------------------------------------------------
    print("\n Phase 1: Training classifier head...\n")
    model = build_model(trainable_layers=0)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=1
    )

    # -----------------------------------------------------
    # PHASE 2: Fine-tune last 40 layers
    # -----------------------------------------------------
    print("\n Phase 2: Fine-tuning deeper layers...\n")
    model = build_model(trainable_layers=40)

    callbacks = [
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy")
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\nEvaluating Test Accuracy...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n Final Test Accuracy: {test_acc:.4f}")

    return model


if __name__ == "__main__":
    train()