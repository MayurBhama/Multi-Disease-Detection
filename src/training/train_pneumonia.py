# train_pneumonia.py
"""
Pneumonia Classification Training
Using Xception with transfer learning and fine-tuning.
"""
import os
import sys

# Disable MLflow autolog
import mlflow
os.environ["MLFLOW_DISABLE_AUTOFML"] = "true"
try:
    mlflow.autolog(disable=True)
except:
    pass

import yaml
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.data_loader.pneumonia_loader import load_pneumonia_dataset
from src.utils.logger import get_logger
from src.utils.exception import DataLoadError, CustomException

logger = get_logger(__name__)


# Load Config
logger.info("Loading configuration from configs/pneumonia.yaml")
with open("configs/pneumonia.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DIR = config["dataset"]["raw_dir"]
TEST_DIR = config["dataset"]["test_dir"]
BATCH_SIZE = config["dataset"]["batch_size"]
SEED = config["dataset"]["seed"]
EPOCHS = config["training"]["epochs"]

MODEL_DIR = config["paths"]["model_dir"]
MODEL_FILENAME = config["paths"]["model_filename"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MLFLOW_URI = config["logging"]["mlflow_tracking_uri"]
MLFLOW_EXPERIMENT = config["paths"]["mlflow_experiment"]

os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (256, 256)
logger.info(f"Config loaded: Batch size={BATCH_SIZE}, Epochs={EPOCHS}")


def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
    ])

augment_layer = get_augmentation_layer()


def build_model(trainable_layers=0):
    """Build Xception model with custom head."""
    logger.info(f"Building model: trainable_layers={trainable_layers}")
    
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(256, 256, 3)
    )

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

    logger.info(f"Model built with {model.count_params():,} parameters")
    return model


def train():
    """Main training function."""
    try:
        logger.info("=" * 60)
        logger.info("PNEUMONIA TRAINING STARTED")
        logger.info("=" * 60)

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        logger.info("Loading datasets...")
        train_ds, val_ds, test_ds = load_pneumonia_dataset(
            raw_dir=RAW_DIR,
            test_dir=TEST_DIR,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            seed=SEED,
            augment_layer=augment_layer
        )

        with mlflow.start_run(run_name="Pneumonia_Xception_Run"):
            mlflow.log_params({
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "image_size": list(IMAGE_SIZE),
            })

            # Phase 1: Train Head
            logger.info("=" * 60)
            logger.info("Phase 1: Training Classifier Head")
            logger.info("=" * 60)
            model = build_model(trainable_layers=0)
            
            history1 = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
            
            p1_acc = float(history1.history["accuracy"][-1])
            p1_val_acc = float(history1.history["val_accuracy"][-1])
            logger.info(f"Phase 1: Train Acc={p1_acc:.4f}, Val Acc={p1_val_acc:.4f}")

            # Phase 2: Fine-Tuning
            logger.info("=" * 60)
            logger.info("Phase 2: Fine-Tuning")
            logger.info("=" * 60)
            model = build_model(trainable_layers=40)

            # Keras 3.x requires .weights.h5 for save_weights_only
            checkpoint_path = MODEL_PATH + ".weights.h5"
            
            callbacks = [
                EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
                ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", save_weights_only=True)
            ]

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )

            final_acc = float(history.history["accuracy"][-1])
            final_val_acc = float(history.history["val_accuracy"][-1])
            mlflow.log_metrics({
                "final_train_acc": final_acc,
                "final_val_acc": final_val_acc,
                "final_val_loss": float(history.history["val_loss"][-1])
            })
            logger.info(f"Phase 2: Train Acc={final_acc:.4f}, Val Acc={final_val_acc:.4f}")

            # Test Evaluation
            logger.info("=" * 60)
            logger.info("Evaluating Test Dataset")
            logger.info("=" * 60)
            test_loss, test_acc = model.evaluate(test_ds)
            
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_loss", test_loss)
            logger.info(f"Test Results: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")

            # Log model weights to MLflow
            if os.path.exists(checkpoint_path):
                mlflow.log_artifact(checkpoint_path, artifact_path="model_weights")

            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED!")
            logger.info(f"Best model saved to: {checkpoint_path}")
            logger.info("=" * 60)

        return model

    except DataLoadError as e:
        logger.error(f"Data loading failed: {e}")
        raise
    except Exception as e:
        logger.critical(f"Training failed: {e}")
        raise CustomException(f"Training failed: {e}", sys.exc_info())


if __name__ == "__main__":
    try:
        model = train()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
