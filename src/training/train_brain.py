# train_brain.py
"""
Brain MRI Tumor Classification Training
Using EfficientNetB0 with transfer learning and fine-tuning.
"""
import os
import sys

# Disable MLflow autolog BEFORE importing tensorflow
import mlflow
os.environ["MLFLOW_DISABLE_AUTOFML"] = "true"
try:
    mlflow.autolog(disable=True)
except:
    pass

import yaml
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.data_loader.brain_loader import load_brain_dataset
from src.utils.logger import get_logger
from src.utils.exception import DataLoadError, CustomException

logger = get_logger(__name__)


def safe_float(x):
    """Converts Tensors, Numpy arrays, or list items to standard python float."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, np.ndarray):
            return float(x.tolist()[-1]) if x.size > 1 else float(x.item())
        if hasattr(x, "numpy"):
            val = x.numpy()
            if isinstance(val, (np.ndarray, list, tuple)):
                return float(np.array(val).tolist()[-1])
            return float(val)
        if isinstance(x, (list, tuple)):
            return float(x[-1])
        return float(x)
    except Exception:
        try:
            return float(np.array(x).astype(float).tolist()[-1])
        except Exception:
            return 0.0


# Load YAML Config
logger.info("Loading configuration from configs/brain_mri.yaml")
with open("configs/brain_mri.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DIR = config["dataset"]["raw_dir"]
TEST_DIR = config["dataset"]["test_dir"]
IMAGE_SIZE = tuple(config["dataset"]["image_size"])
BATCH_SIZE = config["dataset"]["batch_size"]
SEED = config["dataset"]["seed"]
VAL_SPLIT = config["dataset"].get("validation_split", 0.1)

EPOCHS = int(config["training"]["epochs"])
LR = float(config["training"].get("learning_rate", 1e-4))

MODEL_DIR = config["paths"]["model_dir"]
MODEL_FILENAME = config["paths"]["model_filename"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
os.makedirs(MODEL_DIR, exist_ok=True)

MLFLOW_EXPERIMENT = config["paths"]["mlflow_experiment"]
MLFLOW_TRACKING_URI = config["logging"]["mlflow_tracking_uri"]

FT_ENABLED = config["fine_tuning"].get("enabled", True)
FT_AFTER = int(config["fine_tuning"].get("unfreeze_after_epochs", 5))
FT_LAYERS = int(config["fine_tuning"].get("trainable_layers", 30))

logger.info(f"Config loaded: Image size={IMAGE_SIZE}, Batch size={BATCH_SIZE}, Epochs={EPOCHS}")


def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(factor=0.1),
        tf.keras.layers.RandomZoom(0.10),
    ])

augment_layer = get_augmentation_layer()


def build_model(trainable_layers=0, num_classes=4):
    """Build EfficientNetB0 model with custom head."""
    logger.info(f"Building model: trainable_layers={trainable_layers}, num_classes={num_classes}")
    
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))

    for layer in base_model.layers:
        layer.trainable = False

    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])
    
    logger.info(f"Model built with {model.count_params():,} parameters")
    return model


def train():
    """Main training function."""
    try:
        logger.info("=" * 60)
        logger.info("BRAIN MRI TRAINING STARTED")
        logger.info("=" * 60)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        logger.info("Loading datasets...")
        train_ds, val_ds, test_ds, class_names = load_brain_dataset(
            raw_dir=RAW_DIR,
            test_dir=TEST_DIR,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            seed=SEED,
            validation_split=VAL_SPLIT,
            augment_layer=augment_layer,
        )
        
        num_classes = len(class_names)
        logger.info(f"Classes: {class_names}")

        with mlflow.start_run(run_name="BrainMRI_EfficientNet_Run"):
            mlflow.log_params({
                "image_size": list(IMAGE_SIZE),
                "batch_size": int(BATCH_SIZE),
                "epochs": int(EPOCHS),
                "learning_rate": float(LR),
                "fine_tuning_enabled": bool(FT_ENABLED),
                "num_classes": int(num_classes),
            })

            # Phase 1: Train head
            logger.info("=" * 60)
            logger.info("Phase 1: Training Frozen Base")
            logger.info("=" * 60)
            model = build_model(trainable_layers=0, num_classes=num_classes)
            
            history1 = model.fit(train_ds, validation_data=val_ds, epochs=FT_AFTER, verbose=1)

            p1_acc = safe_float(history1.history.get("accuracy", [0])[-1])
            p1_val_acc = safe_float(history1.history.get("val_accuracy", [0])[-1])
            mlflow.log_metric("phase1_train_acc", p1_acc)
            mlflow.log_metric("phase1_val_acc", p1_val_acc)
            logger.info(f"Phase 1: Train Acc={p1_acc:.4f}, Val Acc={p1_val_acc:.4f}")

            # Phase 2: Fine-tuning
            if FT_ENABLED:
                logger.info("=" * 60)
                logger.info("Phase 2: Fine-Tuning")
                logger.info("=" * 60)

                model_ft = build_model(trainable_layers=FT_LAYERS, num_classes=num_classes)
                model_ft.set_weights(model.get_weights())

                # Keras 3.x requires .weights.h5 extension for save_weights_only=True
                checkpoint_path = MODEL_PATH + ".weights.h5"
                
                callbacks_list = [
                    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
                    ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor="val_accuracy",
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1
                    ),
                ]

                history2 = model_ft.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks_list, verbose=1)

                final_acc = safe_float(history2.history.get("accuracy", [0])[-1])
                final_val_acc = safe_float(history2.history.get("val_accuracy", [0])[-1])
                mlflow.log_metric("final_train_acc", final_acc)
                mlflow.log_metric("final_val_acc", final_val_acc)
                logger.info(f"Phase 2: Train Acc={final_acc:.4f}, Val Acc={final_val_acc:.4f}")

                model = model_ft

            # Evaluate Test Dataset
            logger.info("=" * 60)
            logger.info("Evaluating Test Dataset")
            logger.info("=" * 60)
            test_loss, test_acc = model.evaluate(test_ds)
            mlflow.log_metric("test_loss", safe_float(test_loss))
            mlflow.log_metric("test_accuracy", safe_float(test_acc))
            
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