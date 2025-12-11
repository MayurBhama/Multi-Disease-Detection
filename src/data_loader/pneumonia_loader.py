# pneumonia_loader.py
"""
Pneumonia Dataset Loader
Loads chest X-ray dataset for pneumonia classification.
"""
import sys
import os
import tensorflow as tf

# Import logging and exception handling
from src.utils.logger import get_logger
from src.utils.exception import DataLoadError

# Initialize logger
logger = get_logger(__name__)


def load_pneumonia_dataset(
        raw_dir: str,
        test_dir: str,
        image_size=(256, 256),
        batch_size=32,
        seed=123,
        augment_layer=None
    ):
    """
    Loads Pneumonia dataset using the structure:
        raw_dir/NORMAL/, raw_dir/PNEUMONIA/
        test_dir/NORMAL/, test_dir/PNEUMONIA/

    Returns:
        train_ds, val_ds, test_ds
    
    Raises:
        DataLoadError: If dataset loading fails
    """
    try:
        logger.info("=" * 60)
        logger.info("LOADING PNEUMONIA DATASET")
        logger.info("=" * 60)
        logger.info(f"Training dir: {raw_dir}")
        logger.info(f"Test dir: {test_dir}")
        logger.info(f"Image size: {image_size}, Batch size: {batch_size}")

        # Safety checks
        if not os.path.exists(raw_dir):
            logger.error(f"Training directory not found: {raw_dir}")
            raise DataLoadError(f"Training directory not found: {raw_dir}", sys.exc_info())

        if not os.path.exists(test_dir):
            logger.error(f"Test directory not found: {test_dir}")
            raise DataLoadError(f"Test directory not found: {test_dir}", sys.exc_info())

        # Load training dataset
        logger.info("Loading training dataset...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            raw_dir,
            validation_split=0.1,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int"
        )

        # Load validation dataset
        logger.info("Loading validation dataset...")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            raw_dir,
            validation_split=0.1,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int"
        )

        # Load test dataset
        logger.info("Loading test dataset...")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=seed,
            shuffle=False,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int"
        )

        class_names = train_ds.class_names
        logger.info(f"Classes found: {class_names}")

        # Normalization function
        def normalize(x, y):
            return x / 255.0, y

        # Apply augmentation to training data
        if augment_layer:
            logger.info("Applying augmentation to training data...")
            train_ds = train_ds.map(
                lambda x, y: (augment_layer(x / 255.0), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

        # Normalize validation and test
        val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

        # Performance optimization
        logger.info("Configuring prefetch for performance...")
        train_ds = train_ds.shuffle(1000, seed=seed).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        logger.info("=" * 60)
        logger.info("Dataset loading complete!")
        logger.info("=" * 60)

        return train_ds, val_ds, test_ds

    except DataLoadError:
        raise
    except Exception as e:
        logger.error(f"Failed to load pneumonia dataset: {e}")
        raise DataLoadError(f"Failed to load pneumonia dataset: {e}", sys.exc_info())
