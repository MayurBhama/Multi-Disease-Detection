# brain_loader.py
"""
Brain MRI Dataset Loader
Loads dataset with train/val/test splits for brain tumor classification.
"""

import sys
import tensorflow as tf

# Import logging and exception handling
from src.utils.logger import get_logger
from src.utils.exception import DataLoadError

# Initialize logger
logger = get_logger(__name__)


def load_brain_dataset(raw_dir, test_dir, image_size, batch_size, seed, validation_split, augment_layer):
    """
    Loads Brain MRI dataset with correct train/val/test splits.
    
    Args:
        raw_dir: Path to training images directory
        test_dir: Path to test images directory
        image_size: Tuple (height, width) for resizing
        batch_size: Batch size for training
        seed: Random seed for reproducibility
        validation_split: Fraction of data for validation
        augment_layer: Keras augmentation layer (or None)
    
    Returns:
        train_ds, val_ds, test_ds, class_names
    
    Raises:
        DataLoadError: If dataset loading fails
    """
    try:
        logger.info("="*60)
        logger.info("LOADING BRAIN MRI DATASET")
        logger.info("="*60)
        logger.info(f"Training dir: {raw_dir}")
        logger.info(f"Test dir: {test_dir}")
        logger.info(f"Image size: {image_size}, Batch size: {batch_size}")

        # ---------- Load Train Dataset ----------
        logger.info("Loading training dataset...")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            raw_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True
        )

        # ---------- Load Validation Dataset ----------
        logger.info("Loading validation dataset...")
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            raw_dir,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True
        )

        # ---------- Load Test Dataset ----------
        logger.info("Loading test dataset...")
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False
        )

        # ---------- Extract Class Names ----------
        class_names = train_ds.class_names
        logger.info(f"Classes found: {class_names}")
        logger.info(f"Number of classes: {len(class_names)}")

        # ---------- Apply Augmentation (Training Only) ----------
        if augment_layer:
            logger.info("Applying augmentation to training data...")
            train_ds = train_ds.map(
                lambda x, y: (augment_layer(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # ---------- Prefetching for GPU Performance ----------
        logger.info("Configuring prefetch for performance...")
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

        logger.info("="*60)
        logger.info("Dataset loading complete!")
        logger.info("="*60)

        return train_ds, val_ds, test_ds, class_names

    except Exception as e:
        logger.error(f"Failed to load brain MRI dataset: {e}")
        raise DataLoadError(f"Failed to load brain MRI dataset: {e}", sys.exc_info())
