"""
src/data_loader/retina_loader_v2.py

Enhanced data loader for APTOS 2019 Diabetic Retinopathy dataset.
Features:
- Class-weighted sampling for imbalanced data
- MixUp and CutMix augmentation
- Support for larger image sizes
- Improved preprocessing
"""

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.preprocessing.retina_preprocessing import preprocess_retina

# Import logging and exception handling
from src.utils.logger import get_logger
from src.utils.exception import DataLoadError

# Initialize logger
logger = get_logger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def compute_class_weights(labels, num_classes=5, beta=0.9999):
    """
    Compute class weights using effective number of samples.
    From "Class-Balanced Loss Based on Effective Number of Samples"
    """
    samples_per_class = np.bincount(labels, minlength=num_classes)
    
    # Effective number of samples
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return {i: w for i, w in enumerate(weights)}


def mixup_batch(images, labels, alpha=0.2):
    """Apply MixUp augmentation to a batch."""
    batch_size = tf.shape(images)[0]
    
    # Sample lambda from Beta distribution
    lam = tf.random.uniform([], 0, 1)
    if alpha > 0:
        lam = tf.maximum(lam, 1 - lam)  # Ensure lam >= 0.5 for stability
    
    # Cast lambda to match image dtype (important for mixed precision)
    lam = tf.cast(lam, images.dtype)
    
    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images and labels
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    
    # Labels are always float32
    lam_f32 = tf.cast(lam, tf.float32)
    mixed_labels = lam_f32 * labels + (1 - lam_f32) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=1.0):
    """Apply CutMix augmentation to a batch."""
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]
    
    # Sample lambda from Beta distribution
    lam = tf.random.uniform([], 0.3, 0.7)
    
    # Get random box
    cut_ratio = tf.math.sqrt(1 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
    
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    
    # Create mask
    mask_y = tf.range(img_h)
    mask_x = tf.range(img_w)
    mask_y = tf.cast((mask_y >= y1) & (mask_y < y2), tf.float32)
    mask_x = tf.cast((mask_x >= x1) & (mask_x < x2), tf.float32)
    mask = mask_y[:, None] * mask_x[None, :]
    mask = mask[:, :, None]
    
    # Cast mask to match image dtype (important for mixed precision)
    mask = tf.cast(mask, images.dtype)
    
    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    
    # Apply CutMix
    mixed_images = images * (1 - mask) + shuffled_images * mask
    
    # Adjust lambda based on actual cut area (labels are always float32)
    actual_lam = 1 - tf.cast((y2 - y1) * (x2 - x1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
    mixed_labels = actual_lam * labels + (1 - actual_lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels


def create_augmentation_layer(image_size, strength='medium'):
    """Create data augmentation layer."""
    if strength == 'light':
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.1),
        ], name="light_augmentation")
    elif strength == 'medium':
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.15),
        ], name="medium_augmentation")
    else:  # strong
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.3),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.GaussianNoise(0.02),
        ], name="strong_augmentation")


def load_retina_dataset_v2(
    raw_dir,
    csv_path,
    image_size=(640, 640),
    batch_size=8,
    seed=42,
    validation_split=0.2,
    augment_strength='strong',
    use_mixup=True,
    use_cutmix=True,
    mixup_prob=0.3,
    cutmix_prob=0.3,
    use_class_weights=True,
    use_oversampling=True,
    oversample_minority=2,  # How many times to oversample minority classes
):
    """
    Load APTOS dataset with advanced augmentation and class balancing.
    
    Args:
        raw_dir: Path to training images
        csv_path: Path to CSV with labels
        image_size: Target image size (h, w)
        batch_size: Batch size for training
        seed: Random seed
        validation_split: Fraction for validation
        augment_strength: 'light', 'medium', or 'strong'
        use_mixup: Whether to apply MixUp
        use_cutmix: Whether to apply CutMix
        mixup_prob: Probability of applying MixUp
        cutmix_prob: Probability of applying CutMix
        use_class_weights: Whether to compute class weights
        use_oversampling: Whether to oversample minority classes
        oversample_minority: Factor for oversampling
    
    Returns:
        train_ds, val_ds, class_weights, class_distribution
    """
    
    logger.info("="*80)
    logger.info("LOADING APTOS DATASET V2 (Enhanced)")
    logger.info("="*80)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    df["id_code"] = df["id_code"].astype(str).str.strip()
    df["file_path"] = df["id_code"].apply(lambda x: os.path.join(raw_dir, f"{x}.png"))
    
    # Keep only existing images
    df = df[df["file_path"].apply(os.path.exists)]
    
    logger.info(f"Found {len(df)} valid images")
    
    original_dist = df["diagnosis"].value_counts().sort_index()
    logger.info("Original class distribution:")
    for cls in range(5):
        logger.info(f"  Class {cls}: {original_dist.get(cls, 0)}")
    
    # Stratified split BEFORE oversampling
    train_df, val_df = train_test_split(
        df,
        test_size=validation_split,
        random_state=seed,
        stratify=df["diagnosis"]
    )
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}")
    
    # Oversample training data for minority classes
    if use_oversampling:
        class_counts = train_df["diagnosis"].value_counts()
        max_count = class_counts.max()
        
        balanced_dfs = []
        for cls in range(5):
            cls_df = train_df[train_df["diagnosis"] == cls]
            current_count = len(cls_df)
            
            # Determine how much to oversample
            if current_count < max_count * 0.3:  # Very minority
                target_count = min(current_count * oversample_minority, max_count)
            elif current_count < max_count * 0.6:  # Minority
                target_count = min(int(current_count * 1.5), max_count)
            else:
                target_count = current_count
            
            if target_count > current_count:
                cls_df = cls_df.sample(n=int(target_count), replace=True, random_state=seed)
            
            balanced_dfs.append(cls_df)
        
        train_df = pd.concat(balanced_dfs, ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        logger.info("After oversampling:")
        for cls in range(5):
            logger.info(f"  Class {cls}: {len(train_df[train_df['diagnosis'] == cls])}")
    
    train_paths = train_df["file_path"].values
    train_labels = train_df["diagnosis"].values
    val_paths = val_df["file_path"].values
    val_labels = val_df["diagnosis"].values
    
    # Compute class weights for loss function
    if use_class_weights:
        class_weights = compute_class_weights(train_labels)
        logger.info("Class weights:")
        for cls, w in class_weights.items():
            logger.info(f"  Class {cls}: {w:.4f}")
    else:
        class_weights = {i: 1.0 for i in range(5)}
    
    # Preprocessing functions
    def _preprocess(path, label):
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        elif hasattr(path, "numpy"):
            path = path.numpy().decode("utf-8")
        
        img = preprocess_retina(path, target_size=image_size, use_clahe=True)
        return img.astype(np.float32), np.int32(label)
    
    def tf_preprocess(path, label):
        img, lab = tf.py_function(
            _preprocess, [path, label], [tf.float32, tf.int32]
        )
        img.set_shape((*image_size, 3))
        lab.set_shape([])
        return img, lab
    
    def to_one_hot(img, label):
        return img, tf.one_hot(label, depth=5)
    
    # Create augmentation layer
    aug_layer = create_augmentation_layer(image_size, augment_strength)
    
    # Build training dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)
        .map(tf_preprocess, num_parallel_calls=AUTOTUNE)
        .map(to_one_hot, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)
    )
    
    # Apply MixUp or CutMix with probability
    if use_mixup or use_cutmix:
        def apply_mixup_cutmix(images, labels):
            rand = tf.random.uniform([])
            
            # Use tf.cond for proper graph execution
            def do_mixup():
                return mixup_batch(images, labels)
            
            def do_cutmix():
                return cutmix_batch(images, labels)
            
            def do_nothing():
                return images, labels
            
            # Nested tf.cond to handle the three cases
            if use_mixup and use_cutmix:
                result = tf.cond(
                    rand < mixup_prob,
                    do_mixup,
                    lambda: tf.cond(
                        rand < (mixup_prob + cutmix_prob),
                        do_cutmix,
                        do_nothing
                    )
                )
            elif use_mixup:
                result = tf.cond(rand < mixup_prob, do_mixup, do_nothing)
            else:  # use_cutmix only
                result = tf.cond(rand < cutmix_prob, do_cutmix, do_nothing)
            
            return result
        
        train_ds = train_ds.map(apply_mixup_cutmix, num_parallel_calls=AUTOTUNE)
    
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    # Build validation dataset (no augmentation)
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(tf_preprocess, num_parallel_calls=AUTOTUNE)
        .map(to_one_hot, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    
    logger.info("Dataset pipeline ready!")
    logger.info(f"  Training batches: ~{len(train_paths) // batch_size}")
    logger.info(f"  Validation batches: ~{len(val_paths) // batch_size}")
    logger.info("="*80)
    
    return train_ds, val_ds, class_weights, original_dist.to_dict()


def load_test_dataset_v2(test_dir, test_csv_path, image_size=(640, 640), batch_size=8):
    """Load test dataset for inference."""
    
    df = pd.read_csv(test_csv_path)
    df["id_code"] = df["id_code"].astype(str).str.strip()
    df["file_path"] = df["id_code"].apply(lambda x: os.path.join(test_dir, f"{x}.png"))
    
    test_paths = df["file_path"].values
    test_ids = df["id_code"].values
    
    logger.info(f"Found {len(test_paths)} test images")
    
    def _preprocess(path):
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        elif hasattr(path, "numpy"):
            path = path.numpy().decode("utf-8")
        
        img = preprocess_retina(path, target_size=image_size, use_clahe=True)
        return img.astype(np.float32)
    
    def tf_preprocess(path):
        img = tf.py_function(_preprocess, [path], tf.float32)
        img.set_shape((*image_size, 3))
        return img
    
    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(tf_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    
    return test_ds, test_ids
