import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.preprocessing.retina_preprocessing import preprocess_retina

AUTOTUNE = tf.data.AUTOTUNE


def load_retina_dataset_densenet(
    raw_dir,
    csv_path,
    image_size=(512, 512),
    batch_size=8,
    seed=42,
    validation_split=0.2,
    augment_layer=None,
    compute_class_weights=True
):

    print("\n" + "=" * 80)
    print("📦 LOADING APTOS DATASET WITH CLASS BALANCING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    # FIX: Use .str.strip() (not .strip())
    df["id_code"] = df["id_code"].astype(str).str.strip()

    # Build file path
    df["file_path"] = df["id_code"].apply(lambda x: os.path.join(raw_dir, f"{x}.png"))

    # Keep only existing images
    df = df[df["file_path"].apply(os.path.exists)]

    print(f"Found {len(df)} valid images")

    print("\n📌 BEFORE BALANCING:")
    print(df["diagnosis"].value_counts())

    # -------------------------------------------------------------------------
    # BALANCE DATASET (oversampling minority classes)
    # -------------------------------------------------------------------------
    class_counts = df["diagnosis"].value_counts()
    max_count = class_counts.max()

    balanced_df = pd.concat([
        df[df["diagnosis"] == cls].sample(max_count, replace=True, random_state=seed)
        for cls in class_counts.index
    ])

    print("\n📌 AFTER BALANCING:")
    print(balanced_df["diagnosis"].value_counts())

    valid_paths = balanced_df["file_path"].values
    valid_labels = balanced_df["diagnosis"].values

    # Class weights not needed after balancing → Set to 1.0 each
    class_weights = {cls: 1.0 for cls in np.unique(valid_labels)}
    print("\n⚖ CLASS WEIGHTS:")
    print(class_weights)

    # -------------------------------------------------------------------------
    # TRAIN–VAL SPLIT
    # -------------------------------------------------------------------------
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        valid_paths,
        valid_labels,
        test_size=validation_split,
        random_state=seed,
        stratify=valid_labels
    )

    print(f"\n📦 Split: Train={len(train_paths)}, Val={len(val_paths)}")

    # -------------------------------------------------------------------------
    # PREPROCESSING FUNCTIONS
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # BUILD TF DATASETS
    # -------------------------------------------------------------------------
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(len(train_paths), seed=seed)
        .map(tf_preprocess, num_parallel_calls=AUTOTUNE)
        .map(to_one_hot, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    if augment_layer:
        train_ds = train_ds.map(
            lambda x, y: (augment_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(tf_preprocess, num_parallel_calls=AUTOTUNE)
        .map(to_one_hot, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    print("\n✅ Dataset pipeline ready!")
    print("=" * 80 + "\n")

    # Return: train_ds, val_ds, class_weights, class_distribution
    return train_ds, val_ds, class_weights, df["diagnosis"].value_counts().to_dict()

def load_test_dataset(test_dir, test_csv_path, image_size=(512, 512), batch_size=8):
    """Load test dataset (images only, no labels)."""

    df = pd.read_csv(test_csv_path)
    df["id_code"] = df["id_code"].astype(str).str.strip()

    df["file_path"] = df["id_code"].apply(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )

    test_paths = df["file_path"].values
    test_ids = df["id_code"].values

    print(f"📁 Found {len(test_paths)} test images")

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