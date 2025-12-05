import tensorflow as tf
import os


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

        raw_dir/
            NORMAL/
            PNEUMONIA/

        test_dir/
            NORMAL/
            PNEUMONIA/

    Performs:
        - Train/Validation split (10% from training folder)
        - Optional augmentation for training only
        - Normalization (x / 255)
        - Prefetching for GPU/CPU efficiency
        - Deterministic behavior via random seed
    """

    # ---------------------------------------------------------
    # SAFETY CHECKS
    # ---------------------------------------------------------
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Training directory not found: {raw_dir}")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # ---------------------------------------------------------
    # TRAIN / VALIDATION SPLIT FROM raw_dir
    # ---------------------------------------------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        raw_dir,
        validation_split=0.1,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        raw_dir,
        validation_split=0.1,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    # ---------------------------------------------------------
    # TEST DATASET (NO SPLIT, NO SHUFFLE)
    # ---------------------------------------------------------
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=seed,
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    print("\n Class Names Detected:", train_ds.class_names)

    # ---------------------------------------------------------
    # NORMALIZATION FUNCTION
    # ---------------------------------------------------------
    def normalize(x, y):
        return x / 255.0, y

    # ---------------------------------------------------------
    # TRAIN AUGMENTATION (only for train)
    # ---------------------------------------------------------
    if augment_layer:
        train_ds = train_ds.map(
            lambda x, y: (augment_layer(x / 255.0), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Validation/Test: NO augmentation
    val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # ---------------------------------------------------------
    # PERFORMANCE OPTIMIZATION
    # ---------------------------------------------------------
    train_ds = train_ds.shuffle(1000, seed=seed) \
                       .prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    #Test set NOT cached (large datasets crash RAM)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
