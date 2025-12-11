"""
Correct Inference Script for DenseNet201 DR Model
Uses the correct TEST DATA LOADER.
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

# IMPORTANT: Correct loader for TEST data
from src.data_loader.retina_loader_densenet import load_test_dataset


def build_densenet201_inference(img_size, num_classes):
    """Same architecture as training"""
    
    inp = tf.keras.Input(shape=(*img_size, 3))

    base = DenseNet201(
        include_top=False,
        weights="imagenet",
        input_tensor=inp,
        pooling=None
    )

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    out = Dense(num_classes, activation="softmax", dtype="float32")(x)

    return Model(inputs=inp, outputs=out)


def predict_with_tta(model, test_ds, n_tta=5):
    """Test-time augmentation"""
    all_predictions = []

    print(f"\n🔮 Running TTA x {n_tta}...")

    for batch_idx, batch in enumerate(test_ds):
        batch_preds = []
        
        # original
        preds = model.predict(batch, verbose=0)
        batch_preds.append(preds)

        # augmented predictions
        for _ in range(n_tta - 1):
            aug = tf.image.random_flip_left_right(batch)
            aug = tf.image.random_flip_up_down(aug)
            preds_aug = model.predict(aug, verbose=0)
            batch_preds.append(preds_aug)

        avg_preds = np.mean(batch_preds, axis=0)
        all_predictions.append(avg_preds)

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} batches...")

    return np.concatenate(all_predictions, axis=0)


def make_predictions(weights_path=None, use_tta=True, n_tta=5):
    """Full inference pipeline"""
    
    cfg = yaml.safe_load(open("configs/retina_densenet.yaml"))

    img_size = tuple(cfg["dataset"]["image_size"])
    batch_size = cfg["dataset"]["batch_size"]
    num_classes = cfg["model"]["num_classes"]

    test_dir = cfg["dataset"]["test_dir"]
    test_csv_path = cfg["dataset"]["test_csv_path"]

    if weights_path is None:
        weights_path = cfg["paths"]["best_weights"]

    print("\n==============================================")
    print("🔮 DIABETIC RETINOPATHY INFERENCE - DenseNet201")
    print("==============================================")
    print(f"Loading model: {weights_path}")

    # LOAD TEST DATASET CORRECTLY
    test_ds, test_ids = load_test_dataset(
        test_dir=test_dir,
        test_csv_path=test_csv_path,
        image_size=img_size,
        batch_size=batch_size
    )

    model = build_densenet201_inference(img_size, num_classes)
    model.load_weights(weights_path)

    print("Model Loaded Successfully!")

    # RUN PREDICTIONS
    if use_tta:
        preds_proba = predict_with_tta(model, test_ds, n_tta=n_tta)
    else:
        preds_proba = model.predict(test_ds, verbose=1)

    preds = np.argmax(preds_proba, axis=1)
    conf = np.max(preds_proba, axis=1)

    # BUILD RESULTS DF
    df = pd.DataFrame({
        "id_code": test_ids,
        "diagnosis": preds,
        "confidence": conf
    })

    # probability columns
    for i in range(num_classes):
        df[f"prob_class_{i}"] = preds_proba[:, i]

    # SAVE
    df.to_csv("predictions_densenet201.csv", index=False)
    df.to_csv("predictions_densenet201_detailed.csv", index=False)

    print("\nPrediction Distribution:")
    print(df["diagnosis"].value_counts(normalize=True) * 100)

    print("\n🎉 Prediction Complete!\n")

    return df


if __name__ == "__main__":
    results = make_predictions(use_tta=True, n_tta=5)

    print("\nSample Predictions:")
    print(results.head(10))