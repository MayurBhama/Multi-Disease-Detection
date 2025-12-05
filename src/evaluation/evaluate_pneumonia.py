import os
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from tensorflow.keras.models import load_model
from src.data_loader.pneumonia_loader import load_pneumonia_dataset


# ---------------------------------------------------------
# Load YAML Config
# ---------------------------------------------------------
with open("configs/pneumonia.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_CFG = config["dataset"]
PATH_CFG = config["paths"]

RAW_DIR = DATA_CFG["raw_dir"]
TEST_DIR = DATA_CFG["test_dir"]

MODEL_DIR = PATH_CFG["model_dir"]
MODEL_PATH = os.path.join(MODEL_DIR, PATH_CFG["model_filename"])

IMAGE_SIZE = tuple(DATA_CFG["image_size"])
BATCH_SIZE = DATA_CFG["batch_size"]
SEED = DATA_CFG["seed"]


# ---------------------------------------------------------
# Create Evaluation Output Folder
# ---------------------------------------------------------
OUTPUT_DIR = os.path.join(MODEL_DIR, "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load Test Dataset
# ---------------------------------------------------------
_, _, test_ds = load_pneumonia_dataset(
    raw_dir=RAW_DIR,
    test_dir=TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = test_ds.class_names
print("\nClass Names:", class_names)


# ---------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------
print("\nLoading trained model:", MODEL_PATH)
model = load_model(MODEL_PATH)


# ---------------------------------------------------------
# Gather Predictions & True Labels
# ---------------------------------------------------------
y_true = []
y_pred = []
y_prob = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_prob.extend(preds.ravel())
    y_pred.extend((preds > 0.5).astype(int).ravel())
    y_true.extend(labels.numpy().astype(int).ravel())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)


# ---------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------
accuracy = np.mean(y_true == y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("\nConfusion Matrix saved.")


# ---------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

print("\nROC Curve saved.")


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------
report = classification_report(y_true, y_pred, target_names=class_names)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("\nClassification Report saved:")
print(report)


print("\n Evaluation Completed Successfully!")
print(f" All outputs saved in: {OUTPUT_DIR}")
