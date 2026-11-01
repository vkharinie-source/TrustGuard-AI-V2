from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "server" / "training" / "dataset_v3"
MODEL_PATH = ROOT / "model" / "deepfake_mobilenetv2_v3_best.keras"

VAL_DIR = DATASET_DIR / "validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("=" * 70)
print("TRUSTGUARD AI - V3 THRESHOLD CALIBRATION")
print("=" * 70)

print("Validation directory:", VAL_DIR)
print("Model:", MODEL_PATH)

# ------------------------------------------------------------
# Load validation dataset
# ------------------------------------------------------------

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ------------------------------------------------------------
# Load V3 model
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

# ------------------------------------------------------------
# Get true labels
# ------------------------------------------------------------

y_true = np.concatenate(
    [
        labels.numpy().ravel()
        for _, labels in val_ds
    ]
)

# ------------------------------------------------------------
# Get model scores
# ------------------------------------------------------------

scores = model.predict(
    val_ds,
    verbose=1,
).ravel()

# ------------------------------------------------------------
# Search thresholds
# ------------------------------------------------------------

best_accuracy = -1.0
best_threshold = 0.50
best_precision = 0.0
best_recall = 0.0

for threshold in np.arange(0.05, 0.96, 0.01):

    y_pred = (
        scores >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold
        best_precision = precision
        best_recall = recall

# ------------------------------------------------------------
# Final predictions using best threshold
# ------------------------------------------------------------

best_predictions = (
    scores >= best_threshold
).astype(int)

matrix = confusion_matrix(
    y_true,
    best_predictions,
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print()
print("=" * 70)
print("BEST V3 THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_accuracy:.4f}"
)

print(
    f"Precision : {best_precision:.4f}"
)

print(
    f"Recall    : {best_recall:.4f}"
)

print()
print("Confusion Matrix")
print(matrix)

print()
print("Score statistics")

print(
    f"Minimum score : {scores.min():.6f}"
)

print(
    f"Maximum score : {scores.max():.6f}"
)

print(
    f"Mean score    : {scores.mean():.6f}"
)

print()
print("=" * 70)
print("THRESHOLD CALIBRATION COMPLETE")
print("=" * 70)