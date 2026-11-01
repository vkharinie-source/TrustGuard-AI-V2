from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

VAL_DIR = ROOT / "server" / "training" / "dataset_v3" / "validation"
MODEL_PATH = ROOT / "model" / "deepfake_mobilenetv2_v3_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


print("=" * 70)
print("TRUSTGUARD AI - PRECISE V3 THRESHOLD SEARCH")
print("=" * 70)


val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

y_true = np.concatenate(
    [
        labels.numpy().ravel()
        for _, labels in val_ds
    ]
)

scores = model.predict(
    val_ds,
    verbose=1,
).ravel()


best_accuracy = (-1, None, None, None)
best_balanced = (-1, None, None, None)

# Fine search all the way up to 0.9999
thresholds = np.arange(
    0.50,
    1.0000,
    0.0001,
)

for threshold in thresholds:

    y_pred = (
        scores >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced = balanced_accuracy_score(
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

    current = (
        accuracy,
        threshold,
        precision,
        recall,
    )

    if accuracy > best_accuracy[0]:
        best_accuracy = current

    if balanced > best_balanced[0]:
        best_balanced = (
            balanced,
            threshold,
            precision,
            recall,
        )


# ------------------------------------------------------------
# Print best accuracy threshold
# ------------------------------------------------------------

print()
print("=" * 70)
print("BEST ACCURACY THRESHOLD")
print("=" * 70)

print(f"Threshold : {best_accuracy[1]:.4f}")
print(f"Accuracy  : {best_accuracy[0]:.4f}")
print(f"Precision : {best_accuracy[2]:.4f}")
print(f"Recall    : {best_accuracy[3]:.4f}")


# ------------------------------------------------------------
# Print best balanced-accuracy threshold
# ------------------------------------------------------------

print()
print("=" * 70)
print("BEST BALANCED-ACCURACY THRESHOLD")
print("=" * 70)

print(f"Threshold : {best_balanced[1]:.4f}")
print(f"Balanced  : {best_balanced[0]:.4f}")
print(f"Precision : {best_balanced[2]:.4f}")
print(f"Recall    : {best_balanced[3]:.4f}")


# ------------------------------------------------------------
# Confusion matrix for balanced threshold
# ------------------------------------------------------------

threshold = best_balanced[1]

y_pred = (
    scores >= threshold
).astype(int)

matrix = confusion_matrix(
    y_true,
    y_pred,
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(matrix)


# ------------------------------------------------------------
# Score statistics by class
# ------------------------------------------------------------

fake_scores = scores[y_true == 0]
real_scores = scores[y_true == 1]

print()
print("=" * 70)
print("SCORE DISTRIBUTION")
print("=" * 70)

print(
    f"FAKE mean : {fake_scores.mean():.6f}"
)

print(
    f"FAKE min  : {fake_scores.min():.6f}"
)

print(
    f"FAKE max  : {fake_scores.max():.6f}"
)

print()

print(
    f"REAL mean : {real_scores.mean():.6f}"
)

print(
    f"REAL min  : {real_scores.min():.6f}"
)

print(
    f"REAL max  : {real_scores.max():.6f}"
)

print()
print("=" * 70)
print("SEARCH COMPLETE")
print("=" * 70)