from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    ROOT
    / "server"
    / "training"
    / "dataset_v3"
    / "test"
)

MODEL_PATH = (
    ROOT
    / "model"
    / "deepfake_mobilenetv2_v3_best.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

THRESHOLD = 0.9941


print("=" * 70)
print("TRUSTGUARD AI - V3 UNSEEN TEST")
print("=" * 70)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
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
        for _, labels in test_ds
    ]
)

scores = model.predict(
    test_ds,
    verbose=1,
).ravel()

y_pred = (
    scores >= THRESHOLD
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

auc = roc_auc_score(
    y_true,
    scores,
)

matrix = confusion_matrix(
    y_true,
    y_pred,
)

print()
print("=" * 70)
print("V3 FINAL UNSEEN TEST RESULTS")
print("=" * 70)

print(f"Threshold : {THRESHOLD:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"AUC       : {auc:.4f}")

print()
print("Confusion Matrix")
print(matrix)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)