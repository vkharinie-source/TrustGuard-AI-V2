"""
TrustGuard AI - MobileNetV2 DeepFake Image Detection Model V6 (Face-Crop Experiment)
Controlled Experiment Pipeline:
- Consumes pre-cropped 224x224 RGB dataset from training/datasets/deepfake_v6_crops/ directly.
- No redundant re-cropping or resizing.
- Raw [0, 255] RGB input passing through single internal Rescaling(1/127.5, offset=-1.0) layer.
- Exact V5 architecture: MobileNetV2 ImageNet backbone + Dual-Pooling (GAP + GMP) + Regularized ANN head.
- Exact V5 hyperparameter design: seed 42, batch size 32, Stage 1 (4 epochs @ 1e-3, frozen base), Stage 2 (4 epochs @ 1e-4, fine-tuning top 30 layers).
- Checkpoint selection, early stopping, and threshold calibration strictly on Validation crops.
- Locked threshold and metadata saved BEFORE touching the held-out test set.
- Disaggregated single-pass evaluation on held-out Test set (DeepGuard vs Kaggle, Detected vs Fallback).
- Experiment notes documenting the two within-split duplicate pairs in train/real/.
- Complete isolation from production models and services.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2

# ============================================================
# CONFIGURATION & PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "training" / "datasets" / "deepfake_v6_crops"
V6_MANIFEST_PATH = ROOT_DIR / "training" / "reports" / "dataset_manifest_v6_crops.json"
V6_VAL_REPORT_PATH = ROOT_DIR / "training" / "reports" / "v6_dataset_validation_report.json"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Dedicated V6 output directory
ARTIFACTS_DIR = ROOT_DIR / "models" / "deepfake_image_v6"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = ROOT_DIR / "training" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_ARTIFACT = ARTIFACTS_DIR / "deepfake_mobilenetv6_best.keras"
FINAL_MODEL_ARTIFACT = ARTIFACTS_DIR / "deepfake_mobilenetv6_final.keras"

METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"
GLOBAL_REPORT_PATH = REPORTS_DIR / "deepfake_image_v6_evaluation.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 4

CLASS_NAMES = ["fake", "real"]  # Index 0: FAKE, Index 1: REAL

# Documented duplicate pairs within train/real/
DOCUMENTED_TRAIN_DUPLICATES = [
    {
        "pair_id": 1,
        "split": "train",
        "class": "real",
        "file_a": "kaggle_ffhq_real_01370_real_47322.jpg",
        "file_b": "kaggle_ffhq_real_01747_real_3930.jpg",
        "sha256": "0e0a1e4dacc5f36395256a314a19cd5ab51490020a8609b384e9edb3f7a75b79",
        "note": "Near-identical upstream FFHQ images; identical facial crop area. Confined strictly to train/real/.",
    },
    {
        "pair_id": 2,
        "split": "train",
        "class": "real",
        "file_a": "kaggle_ffhq_real_01818_real_69277.jpg",
        "file_b": "kaggle_ffhq_real_01952_real_29987.jpg",
        "sha256": "0479acda4e62b10e233f45c45d643d3c7d0cf7a4bfec041dd93c54091de4ae79",
        "note": "Near-identical upstream FFHQ images; identical facial crop area. Confined strictly to train/real/.",
    },
]

print("=" * 80)
print("TRUSTGUARD AI - MOBILENETV2 V6 (PRE-CROPPED DATASET PIPELINE)")
print("=" * 80)
print(f"Dataset root : {DATASET_DIR}")
print(f"Artifacts dir: {ARTIFACTS_DIR}")
print(f"TensorFlow   : {tf.__version__}")
print(f"Image size   : {IMAGE_SIZE} (Native pre-cropped, no re-cropping)")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Random seed  : {SEED}")

# ============================================================
# VERIFY DATASETS AND MANIFEST
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing required V6 dataset directory: {folder}")

assert V6_MANIFEST_PATH.exists(), f"V6 manifest missing: {V6_MANIFEST_PATH}"
with open(V6_MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)
manifest_records_by_name = {r["filename"]: r for r in manifest_data["records"]}
print(f"Loaded manifest with {len(manifest_records_by_name)} records.")

# Set random seeds
tf.keras.utils.set_random_seed(SEED)

# ============================================================
# LOAD PRE-CROPPED DATASETS (224x224 RGB Direct Ingestion)
# ============================================================

print("\nLoading pre-cropped datasets directly from directory...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
TEST_FILE_PATHS = list(test_ds.file_paths)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Class mapping:", {name: idx for idx, name in enumerate(CLASS_NAMES)})

# ============================================================
# BUILD V6 ARCHITECTURE (MobileNetV2 + Dual-Pooling)
# ============================================================

print("\nConstructing MobileNetV2 V6 Dual-Pooling Architecture...")

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)),
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        layers.RandomContrast(0.12),
        layers.RandomBrightness(0.10),
    ],
    name="v6_crop_augmentation",
)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3), name="image_input")
augmented = data_augmentation(inputs)

# Standard native Rescaling layer: [0, 255] -> [-1, 1]
# Applied strictly once inside the model graph
scaled = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")(augmented)

features = base_model(scaled, training=False)

# Dual-Pooling: concatenate Global Average Pooling and Global Max Pooling
gap = layers.GlobalAveragePooling2D(name="gap")(features)
gmp = layers.GlobalMaxPooling2D(name="gmp")(features)
dual_pooled = layers.Concatenate(name="dual_pool")([gap, gmp])

# Regularized Classification Head (Identical to V5 for direct controlled comparison)
x = layers.BatchNormalization(name="head_bn1")(dual_pooled)
x = layers.Dropout(0.40, name="head_dropout1")(x)
x = layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="head_dense1",
)(x)
x = layers.BatchNormalization(name="head_bn2")(x)
x = layers.Dropout(0.30, name="head_dropout2")(x)
x = layers.Dense(
    64,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="head_dense2",
)(x)
outputs = layers.Dense(1, activation="sigmoid", name="ann_output")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TrustGuard_MobileNetV2_V6_FaceCrop")
print(f"Total model parameters: {model.count_params():,}")

# ============================================================
# STAGE 1: TRAIN CLASSIFICATION HEAD (Base CNN Frozen)
# ============================================================

print("\n" + "=" * 80)
print("STAGE 1: TRAINING DUAL-POOLING HEAD ON FACE CROPS (Base CNN Frozen)")
print("=" * 80)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

t_stage1 = time.time()
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    verbose=1,
)
t_stage1_elapsed = time.time() - t_stage1
print(f"Stage 1 completed in {t_stage1_elapsed:.1f}s.")

# ============================================================
# STAGE 2: FINE-TUNE TOP MOBILENETV2 LAYERS
# ============================================================

print("\n" + "=" * 80)
print("STAGE 2: FINE-TUNING TOP MOBILENETV2 LAYERS ON FACE CROPS (Top 30 Layers)")
print("=" * 80)

base_model.trainable = True

fine_tune_from = max(0, len(base_model.layers) - 30)
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
print(f"Total MobileNetV2 layers: {len(base_model.layers)}")
print(f"Fine-tuning from layer  : {fine_tune_from} (Trainable layers: {trainable_count})")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_ARTIFACT),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
]

t_stage2 = time.time()
history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)
t_stage2_elapsed = time.time() - t_stage2
print(f"Stage 2 completed in {t_stage2_elapsed:.1f}s.")

# Save final checkpoint
model.save(FINAL_MODEL_ARTIFACT)
print(f"\nSaved final model: {FINAL_MODEL_ARTIFACT}")

# Load the verified best checkpoint for evaluation
if BEST_MODEL_ARTIFACT.exists():
    best_model = tf.keras.models.load_model(BEST_MODEL_ARTIFACT)
    print(f"Loaded verified best checkpoint from: {BEST_MODEL_ARTIFACT}")
else:
    best_model = model

# ============================================================
# THRESHOLD CALIBRATION STRICTLY ON VALIDATION DATA
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD CALIBRATION STRICTLY ON VALIDATION FACE CROPS")
print("=" * 80)

y_val_true = np.concatenate([labels.numpy().ravel() for _, labels in val_ds])
y_val_scores = best_model.predict(val_ds, verbose=1).ravel()
val_auc = float(roc_auc_score(y_val_true, y_val_scores))

fpr_val, tpr_val, roc_thresholds_val = roc_curve(y_val_true, y_val_scores)
optimal_idx_val = np.argmax(tpr_val - fpr_val)
val_optimal_thresh = float(roc_thresholds_val[optimal_idx_val])
val_optimal_thresh = float(np.clip(val_optimal_thresh, 0.35, 0.65))

val_acc_at_05 = float(accuracy_score(y_val_true, (y_val_scores >= 0.50).astype(int)))
val_acc_at_opt = float(accuracy_score(y_val_true, (y_val_scores >= val_optimal_thresh).astype(int)))

# Predefined criterion: prefer 0.50 if within 0.02 of optimum
production_thresh = 0.50 if val_acc_at_05 >= (val_acc_at_opt - 0.02) else val_optimal_thresh
print(f"Validation optimal threshold: {val_optimal_thresh:.4f} (Val Acc: {val_acc_at_opt*100:.2f}%)")
print(f"Standard 0.50 threshold     : 0.5000 (Val Acc: {val_acc_at_05*100:.2f}%)")
print(f"LOCKED PRODUCTION THRESHOLD : {production_thresh:.4f}")

# Find best validation loss achieved
best_val_loss = float(min(history_stage2.history["val_loss"]))

# Save metadata with locked threshold BEFORE touching test set
metadata = {
    "model_name": "MobileNetV2 DeepFake Detector V6 (Face-Crop Baseline)",
    "version": "6.0.0",
    "architecture": "MobileNetV2 (ImageNet) + Dual-Pooling (GAP+GMP) + Dense ANN Head",
    "input_size": {"width": 224, "height": 224, "channels": 3},
    "preprocessing": "FaceCrop(margin=0.20, fallback=central_0.80) -> Rescaling(1/127.5, offset=-1.0)",
    "class_mapping": {"0": "FAKE", "1": "REAL"},
    "decision_threshold": production_thresh,
    "threshold_selection_criterion": "Max Youden J on Validation data with standard 0.50 preference within 0.02 margin",
    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "backbone": "MobileNetV2",
    "training_epochs": {
        "stage1_epochs": len(history_stage1.history["loss"]),
        "stage2_epochs": len(history_stage2.history["loss"]),
        "total_epochs": len(history_stage1.history["loss"]) + len(history_stage2.history["loss"]),
    },
    "training_history": {
        "stage1": {k: [float(v) for v in vals] for k, vals in history_stage1.history.items()},
        "stage2": {k: [float(v) for v in vals] for k, vals in history_stage2.history.items()},
    },
    "best_val_loss": best_val_loss,
    "batch_size": BATCH_SIZE,
    "random_seed": SEED,
    "experiment_notes": {
        "train_duplicate_pairs": DOCUMENTED_TRAIN_DUPLICATES,
        "isolation": "Production V4 model, predictor.py, server/main.py, and client/ untouched.",
    },
}
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Locked metadata saved to: {METADATA_PATH}")

# ============================================================
# SINGLE-PASS EVALUATION ON UNTOUCHED HELD-OUT TEST SET
# ============================================================

print("\n" + "=" * 80)
print("EVALUATING MODEL V6 ON UNTOUCHED TEST FACE CROPS (SINGLE-PASS)")
print("=" * 80)

# Extract true labels and predictions
y_test_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds])
y_test_scores = best_model.predict(test_ds, verbose=1).ravel()
y_test_pred = (y_test_scores >= production_thresh).astype(int)

# Map test records with provenance and crop status
test_file_paths = TEST_FILE_PATHS
test_records = []
for p in test_file_paths:
    fname = Path(p).name
    manifest_rec = manifest_records_by_name.get(fname, {})
    src_type = manifest_rec.get("source_type", "deepguard" if "deepguard" in fname.lower() else "kaggle")
    crop_status = manifest_rec.get("crop_status", "UNKNOWN")
    test_records.append({
        "path": p,
        "filename": fname,
        "source_type": src_type,
        "crop_status": crop_status,
    })

final_acc = float(accuracy_score(y_test_true, y_test_pred))
test_auc = float(roc_auc_score(y_test_true, y_test_scores))
final_cm = confusion_matrix(y_test_true, y_test_pred)
tn, fp, fn, tp = [int(v) for v in final_cm.ravel()]

prec_real = float(precision_score(y_test_true, y_test_pred, pos_label=1, zero_division=0))
rec_real = float(recall_score(y_test_true, y_test_pred, pos_label=1, zero_division=0))
f1_real = float(f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0))

prec_fake = float(precision_score(y_test_true, y_test_pred, pos_label=0, zero_division=0))
rec_fake = float(recall_score(y_test_true, y_test_pred, pos_label=0, zero_division=0))
f1_fake = float(f1_score(y_test_true, y_test_pred, pos_label=0, zero_division=0))

false_reals_count = int(fp)  # Ground truth FAKE predicted as REAL (index 0 -> index 1)
false_alarms_count = int(fn)  # Ground truth REAL predicted as FAKE (index 1 -> index 0)

print("\n" + "-" * 60)
print(f"FINAL UNTOUCHED TEST SET OVERALL METRICS (Threshold = {production_thresh:.4f})")
print("-" * 60)
print(f"Accuracy         : {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"ROC-AUC          : {test_auc:.4f}")
print(f"REAL Precision   : {prec_real:.4f} | Recall: {rec_real:.4f} | F1: {f1_real:.4f}")
print(f"FAKE Precision   : {prec_fake:.4f} | Recall: {rec_fake:.4f} | F1: {f1_fake:.4f}")
print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Missed DeepFakes (False-REAL)  : {fp} / 500 fakes ({fp/500*100:.2f}%)")
print(f"False Alarms on Genuine Photos : {fn} / 500 reals ({fn/500*100:.2f}%)")

# ============================================================
# SUBPOPULATION DISAGGREGATED EVALUATION: DeepGuard vs Kaggle
# ============================================================

print("\n" + "-" * 60)
print("SUBPOPULATION BREAKDOWN: DeepGuard vs Kaggle")
print("-" * 60)

dg_indices = [i for i, r in enumerate(test_records) if r["source_type"] == "deepguard"]
kg_indices = [i for i, r in enumerate(test_records) if r["source_type"] == "kaggle"]

# DeepGuard subpopulation
dg_true = y_test_true[dg_indices]
dg_pred = y_test_pred[dg_indices]
dg_scores = y_test_scores[dg_indices]
dg_cm = confusion_matrix(dg_true, dg_pred)
dg_tn, dg_fp, dg_fn, dg_tp = [int(v) for v in dg_cm.ravel()]
dg_rec_fake = float(recall_score(dg_true, dg_pred, pos_label=0, zero_division=0))
dg_prec_fake = float(precision_score(dg_true, dg_pred, pos_label=0, zero_division=0))
dg_rec_real = float(recall_score(dg_true, dg_pred, pos_label=1, zero_division=0))
dg_prec_real = float(precision_score(dg_true, dg_pred, pos_label=1, zero_division=0))
dg_auc = float(roc_auc_score(dg_true, dg_scores))

print(f"1. DeepGuard Face-Swap Subset (Total: {len(dg_indices)} | Real: {sum(dg_true==1)}, Fake: {sum(dg_true==0)}):")
print(f"   Accuracy        : {accuracy_score(dg_true, dg_pred)*100:.2f}%")
print(f"   ROC-AUC         : {dg_auc:.4f}")
print(f"   Face-Swap Recall: {dg_rec_fake*100:.2f}% ({dg_tn}/{sum(dg_true==0)} fakes detected)")
print(f"   Face-Swap Prec. : {dg_prec_fake*100:.2f}%")
print(f"   Real Photo Rec. : {dg_rec_real*100:.2f}% ({dg_tp}/{sum(dg_true==1)} genuine reals correct)")
print(f"   False-REALs     : {dg_fp} missed face-swaps ({dg_fp/sum(dg_true==0)*100:.2f}%)")
print(f"   Confusion Matrix: TN={dg_tn}, FP={dg_fp}, FN={dg_fn}, TP={dg_tp}")

# Kaggle StyleGAN subpopulation
kg_true = y_test_true[kg_indices]
kg_pred = y_test_pred[kg_indices]
kg_scores = y_test_scores[kg_indices]
kg_cm = confusion_matrix(kg_true, kg_pred)
kg_tn, kg_fp, kg_fn, kg_tp = [int(v) for v in kg_cm.ravel()]
kg_rec_fake = float(recall_score(kg_true, kg_pred, pos_label=0, zero_division=0))
kg_prec_fake = float(precision_score(kg_true, kg_pred, pos_label=0, zero_division=0))
kg_rec_real = float(recall_score(kg_true, kg_pred, pos_label=1, zero_division=0))
kg_prec_real = float(precision_score(kg_true, kg_pred, pos_label=1, zero_division=0))
kg_auc = float(roc_auc_score(kg_true, kg_scores))

print(f"\n2. Kaggle StyleGAN Generative Subset (Total: {len(kg_indices)} | Real: {sum(kg_true==1)}, Fake: {sum(kg_true==0)}):")
print(f"   Accuracy        : {accuracy_score(kg_true, kg_pred)*100:.2f}%")
print(f"   ROC-AUC         : {kg_auc:.4f}")
print(f"   Generative Rec. : {kg_rec_fake*100:.2f}% ({kg_tn}/{sum(kg_true==0)} fakes detected)")
print(f"   Generative Prec.: {kg_prec_fake*100:.2f}%")
print(f"   Real Photo Rec. : {kg_rec_real*100:.2f}% ({kg_tp}/{sum(kg_true==1)} genuine reals correct)")
print(f"   False-REALs     : {kg_fp} missed fakes ({kg_fp/sum(kg_true==0)*100:.2f}%)")
print(f"   Confusion Matrix: TN={kg_tn}, FP={kg_fp}, FN={kg_fn}, TP={kg_tp}")

# ============================================================
# CROP STATUS BREAKDOWN: Detected-Face vs Fallback
# ============================================================

print("\n" + "-" * 60)
print("CROP STATUS BREAKDOWN: Detected-Face vs Fallback")
print("-" * 60)

detected_indices = [i for i, r in enumerate(test_records) if r["crop_status"] == "FACE_DETECTED"]
fallback_indices = [i for i, r in enumerate(test_records) if r["crop_status"] == "FALLBACK_CENTRAL_80"]

det_true = y_test_true[detected_indices]
det_pred = y_test_pred[detected_indices]
det_cm = confusion_matrix(det_true, det_pred)
det_tn, det_fp, det_fn, det_tp = [int(v) for v in det_cm.ravel()]
det_rec_fake = float(recall_score(det_true, det_pred, pos_label=0, zero_division=0))
det_prec_fake = float(precision_score(det_true, det_pred, pos_label=0, zero_division=0))
det_rec_real = float(recall_score(det_true, det_pred, pos_label=1, zero_division=0))

print(f"1. Detected Faces (FACE_DETECTED) (Total: {len(detected_indices)} | Real: {sum(det_true==1)}, Fake: {sum(det_true==0)}):")
print(f"   Accuracy        : {accuracy_score(det_true, det_pred)*100:.2f}%")
print(f"   Fake Recall     : {det_rec_fake*100:.2f}% ({det_tn}/{sum(det_true==0)})")
print(f"   Fake Precision  : {det_prec_fake*100:.2f}%")
print(f"   Real Recall     : {det_rec_real*100:.2f}% ({det_tp}/{sum(det_true==1)})")
print(f"   Confusion Matrix: TN={det_tn}, FP={det_fp}, FN={det_fn}, TP={det_tp}")

fb_true = y_test_true[fallback_indices]
fb_pred = y_test_pred[fallback_indices]
fb_cm = confusion_matrix(fb_true, fb_pred)
fb_tn, fb_fp, fb_fn, fb_tp = [int(v) for v in fb_cm.ravel()]
fb_rec_fake = float(recall_score(fb_true, fb_pred, pos_label=0, zero_division=0))
fb_prec_fake = float(precision_score(fb_true, fb_pred, pos_label=0, zero_division=0))
fb_rec_real = float(recall_score(fb_true, fb_pred, pos_label=1, zero_division=0))

print(f"\n2. Fallback Central Prior (FALLBACK_CENTRAL_80) (Total: {len(fallback_indices)} | Real: {sum(fb_true==1)}, Fake: {sum(fb_true==0)}):")
print(f"   Accuracy        : {accuracy_score(fb_true, fb_pred)*100:.2f}%")
print(f"   Fake Recall     : {fb_rec_fake*100:.2f}% ({fb_tn}/{sum(fb_true==0)})")
print(f"   Fake Precision  : {fb_prec_fake*100:.2f}%")
print(f"   Real Recall     : {fb_rec_real*100:.2f}% ({fb_tp}/{sum(fb_true==1)})")
print(f"   Confusion Matrix: TN={fb_tn}, FP={fb_fp}, FN={fb_fn}, TP={fb_tp}")

# ============================================================
# COMPARISON TO HISTORICAL V5 FULL-FRAME BASELINE
# ============================================================

v5_eval_path = ROOT_DIR / "training" / "reports" / "deepfake_image_v5_evaluation.json"
v5_comparison = {}
if v5_eval_path.exists():
    try:
        with open(v5_eval_path, "r", encoding="utf-8") as f:
            v5_data = json.load(f)
            v5_comparison = {
                "v5_overall_accuracy": v5_data.get("overall_metrics", {}).get("accuracy"),
                "v5_overall_fake_recall": v5_data.get("overall_metrics", {}).get("recall_fake"),
                "v5_deepguard_fake_recall": v5_data.get("subgroups", {}).get("deepguard_face_swap", {}).get("fake_detection_rate"),
                "v5_deepguard_false_reals": v5_data.get("subgroups", {}).get("deepguard_face_swap", {}).get("confusion_matrix", {}).get("FP"),
                "v6_overall_accuracy": round(final_acc, 4),
                "v6_overall_fake_recall": round(rec_fake, 4),
                "v6_deepguard_fake_recall": round(dg_rec_fake, 4),
                "v6_deepguard_false_reals": dg_fp,
            }
    except Exception as e:
        print(f"Could not parse V5 evaluation for comparison: {e}")

# ============================================================
# SAVE FINAL EVALUATION REPORTS
# ============================================================

report = {
    "model_name": "MobileNetV2 DeepFake Detector V6 (Face-Crop Baseline)",
    "version": "6.0.0",
    "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dataset": "training/datasets/deepfake_v6_crops",
    "threshold_calibration": {
        "locked_threshold": production_thresh,
        "calibration_method": "Max Youden J on Validation Face Crops with standard 0.50 preference within 0.02 margin",
        "validation_accuracy_at_locked": val_acc_at_05 if production_thresh == 0.50 else val_acc_at_opt,
        "validation_optimal_threshold": val_optimal_thresh,
        "validation_auc": val_auc,
        "best_val_loss": best_val_loss,
    },
    "training_summary": {
        "stage1_epochs": len(history_stage1.history["loss"]),
        "stage2_epochs": len(history_stage2.history["loss"]),
        "stage1_duration_seconds": round(t_stage1_elapsed, 1),
        "stage2_duration_seconds": round(t_stage2_elapsed, 1),
        "best_validation_loss": best_val_loss,
    },
    "overall_test_metrics": {
        "accuracy": round(final_acc, 4),
        "roc_auc": round(test_auc, 4),
        "fake_precision": round(prec_fake, 4),
        "fake_recall": round(rec_fake, 4),
        "fake_f1": round(f1_fake, 4),
        "real_precision": round(prec_real, 4),
        "real_recall": round(rec_real, 4),
        "real_f1": round(f1_real, 4),
        "false_reals_count": false_reals_count,
        "false_alarms_count": false_alarms_count,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    },
    "subgroups": {
        "deepguard_face_swap": {
            "sample_count": len(dg_indices),
            "real_samples": int(sum(dg_true == 1)),
            "fake_samples": int(sum(dg_true == 0)),
            "accuracy": round(float(accuracy_score(dg_true, dg_pred)), 4),
            "roc_auc": round(dg_auc, 4),
            "fake_recall": round(dg_rec_fake, 4),
            "fake_precision": round(dg_prec_fake, 4),
            "real_recall": round(dg_rec_real, 4),
            "real_precision": round(dg_prec_real, 4),
            "false_reals_count": dg_fp,
            "confusion_matrix": {"TN": dg_tn, "FP": dg_fp, "FN": dg_fn, "TP": dg_tp},
        },
        "kaggle_stylegan": {
            "sample_count": len(kg_indices),
            "real_samples": int(sum(kg_true == 1)),
            "fake_samples": int(sum(kg_true == 0)),
            "accuracy": round(float(accuracy_score(kg_true, kg_pred)), 4),
            "roc_auc": round(kg_auc, 4),
            "fake_recall": round(kg_rec_fake, 4),
            "fake_precision": round(kg_prec_fake, 4),
            "real_recall": round(kg_rec_real, 4),
            "real_precision": round(kg_prec_real, 4),
            "false_reals_count": kg_fp,
            "confusion_matrix": {"TN": kg_tn, "FP": kg_fp, "FN": kg_fn, "TP": kg_tp},
        },
    },
    "crop_status_breakdown": {
        "detected_faces": {
            "sample_count": len(detected_indices),
            "real_samples": int(sum(det_true == 1)),
            "fake_samples": int(sum(det_true == 0)),
            "accuracy": round(float(accuracy_score(det_true, det_pred)), 4),
            "fake_recall": round(det_rec_fake, 4),
            "fake_precision": round(det_prec_fake, 4),
            "real_recall": round(det_rec_real, 4),
            "confusion_matrix": {"TN": det_tn, "FP": det_fp, "FN": det_fn, "TP": det_tp},
        },
        "fallback_crops": {
            "sample_count": len(fallback_indices),
            "real_samples": int(sum(fb_true == 1)),
            "fake_samples": int(sum(fb_true == 0)),
            "accuracy": round(float(accuracy_score(fb_true, fb_pred)), 4),
            "fake_recall": round(fb_rec_fake, 4),
            "fake_precision": round(fb_prec_fake, 4),
            "real_recall": round(fb_rec_real, 4),
            "confusion_matrix": {"TN": fb_tn, "FP": fb_fp, "FN": fb_fn, "TP": fb_tp},
        },
    },
    "comparison_against_v5_full_frame": v5_comparison,
    "experiment_notes": {
        "train_duplicate_pairs": DOCUMENTED_TRAIN_DUPLICATES,
        "production_guardrails": "Production V4 model, predictor.py, and server remain completely unchanged.",
    },
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
with open(GLOBAL_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved evaluation report to: {REPORT_PATH}")
print(f"Saved global report to     : {GLOBAL_REPORT_PATH}")
print("=" * 80)
print("TRAINING AND SINGLE-PASS EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 80)
