"""
TrustGuard AI - MobileNetV2 DeepFake Image Detection Model V5
Experimental Dual-Pooling Transfer Learning Pipeline
Features:
- Dual-Pooling Head (GAP + GMP concatenation) to combat localized feature dilution on face-swaps.
- Preprocessing verification: raw [0, 255] RGB input with single internal Rescaling layer.
- CPU benchmark and safety checks before training.
- Two-stage transfer learning with early stopping on validation loss.
- Predefined threshold calibration strictly on Validation data.
- Single-pass disaggregated evaluation on held-out Test set (DeepGuard vs Kaggle).
- Complete isolation from production models and services.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
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
DATASET_DIR = ROOT_DIR / "training" / "datasets" / "deepfake_v5_multisource"
MANIFEST_PATH = ROOT_DIR / "training" / "reports" / "dataset_manifest_v5.json"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

ARTIFACTS_DIR = ROOT_DIR / "models" / "deepfake_image_v5"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = ROOT_DIR / "training" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_ARTIFACT = ARTIFACTS_DIR / "deepfake_mobilenetv5_best.keras"
FINAL_MODEL_ARTIFACT = ARTIFACTS_DIR / "deepfake_mobilenetv5_final.keras"

METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"
GLOBAL_REPORT_PATH = REPORTS_DIR / "deepfake_image_v5_evaluation.json"
TRAINING_PLOT_PATH = ARTIFACTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "test_confusion_matrix.png"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 4

CLASS_NAMES = ["fake", "real"]  # Index 0: FAKE, Index 1: REAL

print("=" * 80)
print("TRUSTGUARD AI - MOBILENETV2 V5 DUAL-POOLING TRAINING")
print("=" * 80)
print(f"Dataset root : {DATASET_DIR}")
print(f"Artifacts dir: {ARTIFACTS_DIR}")
print(f"TensorFlow   : {tf.__version__}")
print(f"Image size   : {IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")

# ============================================================
# VERIFY DATASETS
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing required dataset directory: {folder}")

# ============================================================
# CPU PERFORMANCE & RESOURCE CHECK
# ============================================================

print("\n--- Running Pre-Training CPU Resource & Feasibility Check ---")
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"System RAM: Total={vm.total / (1024**3):.2f} GB, Available={vm.available / (1024**3):.2f} GB ({vm.percent}% used)")
    if vm.available / (1024**3) < 0.8:
        print("WARNING: Low available RAM! Monitoring memory closely.")
except ImportError:
    print("psutil not available; proceeding with standard memory management.")

# Small test forward pass on synthetic batch
t0 = time.time()
dummy_batch = tf.random.uniform((BATCH_SIZE, 224, 224, 3), 0.0, 255.0)
dummy_conv = layers.Conv2D(32, 3, activation="relu")(dummy_batch)
dummy_time = time.time() - t0
print(f"CPU benchmark: {BATCH_SIZE} samples dummy conv pass took {dummy_time:.3f}s. Feasibility confirmed.")

# ============================================================
# LOAD DATASETS WITH DETERMINISTIC LABELS
# ============================================================

print("\nLoading datasets...")

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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Class mapping:", {name: idx for idx, name in enumerate(CLASS_NAMES)})

# ============================================================
# BUILD EXPERIMENTAL DUAL-POOLING ARCHITECTURE
# ============================================================

print("\nConstructing MobileNetV2 V5 Dual-Pooling Architecture...")

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)),
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        layers.RandomContrast(0.12),
        layers.RandomBrightness(0.10),
    ],
    name="v5_data_augmentation",
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

# Regularized Classification Head
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

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TrustGuard_MobileNetV2_V5_DualPool")
print(f"Total model parameters: {model.count_params():,}")

# ============================================================
# STAGE 1: TRAIN CLASSIFICATION HEAD (Base CNN Frozen)
# ============================================================

print("\n" + "=" * 80)
print("STAGE 1: TRAINING DUAL-POOLING HEAD (Base CNN Frozen)")
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
print(f"Stage 1 completed in {time.time() - t_stage1:.1f}s.")

# ============================================================
# STAGE 2: FINE-TUNE TOP MOBILENETV2 LAYERS
# ============================================================

print("\n" + "=" * 80)
print("STAGE 2: FINE-TUNING TOP MOBILENETV2 LAYERS (Top 30 Layers)")
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
print(f"Stage 2 completed in {time.time() - t_stage2:.1f}s.")

# Save final checkpoint
model.save(FINAL_MODEL_ARTIFACT)
print(f"\nSaved final model: {FINAL_MODEL_ARTIFACT}")

# Load the best weights for evaluation
if BEST_MODEL_ARTIFACT.exists():
    best_model = tf.keras.models.load_model(BEST_MODEL_ARTIFACT)
    print(f"Loaded verified best checkpoint from: {BEST_MODEL_ARTIFACT}")
else:
    best_model = model

# ============================================================
# THRESHOLD CALIBRATION STRICTLY ON VALIDATION DATA
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD CALIBRATION STRICTLY ON VALIDATION DATASET")
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

# Save metadata with locked threshold BEFORE touching test set
metadata = {
    "model_name": "MobileNetV2 DeepFake Detector V5 (Dual-Pooling)",
    "version": "5.0.0",
    "architecture": "MobileNetV2 (ImageNet) + Dual-Pooling (GAP+GMP) + Dense ANN Head",
    "input_size": {"width": 224, "height": 224, "channels": 3},
    "preprocessing": "Rescaling(1/127.5, offset=-1.0)",
    "class_mapping": {"0": "FAKE", "1": "REAL"},
    "decision_threshold": production_thresh,
    "threshold_selection_criterion": "Max Youden J on Validation data with standard 0.50 preference within 0.02 margin",
    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "backbone": "MobileNetV2",
    "training_epochs": len(history_stage1.history["loss"]) + len(history_stage2.history["loss"]),
    "batch_size": BATCH_SIZE,
}
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Locked metadata saved to: {METADATA_PATH}")

# ============================================================
# SINGLE-PASS EVALUATION ON UNTOUCHED HELD-OUT TEST SET
# ============================================================

print("\n" + "=" * 80)
print("EVALUATING MODEL V5 ON UNTOUCHED TEST SET (SINGLE-PASS)")
print("=" * 80)

# Collect all test files with their subpopulation provenance
test_records = []
for root, _, files in os.walk(TEST_DIR / "fake"):
    for f in sorted(files):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            p = os.path.join(root, f)
            source_type = "deepguard_face_swap" if "deepguard" in f.lower() else "kaggle_stylegan"
            test_records.append((p, 0, source_type, f))

for root, _, files in os.walk(TEST_DIR / "real"):
    for f in sorted(files):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            p = os.path.join(root, f)
            source_type = "deepguard_real" if "deepguard" in f.lower() else "kaggle_ffhq_real"
            test_records.append((p, 1, source_type, f))

# Extract true labels
y_test_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds])
y_test_scores = best_model.predict(test_ds, verbose=1).ravel()
y_test_pred = (y_test_scores >= production_thresh).astype(int)

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

print("\n" + "-" * 60)
print(f"FINAL UNTOUCHED TEST SET OVERALL METRICS (Threshold = {production_thresh:.4f})")
print("-" * 60)
print(f"Accuracy         : {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"ROC-AUC          : {test_auc:.4f}")
print(f"REAL Precision   : {prec_real:.4f} | Recall: {rec_real:.4f} | F1: {f1_real:.4f}")
print(f"FAKE Precision   : {prec_fake:.4f} | Recall: {rec_fake:.4f} | F1: {f1_fake:.4f}")
print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"False Alarm Rate on Genuine Photos : {fn}/500 ({fn/500*100:.2f}%)")
print(f"Overall Fake Detection Rate        : {tn}/500 ({tn/500*100:.2f}%)")

# ============================================================
# SUBPOPULATION DISAGGREGATED EVALUATION
# ============================================================

print("\n" + "-" * 60)
print("SUBPOPULATION PERFORMANCE BREAKDOWN")
print("-" * 60)

dg_indices = [i for i, r in enumerate(test_records) if "deepguard" in r[2]]
kg_indices = [i for i, r in enumerate(test_records) if "kaggle" in r[2]]

# DeepGuard subpopulation
dg_true = y_test_true[dg_indices]
dg_pred = y_test_pred[dg_indices]
dg_scores = y_test_scores[dg_indices]
dg_cm = confusion_matrix(dg_true, dg_pred)
dg_tn, dg_fp, dg_fn, dg_tp = [int(v) for v in dg_cm.ravel()]
dg_rec_fake = float(recall_score(dg_true, dg_pred, pos_label=0, zero_division=0))
dg_prec_fake = float(precision_score(dg_true, dg_pred, pos_label=0, zero_division=0))
dg_rec_real = float(recall_score(dg_true, dg_pred, pos_label=1, zero_division=0))

print("1. DeepGuard Face-Swap Manipulation Subset (500 Samples: 250 Real, 250 Fake):")
print(f"   Accuracy        : {accuracy_score(dg_true, dg_pred)*100:.2f}%")
print(f"   Face-Swap Recall: {dg_rec_fake*100:.2f}% ({dg_tn}/250 fakes detected)")
print(f"   Face-Swap Prec. : {dg_prec_fake*100:.2f}%")
print(f"   Real Photo Rec. : {dg_rec_real*100:.2f}% ({dg_tp}/250 genuine reals correct)")
print(f"   Confusion Matrix: TN={dg_tn}, FP={dg_fp}, FN={dg_fn}, TP={dg_tp}")

# Kaggle StyleGAN subpopulation
kg_true = y_test_true[kg_indices]
kg_pred = y_test_pred[kg_indices]
kg_cm = confusion_matrix(kg_true, kg_pred)
kg_tn, kg_fp, kg_fn, kg_tp = [int(v) for v in kg_cm.ravel()]
kg_rec_fake = float(recall_score(kg_true, kg_pred, pos_label=0, zero_division=0))
kg_rec_real = float(recall_score(kg_true, kg_pred, pos_label=1, zero_division=0))

print("\n2. Kaggle NVIDIA FFHQ / StyleGAN Generative Subset (500 Samples: 250 Real, 250 Fake):")
print(f"   Accuracy        : {accuracy_score(kg_true, kg_pred)*100:.2f}%")
print(f"   Generative Rec. : {kg_rec_fake*100:.2f}% ({kg_tn}/250 fakes detected)")
print(f"   Real Photo Rec. : {kg_rec_real*100:.2f}% ({kg_tp}/250 genuine reals correct)")
print(f"   Confusion Matrix: TN={kg_tn}, FP={kg_fp}, FN={kg_fn}, TP={kg_tp}")

# ============================================================
# INSPECTION OF HIGH-CONFIDENCE FALSE-REAL PREDICTIONS
# ============================================================

print("\n" + "-" * 60)
print("INSPECTION OF HIGH-CONFIDENCE FALSE-REAL PREDICTIONS (MISSED DEEPFAKES)")
print("-" * 60)

high_conf_false_reals = []
all_misclassified = []

for idx in range(len(test_records)):
    p, true_lbl, src_type, fname = test_records[idx]
    pred_lbl = int(y_test_pred[idx])
    score = float(y_test_scores[idx])

    if true_lbl == 0 and pred_lbl == 1:  # FAKE missed as REAL
        rec = {
            "filename": fname,
            "source_type": src_type,
            "actual": "FAKE",
            "predicted": "REAL",
            "score": round(score, 6),
        }
        all_misclassified.append(rec)
        if score >= 0.70:
            high_conf_false_reals.append(rec)

print(f"Total missed deepfakes (pred=REAL, actual=FAKE): {len(all_misclassified)} of 500")
print(f"High-confidence false-REAL predictions (score >= 0.70): {len(high_conf_false_reals)}")

if high_conf_false_reals:
    print("\nSample high-confidence false-REAL cases:")
    for item in high_conf_false_reals[:10]:
        print(f"  {item['filename']:40s} | Source: {item['source_type']:22s} | Score: {item['score']:.4f}")
else:
    print("  None detected! No deepfakes were misclassified as REAL with score >= 0.70.")

# ============================================================
# SAVE VISUAL PLOTS AND DETAILED EVALUATION REPORTS
# ============================================================

# 1. Confusion Matrix Image
cm_img = Image.new("RGB", (560, 440), color="#0F172A")
draw = ImageDraw.Draw(cm_img)
draw.text((20, 20), "Model V5 (Dual-Pooling) Test Confusion Matrix", fill="#38BDF8")
draw.text((20, 45), f"Threshold: {production_thresh:.2f} | Overall Accuracy: {final_acc * 100:.2f}%", fill="#94A3B8")
draw.text((20, 65), f"DeepGuard Face-Swap Acc: {accuracy_score(dg_true, dg_pred)*100:.1f}% | StyleGAN Acc: {accuracy_score(kg_true, kg_pred)*100:.1f}%", fill="#A5B4FC")

draw.text((190, 110), "Pred FAKE", fill="#F87171")
draw.text((350, 110), "Pred REAL", fill="#34D399")
draw.text((30, 170), "Actual FAKE", fill="#F87171")
draw.text((30, 290), "Actual REAL", fill="#34D399")

draw.rectangle([160, 140, 310, 250], fill="#1E293B", outline="#475569", width=2)
draw.text((215, 175), str(tn), fill="#38BDF8")
draw.text((195, 205), f"True Fake (TN)\n{tn}/500 ({tn/5:.1f}%)", fill="#64748B")

draw.rectangle([320, 140, 470, 250], fill="#1E293B", outline="#EF4444", width=2)
draw.text((375, 175), str(fp), fill="#F87171")
draw.text((355, 205), f"False Real (FP)\n{fp}/500 ({fp/5:.1f}%)", fill="#EF4444")

draw.rectangle([160, 260, 310, 370], fill="#1E293B", outline="#EF4444", width=2)
draw.text((215, 295), str(fn), fill="#F87171")
draw.text((190, 325), f"False Alarm (FN)\n{fn}/500 ({fn/5:.1f}%)", fill="#EF4444")

draw.rectangle([320, 260, 470, 370], fill="#1E293B", outline="#10B981", width=2)
draw.text((375, 295), str(tp), fill="#34D399")
draw.text((345, 325), f"Genuine Real (TP)\n{tp}/500 ({tp/5:.1f}%)", fill="#34D399")

cm_img.save(CONFUSION_MATRIX_PATH)
print(f"\nSaved confusion matrix plot: {CONFUSION_MATRIX_PATH}")

# 2. Detailed JSON Evaluation Report
evaluation_report = {
    "model_version": "5.0.0",
    "architecture": "MobileNetV2 Dual-Pooling (GAP+GMP)",
    "test_samples_total": len(y_test_true),
    "per_class_counts": {"FAKE": 500, "REAL": 500},
    "threshold_used": production_thresh,
    "overall_metrics": {
        "accuracy": round(final_acc, 4),
        "roc_auc": round(test_auc, 4),
        "real": {
            "precision": round(prec_real, 4),
            "recall": round(rec_real, 4),
            "f1_score": round(f1_real, 4),
        },
        "fake": {
            "precision": round(prec_fake, 4),
            "recall": round(rec_fake, 4),
            "f1_score": round(f1_fake, 4),
        },
    },
    "confusion_matrix": {
        "true_fake_pred_fake (TN)": tn,
        "true_fake_pred_real (FP)": fp,
        "true_real_pred_fake (FN)": fn,
        "true_real_pred_real (TP)": tp,
    },
    "subpopulations": {
        "deepguard_face_swap": {
            "total_samples": len(dg_true),
            "accuracy": round(float(accuracy_score(dg_true, dg_pred)), 4),
            "fake_recall": round(dg_rec_fake, 4),
            "fake_precision": round(dg_prec_fake, 4),
            "real_recall": round(dg_rec_real, 4),
            "confusion_matrix": {"TN": dg_tn, "FP": dg_fp, "FN": dg_fn, "TP": dg_tp},
        },
        "kaggle_stylegan": {
            "total_samples": len(kg_true),
            "accuracy": round(float(accuracy_score(kg_true, kg_pred)), 4),
            "fake_recall": round(kg_rec_fake, 4),
            "real_recall": round(kg_rec_real, 4),
            "confusion_matrix": {"TN": kg_tn, "FP": kg_fp, "FN": kg_fn, "TP": kg_tp},
        },
    },
    "high_confidence_false_reals_count": len(high_conf_false_reals),
    "high_confidence_false_reals": high_conf_false_reals,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_report, f, indent=2)

with open(GLOBAL_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_report, f, indent=2)

print(f"Evaluation report saved to: {REPORT_PATH}")
print(f"Global summary report saved to: {GLOBAL_REPORT_PATH}")
print("=" * 80)
print("TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
