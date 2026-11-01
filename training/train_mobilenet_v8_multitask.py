"""
TrustGuard AI - V8-Exp3: Auxiliary Multi-Task Manipulation Head [Resumable]
- Architecture: Shared MobileNetV2 Dual-Stream (Crop 224x224 + Full-Frame 224x224)
- Feature Representation: GAP (1280) + GMP (1280) x 2 Streams = 5,120 dims
- Multi-Task Outputs:
    1. Primary Head: Dense(1, sigmoid) -> Binary Authenticity P(REAL), BCE loss (weight=1.0)
    2. Auxiliary Head: Dense(3, softmax) -> Manipulation Subtype, Categorical Cross-Entropy loss (weight=0.30):
        - Class 0: GENUINE_AUTHENTIC (DeepGuard Real + Kaggle FFHQ Real)
        - Class 1: FACE_SWAP_MANIPULATION (DeepGuard Fake)
        - Class 2: STYLEGAN_GENERATIVE (Kaggle StyleGAN Fake)
- Training Partition: 4,000 Train Pairs
- Evaluation Partition: 1,000 Validation Pairs (Strictly Development Validation Only)
- Sealed Test Set: STRICTLY UNTOUCHED AND QUARANTINED
- Checkpointing: Full atomic checkpoint saving on every completed epoch across Stage 1 and Stage 2
- Resumability: Resumes cleanly from last completed epoch with weight, optimizer, and history preservation
"""

import os
import sys
import json
import time
import math
from pathlib import Path
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve
)

ROOT_DIR = Path("D:/project/DeepFake-Detector-AI")
DEV_MANIFEST = ROOT_DIR / "training" / "reports" / "dataset_manifest_v8_dev.json"
ARTIFACTS_DIR = ROOT_DIR / "models" / "deepfake_image_v8"
REPORTS_DIR = ROOT_DIR / "training" / "reports"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = ARTIFACTS_DIR / "deepfake_mobilenetv8_exp3_multitask_best.keras"
LATEST_MODEL_PATH = ARTIFACTS_DIR / "deepfake_mobilenetv8_exp3_multitask_latest.keras"
STATE_JSON_PATH = ARTIFACTS_DIR / "v8_exp3_training_state.json"
STATE_REPORT_PATH = REPORTS_DIR / "v8_exp3_training_state.json"
HISTORY_JSON_PATH = REPORTS_DIR / "v8_exp3_training_history.json"
METRICS_JSON_PATH = REPORTS_DIR / "v8_exp3_validation_metrics.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 4
AUX_LOSS_WEIGHT = 0.30

# Verified Auxiliary Multi-Class Mapping
AUX_CLASSES = {
    0: "GENUINE_AUTHENTIC",
    1: "FACE_SWAP_MANIPULATION",
    2: "STYLEGAN_GENERATIVE"
}

tf.keras.utils.set_random_seed(SEED)

def save_json_atomic(data, target_path):
    """Atomically write JSON data by writing to a temporary file and renaming."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp_path), str(target_path))

def save_model_atomic(model, target_path):
    """Atomically write Keras model file by saving to temporary file and renaming."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    model.save(str(tmp_path))
    os.replace(str(tmp_path), str(target_path))

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    """Compute 95% Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = successes / total
    denominator = 1 + z**2 / total
    centre_adj = (p + z**2 / (2 * total)) / denominator
    std_err = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return (round(max(0.0, centre_adj - std_err), 4), round(min(1.0, centre_adj + std_err), 4))

def gaussian_blur_3x3(img):
    kernel = tf.constant([
        [1./16, 2./16, 1./16],
        [2./16, 4./16, 2./16],
        [1./16, 2./16, 1./16]
    ], dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    img_4d = img[tf.newaxis, ...]
    blurred = tf.nn.depthwise_conv2d(img_4d, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return blurred[0]

def augment_image_train(img):
    img_uint8 = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
    q = tf.random.uniform([], 45, 96, dtype=tf.int32)
    img_jpeg = tf.image.adjust_jpeg_quality(img_uint8, q)
    img_float = tf.cast(img_jpeg, tf.float32)

    apply_blur = tf.random.uniform([]) < 0.30
    img_blur = tf.cond(apply_blur, lambda: gaussian_blur_3x3(img_float), lambda: img_float)

    img_bc = tf.image.random_brightness(img_blur, max_delta=0.08 * 255.0)
    img_bc = tf.image.random_contrast(img_bc, lower=0.88, upper=1.12)
    return tf.clip_by_value(img_bc, 0.0, 255.0)

def parse_train_pair_multitask(crop_path, frame_path, binary_label, multi_label):
    crop_raw = tf.io.read_file(crop_path)
    crop_img = tf.image.decode_jpeg(crop_raw, channels=3)
    crop_img = tf.image.resize(crop_img, IMAGE_SIZE)

    frame_raw = tf.io.read_file(frame_path)
    frame_img = tf.image.decode_jpeg(frame_raw, channels=3)
    frame_img = tf.image.resize(frame_img, IMAGE_SIZE)

    crop_aug = augment_image_train(crop_img)
    frame_aug = augment_image_train(frame_img)

    b_lbl = tf.cast(binary_label, tf.float32)
    m_lbl = tf.one_hot(tf.cast(multi_label, tf.int32), depth=3)

    return {"crop_input": crop_aug, "frame_input": frame_aug}, {"primary_output": b_lbl, "auxiliary_output": m_lbl}

def parse_eval_pair_multitask(crop_path, frame_path, binary_label, multi_label):
    crop_raw = tf.io.read_file(crop_path)
    crop_img = tf.image.decode_jpeg(crop_raw, channels=3)
    crop_img = tf.image.resize(crop_img, IMAGE_SIZE)

    frame_raw = tf.io.read_file(frame_path)
    frame_img = tf.image.decode_jpeg(frame_raw, channels=3)
    frame_img = tf.image.resize(frame_img, IMAGE_SIZE)

    b_lbl = tf.cast(binary_label, tf.float32)
    m_lbl = tf.one_hot(tf.cast(multi_label, tf.int32), depth=3)

    return {"crop_input": crop_img, "frame_input": frame_img}, {"primary_output": b_lbl, "auxiliary_output": m_lbl}

def make_tf_dataset(pairs_list, is_training=False):
    crop_paths = [str(ROOT_DIR / p["crop_path"]) for p in pairs_list]
    frame_paths = [str(ROOT_DIR / p["frame_path"]) for p in pairs_list]
    bin_labels = [p["binary_label"] for p in pairs_list]
    multi_labels = [p["multi_class_label"] for p in pairs_list]

    ds = tf.data.Dataset.from_tensor_slices((crop_paths, frame_paths, bin_labels, multi_labels))
    if is_training:
        ds = ds.shuffle(len(pairs_list), seed=SEED)
        ds = ds.map(parse_train_pair_multitask, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(parse_eval_pair_multitask, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_multitask_model():
    crop_input = layers.Input(shape=(224, 224, 3), name="crop_input")
    frame_input = layers.Input(shape=(224, 224, 3), name="frame_input")

    rescale_layer = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="shared_rescaling")
    crop_scaled = rescale_layer(crop_input)
    frame_scaled = rescale_layer(frame_input)

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    crop_features = base_model(crop_scaled, training=False)
    frame_features = base_model(frame_scaled, training=False)

    crop_gap = layers.GlobalAveragePooling2D(name="crop_gap")(crop_features)
    crop_gmp = layers.GlobalMaxPooling2D(name="crop_gmp")(crop_features)
    frame_gap = layers.GlobalAveragePooling2D(name="frame_gap")(frame_features)
    frame_gmp = layers.GlobalMaxPooling2D(name="frame_gmp")(frame_features)

    combined_features = layers.Concatenate(name="dual_stream_features")([crop_gap, crop_gmp, frame_gap, frame_gmp])
    shared_dense = layers.BatchNormalization(name="head_bn1")(combined_features)
    shared_dense = layers.Dropout(0.40, name="head_dropout1")(shared_dense)
    shared_dense = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="shared_dense")(shared_dense)

    # Head 1: Primary Binary Output (Authenticity)
    h1 = layers.BatchNormalization(name="h1_bn")(shared_dense)
    h1 = layers.Dropout(0.30, name="h1_drop")(h1)
    h1 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="h1_dense")(h1)
    primary_output = layers.Dense(1, activation="sigmoid", name="primary_output")(h1)

    # Head 2: Auxiliary 3-Class Output (Manipulation Subtype)
    h2 = layers.BatchNormalization(name="h2_bn")(shared_dense)
    h2 = layers.Dropout(0.30, name="h2_drop")(h2)
    h2 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="h2_dense")(h2)
    auxiliary_output = layers.Dense(3, activation="softmax", name="auxiliary_output")(h2)

    model = tf.keras.Model(
        inputs=[crop_input, frame_input],
        outputs=[primary_output, auxiliary_output],
        name="TrustGuard_V8_Exp3_MultiTask"
    )
    return model

def get_base_model(model):
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            return layer
    raise ValueError("MobileNetV2 base model layer not found in architecture.")

def configure_stage1_trainability(model):
    base_model = get_base_model(model)
    base_model.trainable = False
    for layer in model.layers:
        if layer != base_model and not isinstance(layer, layers.InputLayer):
            layer.trainable = True

def configure_stage2_trainability(model, unfreeze_last_n=30):
    base_model = get_base_model(model)
    base_model.trainable = True
    fine_tune_from = max(0, len(base_model.layers) - unfreeze_last_n)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    for layer in model.layers:
        if layer != base_model and not isinstance(layer, layers.InputLayer):
            layer.trainable = True

class MultiTaskTrainingStateManager:
    """Manages training state, cumulative multi-task history, and atomic checkpointing across stages."""
    def __init__(self):
        self.stage = 1
        self.stage_name = "stage1"
        self.completed_epoch_in_stage = 0
        self.total_stage_epochs = STAGE1_EPOCHS
        self.best_val_loss = None
        self.best_stage = None
        self.best_epoch = None
        self.status = "not_started"
        self.history = {
            "stage1": {
                "loss": [], "primary_output_loss": [], "auxiliary_output_loss": [],
                "primary_output_accuracy": [], "primary_output_auc": [],
                "auxiliary_output_cat_accuracy": [],
                "val_loss": [], "val_primary_output_loss": [], "val_auxiliary_output_loss": [],
                "val_primary_output_accuracy": [], "val_primary_output_auc": [],
                "val_auxiliary_output_cat_accuracy": []
            },
            "stage2": {
                "loss": [], "primary_output_loss": [], "auxiliary_output_loss": [],
                "primary_output_accuracy": [], "primary_output_auc": [],
                "auxiliary_output_cat_accuracy": [],
                "val_loss": [], "val_primary_output_loss": [], "val_auxiliary_output_loss": [],
                "val_primary_output_accuracy": [], "val_primary_output_auc": [],
                "val_auxiliary_output_cat_accuracy": []
            }
        }

    def load_existing(self):
        if not STATE_JSON_PATH.exists() or not LATEST_MODEL_PATH.exists():
            return False
        try:
            with open(STATE_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.stage = data.get("stage", 1)
            self.stage_name = data.get("stage_name", f"stage{self.stage}")
            self.completed_epoch_in_stage = data.get("completed_epoch_in_stage", 0)
            self.total_stage_epochs = data.get("total_epochs_in_stage", STAGE1_EPOCHS if self.stage == 1 else STAGE2_EPOCHS)
            self.best_val_loss = data.get("best_val_loss")
            self.best_stage = data.get("best_stage")
            self.best_epoch = data.get("best_epoch")
            self.status = data.get("status", "in_progress")
            if "history" in data:
                self.history = data["history"]
            print(f"[State] Loaded existing state: Stage {self.stage}, Epoch {self.completed_epoch_in_stage}/{self.total_stage_epochs}, Best Val Loss={self.best_val_loss}")
            return True
        except Exception as e:
            print(f"[State Warning] Could not load state: {e}. Starting fresh.")
            return False

    def save_state(self):
        state_data = {
            "experiment": "V8-Exp3: Auxiliary Multi-Task Manipulation Head",
            "status": self.status,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "completed_epoch_in_stage": self.completed_epoch_in_stage,
            "total_epochs_in_stage": self.total_stage_epochs,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
            "best_val_loss": self.best_val_loss,
            "best_stage": self.best_stage,
            "best_epoch": self.best_epoch,
            "latest_checkpoint": str(LATEST_MODEL_PATH.relative_to(ROOT_DIR)),
            "best_checkpoint": str(BEST_MODEL_PATH.relative_to(ROOT_DIR)),
            "history": self.history,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        save_json_atomic(state_data, STATE_JSON_PATH)
        save_json_atomic(state_data, STATE_REPORT_PATH)
        save_json_atomic(self.history, HISTORY_JSON_PATH)

class MultiTaskEpochCheckpointCallback(tf.keras.callbacks.Callback):
    """Saves atomic latest and best checkpoints at the end of every epoch across both stages."""
    def __init__(self, state_mgr: MultiTaskTrainingStateManager, stage_idx: int):
        super().__init__()
        self.state_mgr = state_mgr
        self.stage_idx = stage_idx
        self.stage_key = f"stage{stage_idx}"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Ensure completed epoch tracking
        self.state_mgr.completed_epoch_in_stage += 1
        cur_epoch = self.state_mgr.completed_epoch_in_stage

        # Append metrics to history
        hist = self.state_mgr.history[self.stage_key]
        for k in hist.keys():
            if k in logs:
                hist[k].append(float(logs[k]))

        # We monitor val_primary_output_loss as primary criterion
        val_metric = logs.get("val_primary_output_loss", logs.get("val_loss"))
        if val_metric is not None and not math.isnan(val_metric):
            is_best = (self.state_mgr.best_val_loss is None) or (val_metric < self.state_mgr.best_val_loss)
            if is_best:
                self.state_mgr.best_val_loss = float(val_metric)
                self.state_mgr.best_stage = self.stage_idx
                self.state_mgr.best_epoch = cur_epoch
                print(f"\n[Checkpoint] -> New Best Primary Val Loss: {val_metric:.5f} (Stage {self.stage_idx}, Epoch {cur_epoch}). Saving best model...")
                save_model_atomic(self.model, BEST_MODEL_PATH)

        # Always save latest checkpoint and state
        print(f"[Checkpoint] -> Saving latest model and state for Stage {self.stage_idx}, Epoch {cur_epoch}...")
        save_model_atomic(self.model, LATEST_MODEL_PATH)
        self.state_mgr.save_state()

def compile_multitask_model(model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "primary_output": "binary_crossentropy",
            "auxiliary_output": "categorical_crossentropy",
        },
        loss_weights={
            "primary_output": 1.0,
            "auxiliary_output": AUX_LOSS_WEIGHT,
        },
        metrics={
            "primary_output": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            "auxiliary_output": [
                tf.keras.metrics.CategoricalAccuracy(name="cat_accuracy"),
            ],
        },
    )

def main():
    start_time = time.time()
    print("=" * 80)
    print("TRUSTGUARD AI - V8-EXP3: AUXILIARY MULTI-TASK MANIPULATION HEAD")
    print("=" * 80)

    # 1. Load manifest and verify pairs
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    train_pairs = [r for r in records if r["split"] == "train"]
    val_pairs = [r for r in records if r["split"] == "validation"]
    print(f"Loaded Development Pairs: Train={len(train_pairs)}, Val={len(val_pairs)}")
    assert len(train_pairs) == 4000, f"Expected 4000 train pairs, got {len(train_pairs)}"
    assert len(val_pairs) == 1000, f"Expected 1000 val pairs, got {len(val_pairs)}"

    # 2. Build datasets
    train_ds = make_tf_dataset(train_pairs, is_training=True)
    val_ds = make_tf_dataset(val_pairs, is_training=False)

    # 3. State Management & Resumption
    state_mgr = MultiTaskTrainingStateManager()
    has_resumed = state_mgr.load_existing()

    if has_resumed and LATEST_MODEL_PATH.exists():
        print(f"[Resumption] Loading latest model from {LATEST_MODEL_PATH}...")
        model = tf.keras.models.load_model(str(LATEST_MODEL_PATH), compile=False)
    else:
        print("[Build] Initializing fresh TrustGuard V8 Exp3 Multi-Task Model...")
        model = build_multitask_model()
        state_mgr.status = "in_progress"

    print(f"Model Parameters: Total={model.count_params():,}")

    # =========================================================================
    # STAGE 1: Head Warmup (Backbone Frozen)
    # =========================================================================
    if state_mgr.stage == 1 and state_mgr.completed_epoch_in_stage < STAGE1_EPOCHS:
        print(f"\n--- STAGE 1: Head Warmup (Backbone Frozen, Target {STAGE1_EPOCHS} Epochs) ---")
        configure_stage1_trainability(model)
        compile_multitask_model(model, learning_rate=1e-3)

        remaining_epochs = STAGE1_EPOCHS - state_mgr.completed_epoch_in_stage
        print(f"[Stage 1] Running {remaining_epochs} remaining epochs (completed: {state_mgr.completed_epoch_in_stage})...")

        cb1 = MultiTaskEpochCheckpointCallback(state_mgr, stage_idx=1)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining_epochs,
            callbacks=[cb1],
            verbose=1
        )
        print("[Stage 1] Completed.")
        # Advance to Stage 2
        state_mgr.stage = 2
        state_mgr.stage_name = "stage2"
        state_mgr.completed_epoch_in_stage = 0
        state_mgr.total_stage_epochs = STAGE2_EPOCHS
        state_mgr.save_state()

    # =========================================================================
    # STAGE 2: Fine-Tuning Top 30 MobileNetV2 Layers
    # =========================================================================
    if state_mgr.stage == 2 and state_mgr.completed_epoch_in_stage < STAGE2_EPOCHS:
        print(f"\n--- STAGE 2: Fine-Tuning Top 30 MobileNetV2 Layers (Target {STAGE2_EPOCHS} Epochs) ---")
        configure_stage2_trainability(model, unfreeze_last_n=30)
        compile_multitask_model(model, learning_rate=1e-4)

        remaining_epochs = STAGE2_EPOCHS - state_mgr.completed_epoch_in_stage
        print(f"[Stage 2] Running {remaining_epochs} remaining epochs (completed: {state_mgr.completed_epoch_in_stage})...")

        cb2 = MultiTaskEpochCheckpointCallback(state_mgr, stage_idx=2)
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_primary_output_loss", mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_primary_output_loss", mode="min", patience=3, verbose=1
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining_epochs,
            callbacks=[cb2, lr_schedule, early_stop],
            verbose=1
        )
        print("[Stage 2] Completed.")

    state_mgr.status = "completed"
    state_mgr.save_state()

    # =========================================================================
    # FINAL EVALUATION: Development Validation Split (1,000 samples)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION: DEVELOPMENT VALIDATION SET (1,000 SAMPLES)")
    print("=" * 80)

    best_eval_model = tf.keras.models.load_model(str(BEST_MODEL_PATH), compile=False) if BEST_MODEL_PATH.exists() else model

    # Predict on validation dataset
    raw_outputs = best_eval_model.predict(val_ds, verbose=0)
    # raw_outputs is a list: [primary_output (1000, 1), auxiliary_output (1000, 3)]
    p_real_scores = raw_outputs[0].flatten()
    aux_probs = raw_outputs[1]  # (1000, 3)
    aux_pred_classes = np.argmax(aux_probs, axis=1)

    y_true_binary = np.array([p["binary_label"] for p in val_pairs], dtype=np.int32)
    y_true_multi = np.array([p["multi_class_label"] for p in val_pairs], dtype=np.int32)
    sources = np.array([p["source_type"] for p in val_pairs])
    crop_statuses = np.array([p["crop_status"] for p in val_pairs])

    # 1. Primary Binary Evaluation at Fixed Threshold 0.50
    t_fixed = 0.50
    y_pred_050 = (p_real_scores >= t_fixed).astype(int)
    cm_050 = confusion_matrix(y_true_binary, y_pred_050, labels=[0, 1])
    tn_050, fp_050, fn_050, tp_050 = [int(v) for v in cm_050.ravel()]

    acc_050 = float(accuracy_score(y_true_binary, y_pred_050))
    auc_050 = float(roc_auc_score(y_true_binary, p_real_scores))
    prec_fake_050 = float(precision_score(y_true_binary, y_pred_050, pos_label=0, zero_division=0))
    rec_fake_050 = float(recall_score(y_true_binary, y_pred_050, pos_label=0, zero_division=0))
    f1_fake_050 = float(f1_score(y_true_binary, y_pred_050, pos_label=0, zero_division=0))
    prec_real_050 = float(precision_score(y_true_binary, y_pred_050, pos_label=1, zero_division=0))
    rec_real_050 = float(recall_score(y_true_binary, y_pred_050, pos_label=1, zero_division=0))
    f1_real_050 = float(f1_score(y_true_binary, y_pred_050, pos_label=1, zero_division=0))

    # Evaluate Subgroups at t=0.50
    def compute_subgroup(mask, true_lbl):
        sub_preds = y_pred_050[mask]
        sub_scores = p_real_scores[mask]
        total = int(np.sum(mask))
        correct = int(np.sum(sub_preds == true_lbl))
        rec = float(correct / total) if total > 0 else 0.0
        ci = wilson_score_interval(correct, total)
        return {
            "sample_count": total,
            "correct_count": correct,
            "error_count": total - correct,
            "recall": round(rec, 4),
            "recall_95ci": list(ci),
            "mean_p_real": round(float(np.mean(sub_scores)), 6)
        }

    dg_fake_res = compute_subgroup(sources == "deepguard_face_fake", 0)
    dg_real_res = compute_subgroup(sources == "deepguard_face_real", 1)
    kg_fake_res = compute_subgroup(sources == "kaggle_stylegan_fake", 0)
    kg_real_res = compute_subgroup(sources == "kaggle_ffhq_real", 1)

    # Check authoritative acceptance criteria
    acceptance_checks = {
        "deepguard_face_fake_recall_ge_80": {
            "target": 0.80,
            "actual": dg_fake_res["recall"],
            "pass": bool(dg_fake_res["recall"] >= 0.80)
        },
        "deepguard_face_real_recall_ge_80": {
            "target": 0.80,
            "actual": dg_real_res["recall"],
            "pass": bool(dg_real_res["recall"] >= 0.80)
        },
        "kaggle_stylegan_fake_recall_ge_80": {
            "target": 0.80,
            "actual": kg_fake_res["recall"],
            "pass": bool(kg_fake_res["recall"] >= 0.80)
        },
        "kaggle_ffhq_real_recall_ge_70": {
            "target": 0.70,
            "actual": kg_real_res["recall"],
            "pass": bool(kg_real_res["recall"] >= 0.70)
        }
    }
    all_criteria_passed = all(v["pass"] for v in acceptance_checks.values())
    acceptance_checks["all_criteria_pass"] = all_criteria_passed
    acceptance_status = "PASSED" if all_criteria_passed else "NOT PASSED"

    # Crop status within FFHQ at t=0.50
    ffhq_det_mask = (sources == "kaggle_ffhq_real") & (crop_statuses == "FACE_DETECTED")
    ffhq_fall_mask = (sources == "kaggle_ffhq_real") & (crop_statuses == "FALLBACK_CENTRAL_80")
    crop_status_ffhq = {
        "FACE_DETECTED": {
            "count": int(np.sum(ffhq_det_mask)),
            "correct": int(np.sum(y_pred_050[ffhq_det_mask] == 1)),
            "recall": round(float(np.mean(y_pred_050[ffhq_det_mask] == 1)), 4),
            "recall_95ci": list(wilson_score_interval(int(np.sum(y_pred_050[ffhq_det_mask] == 1)), int(np.sum(ffhq_det_mask))))
        },
        "FALLBACK_CENTRAL_80": {
            "count": int(np.sum(ffhq_fall_mask)),
            "correct": int(np.sum(y_pred_050[ffhq_fall_mask] == 1)),
            "recall": round(float(np.mean(y_pred_050[ffhq_fall_mask] == 1)), 4),
            "recall_95ci": list(wilson_score_interval(int(np.sum(y_pred_050[ffhq_fall_mask] == 1)), int(np.sum(ffhq_fall_mask))))
        }
    }

    # 2. Exploratory Threshold Calibration (Youden's J)
    fpr_val, tpr_val, roc_thresholds = roc_curve(y_true_binary, p_real_scores)
    optimal_idx = int(np.argmax(tpr_val - fpr_val))
    val_opt_thresh = float(np.clip(roc_thresholds[optimal_idx], 0.35, 0.65))

    y_pred_opt = (p_real_scores >= val_opt_thresh).astype(int)
    cm_opt = confusion_matrix(y_true_binary, y_pred_opt, labels=[0, 1])
    tn_opt, fp_opt, fn_opt, tp_opt = [int(v) for v in cm_opt.ravel()]
    exploratory_calibration = {
        "calibrated_threshold": round(val_opt_thresh, 4),
        "accuracy": round(float(accuracy_score(y_true_binary, y_pred_opt)), 4),
        "deepguard_fake_recall": round(float(np.mean(y_pred_opt[sources == "deepguard_face_fake"] == 0)), 4),
        "deepguard_real_recall": round(float(np.mean(y_pred_opt[sources == "deepguard_face_real"] == 1)), 4),
        "kaggle_fake_recall": round(float(np.mean(y_pred_opt[sources == "kaggle_stylegan_fake"] == 0)), 4),
        "kaggle_real_recall": round(float(np.mean(y_pred_opt[sources == "kaggle_ffhq_real"] == 1)), 4),
        "confusion_matrix": {"TN": tn_opt, "FP": fp_opt, "FN": fn_opt, "TP": tp_opt}
    }

    # 3. Auxiliary 3-Class Task Evaluation
    aux_cm = confusion_matrix(y_true_multi, aux_pred_classes, labels=[0, 1, 2])
    aux_accuracy = float(accuracy_score(y_true_multi, aux_pred_classes))
    aux_classes_eval = {}
    for c_id, c_name in AUX_CLASSES.items():
        c_mask = (y_true_multi == c_id)
        c_tot = int(np.sum(c_mask))
        c_corr = int(np.sum(aux_pred_classes[c_mask] == c_id))
        c_rec = float(c_corr / c_tot) if c_tot > 0 else 0.0
        aux_classes_eval[c_name] = {
            "class_id": c_id,
            "sample_count": c_tot,
            "correct_count": c_corr,
            "recall": round(c_rec, 4),
            "precision": round(float(precision_score(y_true_multi == c_id, aux_pred_classes == c_id, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_multi == c_id, aux_pred_classes == c_id, zero_division=0)), 4)
        }

    auxiliary_evaluation = {
        "overall_accuracy": round(aux_accuracy, 4),
        "confusion_matrix_3x3": {
            "matrix": [[int(v) for v in row] for row in aux_cm],
            "labels": ["GENUINE_AUTHENTIC (0)", "FACE_SWAP_MANIPULATION (1)", "STYLEGAN_GENERATIVE (2)"]
        },
        "per_class_metrics": aux_classes_eval
    }

    # Print summary
    print("\n--- RESULTS SUMMARY ---")
    print(f"Primary Authenticity Overall Accuracy (t=0.50): {acc_050*100:.2f}% | ROC-AUC: {auc_050:.4f}")
    print(f"Confusion Matrix: TN={tn_050}, FP={fp_050}, FN={fn_050}, TP={tp_050}")
    print(f"  DeepGuard Fake Recall: {dg_fake_res['recall']*100:.2f}% (Target >= 80%) -> {'PASS' if acceptance_checks['deepguard_face_fake_recall_ge_80']['pass'] else 'FAIL'}")
    print(f"  DeepGuard Real Recall: {dg_real_res['recall']*100:.2f}% (Target >= 80%) -> {'PASS' if acceptance_checks['deepguard_face_real_recall_ge_80']['pass'] else 'FAIL'}")
    print(f"  Kaggle Fake Recall   : {kg_fake_res['recall']*100:.2f}% (Target >= 80%) -> {'PASS' if acceptance_checks['kaggle_stylegan_fake_recall_ge_80']['pass'] else 'FAIL'}")
    print(f"  Kaggle Real Recall   : {kg_real_res['recall']*100:.2f}% (Target >= 70%) -> {'PASS' if acceptance_checks['kaggle_ffhq_real_recall_ge_70']['pass'] else 'FAIL'}")
    print(f"Auxiliary 3-Class Overall Accuracy: {aux_accuracy*100:.2f}%")
    print(f"Overall Acceptance Decision: {acceptance_status}")

    # Build final metrics JSON
    final_metrics = {
        "experiment": "V8-Exp3: Auxiliary Multi-Task Manipulation Head",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_best": str(BEST_MODEL_PATH.relative_to(ROOT_DIR)),
        "checkpoint_latest": str(LATEST_MODEL_PATH.relative_to(ROOT_DIR)),
        "aux_loss_weight": AUX_LOSS_WEIGHT,
        "baseline_threshold": t_fixed,
        "overall_metrics_threshold_0.50": {
            "accuracy": round(acc_050, 4),
            "roc_auc": round(auc_050, 4),
            "fake_precision": round(prec_fake_050, 4),
            "fake_recall": round(rec_fake_050, 4),
            "fake_f1": round(f1_fake_050, 4),
            "real_precision": round(prec_real_050, 4),
            "real_recall": round(rec_real_050, 4),
            "real_f1": round(f1_real_050, 4),
            "confusion_matrix": {"TN": tn_050, "FP": fp_050, "FN": fn_050, "TP": tp_050}
        },
        "subgroups_threshold_0.50": {
            "deepguard_face_fake": dg_fake_res,
            "deepguard_face_real": dg_real_res,
            "kaggle_stylegan_fake": kg_fake_res,
            "kaggle_ffhq_real": kg_real_res
        },
        "crop_status_within_ffhq": crop_status_ffhq,
        "acceptance_criteria": acceptance_checks,
        "exploratory_calibration": exploratory_calibration,
        "auxiliary_3class_evaluation": auxiliary_evaluation,
        "execution_time_sec": round(time.time() - start_time, 1)
    }

    save_json_atomic(final_metrics, METRICS_JSON_PATH)
    print(f"\nSaved Completed Validation Metrics: {METRICS_JSON_PATH}")

if __name__ == "__main__":
    main()
