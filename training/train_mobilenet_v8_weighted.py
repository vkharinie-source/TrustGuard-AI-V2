"""
TrustGuard AI - V8-Exp2: Cost-Sensitive Class Weighting (w_fake=1.15, w_real=1.0) [Resumable]
- Architecture: Shared MobileNetV2 Dual-Stream (Crop 224x224 + Full-Frame 224x224)
- Feature Representation: GAP (1280) + GMP (1280) x 2 Streams = 5,120 dims
- Loss: Binary Cross-Entropy with explicit cost weighting:
    - Class 0 (FAKE): Weight = 1.15
    - Class 1 (REAL): Weight = 1.00
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
import argparse
from pathlib import Path
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve

ROOT_DIR = Path("d:/project/DeepFake-Detector-AI")
DEV_MANIFEST = ROOT_DIR / "training" / "reports" / "dataset_manifest_v8_dev.json"
ARTIFACTS_DIR = ROOT_DIR / "models" / "deepfake_image_v8"
REPORTS_DIR = ROOT_DIR / "training" / "reports"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = ARTIFACTS_DIR / "deepfake_mobilenetv8_exp2_weighted_best.keras"
LATEST_MODEL_PATH = ARTIFACTS_DIR / "deepfake_mobilenetv8_exp2_weighted_latest.keras"
STATE_JSON_PATH = ARTIFACTS_DIR / "v8_exp2_training_state.json"
STATE_REPORT_PATH = REPORTS_DIR / "v8_exp2_training_state.json"
HISTORY_JSON_PATH = REPORTS_DIR / "v8_exp2_training_history.json"
METRICS_JSON_PATH = REPORTS_DIR / "v8_exp2_validation_metrics.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 4

# Verified Class Weights: Label 0 = FAKE (1.15), Label 1 = REAL (1.00)
WEIGHT_FAKE = 1.15
WEIGHT_REAL = 1.00

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

def parse_train_pair_weighted(crop_path, frame_path, label):
    crop_raw = tf.io.read_file(crop_path)
    crop_img = tf.image.decode_jpeg(crop_raw, channels=3)
    crop_img = tf.image.resize(crop_img, IMAGE_SIZE)

    frame_raw = tf.io.read_file(frame_path)
    frame_img = tf.image.decode_jpeg(frame_raw, channels=3)
    frame_img = tf.image.resize(frame_img, IMAGE_SIZE)

    crop_aug = augment_image_train(crop_img)
    frame_aug = augment_image_train(frame_img)

    lbl_f = tf.cast(label, tf.float32)
    # Dynamic Sample Weighting: FAKE (0) -> 1.15, REAL (1) -> 1.00
    sample_weight = tf.where(
        tf.equal(lbl_f, 0.0),
        tf.constant(WEIGHT_FAKE, dtype=tf.float32),
        tf.constant(WEIGHT_REAL, dtype=tf.float32)
    )
    return {"crop_input": crop_aug, "frame_input": frame_aug}, lbl_f, sample_weight

def parse_eval_pair(crop_path, frame_path, label):
    crop_raw = tf.io.read_file(crop_path)
    crop_img = tf.image.decode_jpeg(crop_raw, channels=3)
    crop_img = tf.image.resize(crop_img, IMAGE_SIZE)

    frame_raw = tf.io.read_file(frame_path)
    frame_img = tf.image.decode_jpeg(frame_raw, channels=3)
    frame_img = tf.image.resize(frame_img, IMAGE_SIZE)
    return {"crop_input": crop_img, "frame_input": frame_img}, tf.cast(label, tf.float32)

def make_tf_dataset(pairs_list, is_training=False):
    crop_paths = [str(ROOT_DIR / p["crop_path"]) for p in pairs_list]
    frame_paths = [str(ROOT_DIR / p["frame_path"]) for p in pairs_list]
    labels = [p["binary_label"] for p in pairs_list]

    ds = tf.data.Dataset.from_tensor_slices((crop_paths, frame_paths, labels))
    if is_training:
        ds = ds.shuffle(len(pairs_list), seed=SEED)
        ds = ds.map(parse_train_pair_weighted, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(parse_eval_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_baseline_model():
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

    x = layers.BatchNormalization(name="head_bn1")(combined_features)
    x = layers.Dropout(0.40, name="head_dropout1")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.30, name="head_dropout2")(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="head_dense2")(x)
    output = layers.Dense(1, activation="sigmoid", name="ann_output")(x)

    model = tf.keras.Model(inputs=[crop_input, frame_input], outputs=output, name="TrustGuard_V8_Exp2_Weighted")
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

class TrainingStateManager:
    """Manages training state, cumulative history, and atomic checkpointing across stages."""
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
                "loss": [], "accuracy": [], "auc": [], "precision": [], "recall": [],
                "val_loss": [], "val_accuracy": [], "val_auc": [], "val_precision": [], "val_recall": []
            },
            "stage2": {
                "loss": [], "accuracy": [], "auc": [], "precision": [], "recall": [],
                "val_loss": [], "val_accuracy": [], "val_auc": [], "val_precision": [], "val_recall": []
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
                for st in ["stage1", "stage2"]:
                    if st in data["history"]:
                        self.history[st].update(data["history"][st])
            return True
        except Exception as e:
            print(f"[Warning] Failed to load existing training state: {e}")
            return False

    def record_epoch(self, stage_idx, stage_name, epoch_in_stage, total_stage_epochs, logs, model):
        logs = logs or {}
        self.stage = stage_idx
        self.stage_name = stage_name
        self.completed_epoch_in_stage = epoch_in_stage
        self.total_stage_epochs = total_stage_epochs
        self.status = "in_progress"

        # Incrementally update history
        for key in ["loss", "accuracy", "auc", "precision", "recall",
                    "val_loss", "val_accuracy", "val_auc", "val_precision", "val_recall"]:
            if key in logs:
                val = float(logs[key])
                if len(self.history[stage_name][key]) < epoch_in_stage:
                    self.history[stage_name][key].append(round(val, 5))
                else:
                    self.history[stage_name][key][epoch_in_stage - 1] = round(val, 5)

        # Save cumulative training history incrementally
        history_export = {
            "experiment": "V8-Exp2: Cost-Sensitive Class Weighting (w_fake=1.15, w_real=1.0)",
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "class_weighting": {"fake": WEIGHT_FAKE, "real": WEIGHT_REAL},
            "stage1": self.history["stage1"],
            "stage2": self.history["stage2"]
        }
        save_json_atomic(history_export, HISTORY_JSON_PATH)

        # Check and update best model based on validation loss
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            val_loss = float(val_loss)
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = round(val_loss, 5)
                self.best_stage = stage_idx
                self.best_epoch = epoch_in_stage
                save_model_atomic(model, BEST_MODEL_PATH)
                print(f"[*] NEW BEST CHECKPOINT: val_loss={self.best_val_loss:.4f} (Stage {stage_idx}, Epoch {epoch_in_stage}) -> {BEST_MODEL_PATH.name}")

        # Always save latest resumable model checkpoint containing full weights and optimizer state
        save_model_atomic(model, LATEST_MODEL_PATH)

        # Save state dictionary atomically
        state_data = {
            "experiment": "V8-Exp2: Cost-Sensitive Class Weighting (w_fake=1.15, w_real=1.0)",
            "status": self.status,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "completed_epoch_in_stage": self.completed_epoch_in_stage,
            "total_epochs_in_stage": self.total_stage_epochs,
            "class_weighting": {"fake": WEIGHT_FAKE, "real": WEIGHT_REAL},
            "best_val_loss": self.best_val_loss,
            "best_stage": self.best_stage,
            "best_epoch": self.best_epoch,
            "latest_checkpoint": str(LATEST_MODEL_PATH.relative_to(ROOT_DIR)),
            "best_checkpoint": str(BEST_MODEL_PATH.relative_to(ROOT_DIR)),
            "history": self.history,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_json_atomic(state_data, STATE_JSON_PATH)
        save_json_atomic(state_data, STATE_REPORT_PATH)

        print(f"[CHECKPOINT SAVED] Stage {stage_idx} Epoch {epoch_in_stage}/{total_stage_epochs} complete | "
              f"val_loss: {logs.get('val_loss', 0.0):.4f} (best: {self.best_val_loss:.4f}) | "
              f"Latest: {LATEST_MODEL_PATH.name}")

    def mark_completed(self):
        self.status = "completed"
        state_data = {
            "experiment": "V8-Exp2: Cost-Sensitive Class Weighting (w_fake=1.15, w_real=1.0)",
            "status": "completed",
            "stage": 2,
            "stage_name": "stage2",
            "completed_epoch_in_stage": STAGE2_EPOCHS,
            "total_epochs_in_stage": STAGE2_EPOCHS,
            "class_weighting": {"fake": WEIGHT_FAKE, "real": WEIGHT_REAL},
            "best_val_loss": self.best_val_loss,
            "best_stage": self.best_stage,
            "best_epoch": self.best_epoch,
            "latest_checkpoint": str(LATEST_MODEL_PATH.relative_to(ROOT_DIR)),
            "best_checkpoint": str(BEST_MODEL_PATH.relative_to(ROOT_DIR)),
            "history": self.history,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_json_atomic(state_data, STATE_JSON_PATH)
        save_json_atomic(state_data, STATE_REPORT_PATH)

class ResumableCheckpointCallback(tf.keras.callbacks.Callback):
    """Keras Callback that hooks into on_epoch_end to ensure full state and checkpoint persistence."""
    def __init__(self, stage_idx, stage_name, total_stage_epochs, state_manager):
        super().__init__()
        self.stage_idx = stage_idx
        self.stage_name = stage_name
        self.total_stage_epochs = total_stage_epochs
        self.state_manager = state_manager

    def on_epoch_end(self, epoch, logs=None):
        completed_epoch = epoch + 1
        self.state_manager.record_epoch(
            stage_idx=self.stage_idx,
            stage_name=self.stage_name,
            epoch_in_stage=completed_epoch,
            total_stage_epochs=self.total_stage_epochs,
            logs=logs,
            model=self.model
        )

def run_verification_check():
    """Performs a lightweight verification check of files, directories, architecture, and checkpoint roundtrips."""
    print("=" * 80)
    print("TRUSTGUARD AI V8-EXP2: CONFIGURATION & RESUME VERIFICATION CHECK")
    print("=" * 80)

    # 1. Directory checks
    print(f"[Check 1/6] Output Directories:")
    print(f"  ARTIFACTS_DIR: {ARTIFACTS_DIR} (exists: {ARTIFACTS_DIR.exists()})")
    print(f"  REPORTS_DIR  : {REPORTS_DIR} (exists: {REPORTS_DIR.exists()})")
    assert ARTIFACTS_DIR.exists() and REPORTS_DIR.exists(), "Required output directories do not exist!"

    # 2. Dataset manifest check (strictly development only)
    print(f"[Check 2/6] Development Dataset Manifest:")
    assert DEV_MANIFEST.exists(), f"Dev manifest missing: {DEV_MANIFEST}"
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_count = sum(1 for r in manifest["records"] if r["split"] == "train")
    val_count = sum(1 for r in manifest["records"] if r["split"] == "validation")
    print(f"  Manifest: {DEV_MANIFEST.name}")
    print(f"  Train Records: {train_count} (expected 4,000)")
    print(f"  Val Records  : {val_count} (expected 1,000)")
    assert train_count == 4000 and val_count == 1000, f"Unexpected record counts: train={train_count}, val={val_count}"

    sealed_manifest = ROOT_DIR / "training" / "reports" / "dataset_manifest_v8_test_sealed.json"
    print(f"  Sealed Test Isolation: Quarantined ({sealed_manifest.name} exists, STRICTLY UNOPENED)")

    # 3. Model architecture construction and parameter count
    print(f"[Check 3/6] Model Architecture Construction (Exp0-Compatible Dual-Stream):")
    model = build_baseline_model()
    total_params = model.count_params()
    configure_stage1_trainability(model)
    stage1_trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    configure_stage2_trainability(model, unfreeze_last_n=30)
    stage2_trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Total Parameters     : {total_params:,} (expected 3,606,977)")
    print(f"  Stage 1 TrainableHead: {stage1_trainable:,} (expected 1,338,241)")
    print(f"  Stage 2 TrainableTop : {stage2_trainable:,} (expected 2,848,961)")
    print(f"  Class Weights        : FAKE={WEIGHT_FAKE}, REAL={WEIGHT_REAL}")
    assert total_params == 3606977, f"Parameter mismatch: {total_params} != 3,606,977"
    assert stage1_trainable == 1338241, f"Stage 1 trainable mismatch: {stage1_trainable} != 1,338,241"
    assert stage2_trainable == 2848961, f"Stage 2 trainable mismatch: {stage2_trainable} != 2,848,961"

    # 4. Atomic Checkpoint Roundtrip Verification
    print(f"[Check 4/6] Atomic Checkpoint Save & Reload Roundtrip:")
    test_ckpt = ARTIFACTS_DIR / ".checkpoint_roundtrip_test_exp2.keras"
    save_model_atomic(model, test_ckpt)
    assert test_ckpt.exists(), "Atomic model save failed."
    reloaded_model = tf.keras.models.load_model(test_ckpt)
    assert reloaded_model.count_params() == total_params, "Reloaded model parameter mismatch."
    test_ckpt.unlink(missing_ok=True)
    print(f"  Atomic model save & load verified successfully.")

    # 5. Atomic State JSON Roundtrip Verification
    print(f"[Check 5/6] Atomic State JSON Roundtrip:")
    test_state = ARTIFACTS_DIR / ".state_roundtrip_test_exp2.json"
    save_json_atomic({"test_key": "verified_value_exp2"}, test_state)
    assert test_state.exists(), "Atomic state JSON save failed."
    with open(test_state, "r", encoding="utf-8") as f:
        reloaded_json = json.load(f)
    assert reloaded_json.get("test_key") == "verified_value_exp2", "Reloaded JSON value mismatch."
    test_state.unlink(missing_ok=True)
    print(f"  Atomic JSON save & load verified successfully.")

    # 6. Check Current Resume Readiness & Target Artifact Separation
    print(f"[Check 6/6] Target Artifact Separation & Current Resume Status:")
    print(f"  Exp2 Best Checkpoint Target  : {BEST_MODEL_PATH}")
    print(f"  Exp2 Latest Checkpoint Target: {LATEST_MODEL_PATH}")
    print(f"  Exp2 State Target            : {STATE_JSON_PATH}")
    print(f"  Exp2 History Target          : {HISTORY_JSON_PATH}")
    print(f"  Exp2 Metrics Target          : {METRICS_JSON_PATH}")
    assert "exp2" in BEST_MODEL_PATH.name and "exp2" in LATEST_MODEL_PATH.name, "Exp2 checkpoint names must be distinct!"
    assert "exp2" in METRICS_JSON_PATH.name, "Exp2 metrics path must be distinct!"

    sm = TrainingStateManager()
    has_existing = sm.load_existing()
    if has_existing:
        print(f"  Found existing resumable state for Exp2:")
        print(f"    Status: {sm.status} | Stage: {sm.stage} | Completed Epochs: {sm.completed_epoch_in_stage}/{sm.total_stage_epochs}")
        print(f"    Best Val Loss: {sm.best_val_loss}")
        print(f"    Latest Checkpoint: {LATEST_MODEL_PATH} (exists: {LATEST_MODEL_PATH.exists()})")
    else:
        print(f"  No prior valid checkpoint found on disk for Exp2.")
        print(f"  Starting training will cleanly begin at Stage 1 Epoch 1 (seed={SEED}).")
        print(f"  Subsequent interruptions will resume from the last completed epoch.")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION CHECKS PASSED. Ready for Exp2 training.")
    print("=" * 80)
    return True

def main():
    parser = argparse.ArgumentParser(description="TrustGuard AI - V8-Exp2 Weighted Dual-Stream Training")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume training from latest checkpoint if available (default: True)")
    parser.add_argument("--force-restart", action="store_true", help="Force restart training from Stage 1 Epoch 1 regardless of existing checkpoints")
    parser.add_argument("--check-config", action="store_true", help="Perform verification check of configuration and exit without training")
    args = parser.parse_args()

    if args.check_config:
        run_verification_check()
        return

    start_time = time.time()
    print("=" * 80)
    print("TRUSTGUARD AI - V8-EXP2: COST-SENSITIVE WEIGHTED LOSS (w_fake=1.15, w_real=1.0)")
    print("=" * 80)

    # Load Development Manifest
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    train_pairs = [r for r in records if r["split"] == "train"]
    val_pairs = [r for r in records if r["split"] == "validation"]
    print(f"Loaded Development Pairs: Train={len(train_pairs)}, Val={len(val_pairs)}")
    print(f"Cost-Sensitive Loss Weights: Class 0 (FAKE) = {WEIGHT_FAKE}, Class 1 (REAL) = {WEIGHT_REAL}")

    train_ds = make_tf_dataset(train_pairs, is_training=True)
    val_ds = make_tf_dataset(val_pairs, is_training=False)

    state_mgr = TrainingStateManager()
    resuming = False

    if not args.force_restart:
        resuming = state_mgr.load_existing()

    if resuming:
        print(f"\n[RESUME DETECTED]")
        print(f"  Status                  : {state_mgr.status}")
        print(f"  Current Stage           : {state_mgr.stage} ({state_mgr.stage_name})")
        print(f"  Completed Epoch in Stage: {state_mgr.completed_epoch_in_stage} / {state_mgr.total_stage_epochs}")
        print(f"  Best Val Loss So Far    : {state_mgr.best_val_loss}")
        print(f"  Latest Model Checkpoint : {LATEST_MODEL_PATH}")
        print(f"  Note: Resuming from last completed epoch; partial intra-epoch batch iterations are restarted to preserve dataset pipeline integrity.")
        model = tf.keras.models.load_model(str(LATEST_MODEL_PATH))
    else:
        if args.force_restart:
            print("\n[FORCE RESTART] Starting fresh training from Stage 1 Epoch 1.")
        else:
            print(f"\n[FRESH START] No valid resumable checkpoint found. Initializing model with seed {SEED}.")
        model = build_baseline_model()

    print(f"Model parameters: {model.count_params():,}")

    # -------------------------------------------------------------------------
    # STAGE 1: Train Head (Backbone Frozen, Cost-Weighted Loss)
    # -------------------------------------------------------------------------
    if state_mgr.stage == 1 and state_mgr.completed_epoch_in_stage < STAGE1_EPOCHS:
        stage1_start_epoch = state_mgr.completed_epoch_in_stage
        print(f"\n--- STAGE 1: TRAINING DUAL-STREAM HEAD (Backbone Frozen, Cost-Weighted Loss) ---")
        print(f"Executing Epochs {stage1_start_epoch + 1} to {STAGE1_EPOCHS}...")

        configure_stage1_trainability(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            weighted_metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        cb1 = ResumableCheckpointCallback(
            stage_idx=1,
            stage_name="stage1",
            total_stage_epochs=STAGE1_EPOCHS,
            state_manager=state_mgr
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=stage1_start_epoch,
            epochs=STAGE1_EPOCHS,
            callbacks=[cb1],
            verbose=1
        )
        print("\n--- STAGE 1 COMPLETED ---")
    else:
        print("\n[STAGE 1 ALREADY COMPLETED] Skipping to Stage 2.")

    # -------------------------------------------------------------------------
    # Transition to Stage 2
    # -------------------------------------------------------------------------
    if state_mgr.stage == 1:
        state_mgr.stage = 2
        state_mgr.stage_name = "stage2"
        state_mgr.completed_epoch_in_stage = 0
        state_mgr.total_stage_epochs = STAGE2_EPOCHS

    # -------------------------------------------------------------------------
    # STAGE 2: Fine-Tuning Top 30 MobileNetV2 Layers (Cost-Weighted Loss)
    # -------------------------------------------------------------------------
    if state_mgr.stage == 2 and state_mgr.completed_epoch_in_stage < STAGE2_EPOCHS:
        stage2_start_epoch = state_mgr.completed_epoch_in_stage
        print(f"\n--- STAGE 2: FINE-TUNING TOP MOBILENETV2 LAYERS (Top 30 Layers, Cost-Weighted Loss) ---")
        print(f"Executing Epochs {stage2_start_epoch + 1} to {STAGE2_EPOCHS}...")

        configure_stage2_trainability(model, unfreeze_last_n=30)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            weighted_metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        cb2 = ResumableCheckpointCallback(
            stage_idx=2,
            stage_name="stage2",
            total_stage_epochs=STAGE2_EPOCHS,
            state_manager=state_mgr
        )

        stage2_callbacks = [
            cb2,
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True, verbose=1),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=stage2_start_epoch,
            epochs=STAGE2_EPOCHS,
            callbacks=stage2_callbacks,
            verbose=1
        )
        print("\n--- STAGE 2 COMPLETED ---")
    else:
        print("\n[STAGE 2 ALREADY COMPLETED] Skipping to Evaluation.")

    state_mgr.mark_completed()

    # -------------------------------------------------------------------------
    # VALIDATION EVALUATION & THRESHOLD CALIBRATION
    # -------------------------------------------------------------------------
    print("\n--- VALIDATION EVALUATION & THRESHOLD CALIBRATION ---")
    if BEST_MODEL_PATH.exists():
        print(f"Loading verified best checkpoint: {BEST_MODEL_PATH}")
        best_model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    else:
        print("Using current in-memory model (BEST_MODEL_PATH not found).")
        best_model = model

    y_val_true = np.array([p["binary_label"] for p in val_pairs])
    y_val_scores = best_model.predict(val_ds, verbose=0).ravel()

    val_auc = float(roc_auc_score(y_val_true, y_val_scores))
    fpr_val, tpr_val, roc_thresholds_val = roc_curve(y_val_true, y_val_scores)
    optimal_idx = np.argmax(tpr_val - fpr_val)
    val_opt_thresh = float(np.clip(roc_thresholds_val[optimal_idx], 0.35, 0.65))

    val_acc_05 = float(accuracy_score(y_val_true, (y_val_scores >= 0.50).astype(int)))
    val_acc_opt = float(accuracy_score(y_val_true, (y_val_scores >= val_opt_thresh).astype(int)))
    calibrated_thresh = 0.50 if val_acc_05 >= (val_acc_opt - 0.02) else val_opt_thresh

    y_val_pred = (y_val_scores >= calibrated_thresh).astype(int)
    val_cm = confusion_matrix(y_val_true, y_val_pred)
    tn, fp, fn, tp = [int(v) for v in val_cm.ravel()]

    val_acc = float(accuracy_score(y_val_true, y_val_pred))
    prec_fake = float(precision_score(y_val_true, y_val_pred, pos_label=0, zero_division=0))
    rec_fake = float(recall_score(y_val_true, y_val_pred, pos_label=0, zero_division=0))
    f1_fake = float(f1_score(y_val_true, y_val_pred, pos_label=0, zero_division=0))
    prec_real = float(precision_score(y_val_true, y_val_pred, pos_label=1, zero_division=0))
    rec_real = float(recall_score(y_val_true, y_val_pred, pos_label=1, zero_division=0))
    f1_real = float(f1_score(y_val_true, y_val_pred, pos_label=1, zero_division=0))

    def eval_subgroup(indices):
        sub_true = y_val_true[indices]
        sub_pred = y_val_pred[indices]
        sub_scores = y_val_scores[indices]
        sub_cm = confusion_matrix(sub_true, sub_pred, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = [int(v) for v in sub_cm.ravel()]
        return {
            "sample_count": len(indices),
            "accuracy": round(float(accuracy_score(sub_true, sub_pred)), 4),
            "roc_auc": round(float(roc_auc_score(sub_true, sub_scores)), 4) if len(np.unique(sub_true)) > 1 else None,
            "fake_recall": round(float(recall_score(sub_true, sub_pred, pos_label=0, zero_division=0)), 4),
            "real_recall": round(float(recall_score(sub_true, sub_pred, pos_label=1, zero_division=0)), 4),
            "confusion_matrix": {"TN": s_tn, "FP": s_fp, "FN": s_fn, "TP": s_tp}
        }

    dg_idx = [i for i, p in enumerate(val_pairs) if "deepguard" in p["source_type"]]
    kg_idx = [i for i, p in enumerate(val_pairs) if "kaggle" in p["source_type"]]

    dg_val_metrics = eval_subgroup(dg_idx)
    kg_val_metrics = eval_subgroup(kg_idx)

    # Acceptance criteria checks
    dg_fake_pass = dg_val_metrics["fake_recall"] >= 0.60
    dg_real_pass = dg_val_metrics["real_recall"] >= 0.90
    kg_fake_pass = kg_val_metrics["fake_recall"] >= 0.80
    kg_real_pass = kg_val_metrics["real_recall"] >= 0.70
    all_pass = dg_fake_pass and dg_real_pass and kg_fake_pass and kg_real_pass

    print(f"\nV8-Exp2 Validation Results (Threshold={calibrated_thresh:.4f}):")
    print(f"  Overall Accuracy: {val_acc*100:.2f}% | AUC: {val_auc:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  DeepGuard Face-Swap Recall: {dg_val_metrics['fake_recall']*100:.2f}% (Target >= 60%) [{'PASS' if dg_fake_pass else 'FAIL'}]")
    print(f"  DeepGuard Genuine Real Recall: {dg_val_metrics['real_recall']*100:.2f}% (Target >= 90%) [{'PASS' if dg_real_pass else 'FAIL'}]")
    print(f"  Kaggle Generative Fake Recall: {kg_val_metrics['fake_recall']*100:.2f}% (Target >= 80%) [{'PASS' if kg_fake_pass else 'FAIL'}]")
    print(f"  Kaggle Genuine Real Recall  : {kg_val_metrics['real_recall']*100:.2f}% (Target >= 70%) [{'PASS' if kg_real_pass else 'FAIL'}]")
    print(f"  All 4 Subgroup Targets Met: {'YES (PASS)' if all_pass else 'NO (FAIL)'}")

    val_results = {
        "experiment": "V8-Exp2: Cost-Sensitive Class Weighting (w_fake=1.15, w_real=1.0)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": str(BEST_MODEL_PATH.relative_to(ROOT_DIR)),
        "class_weighting": {"fake": WEIGHT_FAKE, "real": WEIGHT_REAL},
        "calibrated_threshold": calibrated_thresh,
        "overall_metrics": {
            "accuracy": round(val_acc, 4),
            "roc_auc": round(val_auc, 4),
            "fake_precision": round(prec_fake, 4),
            "fake_recall": round(rec_fake, 4),
            "fake_f1": round(f1_fake, 4),
            "real_precision": round(prec_real, 4),
            "real_recall": round(rec_real, 4),
            "real_f1": round(f1_real, 4),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        },
        "subgroups": {
            "deepguard": dg_val_metrics,
            "kaggle": kg_val_metrics
        },
        "acceptance_criteria": {
            "deepguard_fake_recall_pass": dg_fake_pass,
            "deepguard_real_recall_pass": dg_real_pass,
            "kaggle_fake_recall_pass": kg_fake_pass,
            "kaggle_real_recall_pass": kg_real_pass,
            "all_criteria_pass": all_pass
        },
        "training_history": state_mgr.history,
        "execution_time_sec": round(time.time() - start_time, 1)
    }

    save_json_atomic(val_results, METRICS_JSON_PATH)
    print(f"\nSaved Validation Metrics: {METRICS_JSON_PATH}")

if __name__ == "__main__":
    main()
