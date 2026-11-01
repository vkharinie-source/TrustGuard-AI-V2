"""
TrustGuard AI - V8-Exp4: Attenuated Multi-Task Auxiliary Loss (lambda_aux = 0.12)
- Hypothesis:
    Reducing auxiliary loss weight from 0.30 (Exp3) to 0.12 (Exp4) maintains generative
    feature disentanglement (FFHQ Real vs StyleGAN Fake) while reducing gradient penalty
    on video-compression artifacts by 60%, restoring DeepGuard Face Real recall to >= 80%.
    (Documented alternative: lambda_aux = 0.15)
- Architecture: Shared MobileNetV2 Dual-Stream (Crop 224x224 + Full-Frame 224x224)
- Representation: GAP (1280) + GMP (1280) x 2 Streams = 5,120 dims
- Primary Head: Dense(1, sigmoid) -> Binary Authenticity P(REAL), BCE loss (weight=1.0)
- Auxiliary Head: Dense(3, softmax) -> Manipulation Subtype, CCE loss (weight=0.12):
    - Class 0: GENUINE_AUTHENTIC (DeepGuard Real + Kaggle FFHQ Real)
    - Class 1: FACE_SWAP_MANIPULATION (DeepGuard Fake)
    - Class 2: STYLEGAN_GENERATIVE (Kaggle StyleGAN Fake)
- Partitions: Immutable 4,000 Train Pairs, 1,000 Dev-Validation Pairs (dataset_manifest_v8_dev.json)
- Sealed Test Set: STRICTLY UNTOUCHED AND QUARANTINED
- BatchNormalization: ALL BN layers explicitly frozen during fine-tuning.
- Safety: Duplicate-process lockfile, atomic saving, safe resumption.
"""

import os
import sys
import json
import time
import math
import atexit
import signal
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

# Dedicated Exp4 File Paths
BEST_MODEL_PATH = ARTIFACTS_DIR / "deepfake_mobilenetv8_exp4_multitask_best.keras"
LATEST_MODEL_PATH = ARTIFACTS_DIR / "deepfake_mobilenetv8_exp4_multitask_latest.keras"
STATE_JSON_PATH = ARTIFACTS_DIR / "v8_exp4_training_state.json"
STATE_REPORT_PATH = REPORTS_DIR / "v8_exp4_training_state.json"
HISTORY_JSON_PATH = REPORTS_DIR / "v8_exp4_training_history.json"
METRICS_JSON_PATH = REPORTS_DIR / "v8_exp4_validation_metrics.json"
FINAL_REPORT_PATH = REPORTS_DIR / "v8_exp4_final_report.md"
LOCKFILE_PATH = REPORTS_DIR / "v8_exp4_training.lock"

# Baseline Exp2 and Exp3 metrics for automated comparison
EXP2_METRICS_PATH = REPORTS_DIR / "v8_exp2_validation_metrics.json"
EXP3_METRICS_PATH = REPORTS_DIR / "v8_exp3_validation_metrics.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 4

# PRIMARY HYPOTHESIS: 0.12 (Alternative: 0.15)
AUX_LOSS_WEIGHT = 0.12
AUX_LOSS_WEIGHT_ALTERNATIVE = 0.15

# Verified Auxiliary Multi-Class Mapping
AUX_CLASSES = {
    0: "GENUINE_AUTHENTIC",
    1: "FACE_SWAP_MANIPULATION",
    2: "STYLEGAN_GENERATIVE"
}

tf.keras.utils.set_random_seed(SEED)

# =========================================================================
# PROCESS LOCK & DUPLICATE PREVENTION
# =========================================================================
class ProcessLock:
    """Ensures only a single instance of Exp4 training runs at any time."""
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path

    def acquire(self):
        if self.lock_path.exists():
            try:
                content = self.lock_path.read_text(encoding="utf-8").strip()
                pid = int(content)
                # Check if process is still active on Windows
                if self._is_pid_running(pid):
                    print(f"\n[DUPLICATE PROCESS DETECTED] Process with PID {pid} is actively running Exp4 training.")
                    print(f"Lockfile: {self.lock_path}")
                    print("Exiting to prevent duplicate training and checkpoint corruption.")
                    sys.exit(1)
                else:
                    print(f"[Lock] Found stale lockfile from PID {pid}. Cleaning up and acquiring lock.")
            except Exception as e:
                print(f"[Lock Warning] Could not parse existing lockfile: {e}. Overwriting.")

        # Write current PID
        my_pid = os.getpid()
        self.lock_path.write_text(str(my_pid), encoding="utf-8")
        atexit.register(self.release)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        print(f"[Lock Acquired] Process PID {my_pid} registered at {self.lock_path}")

    def release(self):
        if self.lock_path.exists():
            try:
                content = self.lock_path.read_text(encoding="utf-8").strip()
                if int(content) == os.getpid():
                    self.lock_path.unlink()
                    print(f"[Lock Released] Removed {self.lock_path}")
            except Exception:
                pass

    def _signal_handler(self, signum, frame):
        self.release()
        sys.exit(0)

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        if sys.platform == "win32":
            import subprocess
            try:
                output = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                return str(pid) in output
            except Exception:
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

# =========================================================================
# ATOMIC FILE AND MODEL SAVING
# =========================================================================
def save_json_atomic(data, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp_path), str(target_path))

def save_model_atomic(model, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    model.save(str(tmp_path))
    os.replace(str(tmp_path), str(target_path))

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = successes / total
    denominator = 1 + z**2 / total
    centre_adj = (p + z**2 / (2 * total)) / denominator
    std_err = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return (round(max(0.0, centre_adj - std_err), 4), round(min(1.0, centre_adj + std_err), 4))

# =========================================================================
# DATASET PREPROCESSING & PIPELINE
# =========================================================================
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

# =========================================================================
# MODEL ARCHITECTURE
# =========================================================================
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

    # Primary Binary Head (Authenticity: P(REAL))
    h1 = layers.BatchNormalization(name="h1_bn")(shared_dense)
    h1 = layers.Dropout(0.30, name="h1_drop")(h1)
    h1 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="h1_dense")(h1)
    primary_output = layers.Dense(1, activation="sigmoid", name="primary_output")(h1)

    # Auxiliary 3-Class Head (Manipulation Subtype)
    h2 = layers.BatchNormalization(name="h2_bn")(shared_dense)
    h2 = layers.Dropout(0.30, name="h2_drop")(h2)
    h2 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="h2_dense")(h2)
    auxiliary_output = layers.Dense(3, activation="softmax", name="auxiliary_output")(h2)

    model = tf.keras.Model(
        inputs=[crop_input, frame_input],
        outputs=[primary_output, auxiliary_output],
        name="TrustGuard_V8_Exp4_MultiTask"
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

    # Critical requirement: freeze ALL BatchNormalization layers across the entire model during fine-tuning
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        elif layer != base_model and not isinstance(layer, layers.InputLayer):
            layer.trainable = True

# =========================================================================
# STATE MANAGEMENT & SAFE RESUMPTION
# =========================================================================
class MultiTaskTrainingStateManager:
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
            "experiment": "V8-Exp4: Attenuated Multi-Task Auxiliary Loss (lambda_aux=0.12)",
            "status": self.status,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "completed_epoch_in_stage": self.completed_epoch_in_stage,
            "total_epochs_in_stage": self.total_stage_epochs,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
            "aux_loss_weight_alternative": AUX_LOSS_WEIGHT_ALTERNATIVE,
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
    def __init__(self, state_mgr: MultiTaskTrainingStateManager, stage_idx: int):
        super().__init__()
        self.state_mgr = state_mgr
        self.stage_idx = stage_idx
        self.stage_key = f"stage{stage_idx}"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.state_mgr.completed_epoch_in_stage += 1
        cur_epoch = self.state_mgr.completed_epoch_in_stage

        hist = self.state_mgr.history[self.stage_key]
        for k in hist.keys():
            if k in logs:
                hist[k].append(float(logs[k]))

        val_metric = logs.get("val_primary_output_loss", logs.get("val_loss"))
        if val_metric is not None and not math.isnan(val_metric):
            is_best = (self.state_mgr.best_val_loss is None) or (val_metric < self.state_mgr.best_val_loss)
            if is_best:
                self.state_mgr.best_val_loss = float(val_metric)
                self.state_mgr.best_stage = self.stage_idx
                self.state_mgr.best_epoch = cur_epoch
                print(f"\n[Checkpoint] -> New Best Primary Val Loss: {val_metric:.5f} (Stage {self.stage_idx}, Epoch {cur_epoch}). Saving best model...")
                save_model_atomic(self.model, BEST_MODEL_PATH)

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

# =========================================================================
# MAIN TRAINING AND EVALUATION PIPELINE
# =========================================================================
def main():
    start_time = time.time()
    print("=" * 80)
    print("TRUSTGUARD AI - V8-EXP4: ATTENUATED MULTI-TASK AUXILIARY LOSS")
    print(f"Auxiliary Loss Weight: lambda_aux = {AUX_LOSS_WEIGHT} (Alternative documented: {AUX_LOSS_WEIGHT_ALTERNATIVE})")
    print("=" * 80)

    # 0. Acquire Process Lock
    lock = ProcessLock(LOCKFILE_PATH)
    lock.acquire()

    # 1. Preflight Manifest Verification
    print("\n[Preflight 1/4] Verifying dataset manifest...")
    if not DEV_MANIFEST.exists():
        print(f"ERROR: Dataset manifest does not exist at {DEV_MANIFEST}")
        sys.exit(1)

    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    train_pairs = [r for r in records if r["split"] == "train"]
    val_pairs = [r for r in records if r["split"] == "validation"]
    print(f"Loaded Development Pairs: Train={len(train_pairs)}, Val={len(val_pairs)}")
    if len(train_pairs) != 4000 or len(val_pairs) != 1000:
        print(f"ERROR: Expected 4000 train and 1000 val pairs, got {len(train_pairs)} / {len(val_pairs)}")
        sys.exit(1)

    # Check for train/val data leakage
    train_crops = set(r["crop_path"] for r in train_pairs)
    val_crops = set(r["crop_path"] for r in val_pairs)
    train_frames = set(r["frame_path"] for r in train_pairs)
    val_frames = set(r["frame_path"] for r in val_pairs)
    if train_crops.intersection(val_crops) or train_frames.intersection(val_frames):
        print("ERROR: Train/validation image leakage detected!")
        sys.exit(1)
    print("[Preflight 1/4] PASSED: 0 train/val leakage, exactly 4,000 train / 1,000 val pairs.")

    # 2. Check dedicated artifact isolation
    print("\n[Preflight 2/4] Verifying dedicated Exp4 artifact paths...")
    exp3_artifacts = [
        ROOT_DIR / "models" / "deepfake_image_v8" / "deepfake_mobilenetv8_exp3_multitask_best.keras",
        ROOT_DIR / "training" / "reports" / "v8_exp3_validation_metrics.json",
        ROOT_DIR / "training" / "reports" / "v8_exp3_final_report.md"
    ]
    for exp3_art in exp3_artifacts:
        if not exp3_art.exists():
            print(f"[Warning] Exp3 artifact {exp3_art.name} does not exist.")
    if BEST_MODEL_PATH in exp3_artifacts or METRICS_JSON_PATH in exp3_artifacts:
        print("ERROR: Exp4 artifact paths collide with Exp3!")
        sys.exit(1)
    print(f"[Preflight 2/4] PASSED: Isolated checkpoint: {BEST_MODEL_PATH.name}")

    # 3. Build tf.data pipelines
    print("\n[Preflight 3/4] Initializing TensorFlow datasets...")
    train_ds = make_tf_dataset(train_pairs, is_training=True)
    val_ds = make_tf_dataset(val_pairs, is_training=False)
    print("[Preflight 3/4] PASSED: TensorFlow datasets ready.")

    # 4. State Management & Resumption
    print("\n[Preflight 4/4] Checking state management...")
    state_mgr = MultiTaskTrainingStateManager()
    has_resumed = state_mgr.load_existing()

    if has_resumed and LATEST_MODEL_PATH.exists():
        print(f"[Resumption] Loading latest model from {LATEST_MODEL_PATH}...")
        model = tf.keras.models.load_model(str(LATEST_MODEL_PATH), compile=False)
    else:
        print("[Build] Initializing fresh TrustGuard V8 Exp4 Multi-Task Model...")
        model = build_multitask_model()
        state_mgr.status = "in_progress"

    print(f"Model Parameters: Total={model.count_params():,}")
    print("[Preflight 4/4] PASSED: All preflight checks satisfied.")

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

        # Preflight verification: Confirm all BatchNormalization layers are non-trainable
        bn_trainable_count = sum(1 for l in model.layers if isinstance(l, layers.BatchNormalization) and l.trainable)
        base_m = get_base_model(model)
        bn_trainable_count += sum(1 for l in base_m.layers if isinstance(l, layers.BatchNormalization) and l.trainable)
        assert bn_trainable_count == 0, f"Error: {bn_trainable_count} BatchNormalization layers are trainable!"
        print(f"[Stage 2] Verified: All BatchNormalization layers explicitly frozen (trainable=False).")

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

    raw_outputs = best_eval_model.predict(val_ds, verbose=0)
    p_real_scores = raw_outputs[0].flatten()
    aux_probs = raw_outputs[1]
    aux_pred_classes = np.argmax(aux_probs, axis=1)

    y_true_binary = np.array([p["binary_label"] for p in val_pairs], dtype=np.int32)
    y_true_multi = np.array([p["multi_class_label"] for p in val_pairs], dtype=np.int32)
    sources = np.array([p["source_type"] for p in val_pairs])
    crop_statuses = np.array([p.get("crop_status", "UNKNOWN") for p in val_pairs])

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

    # Evaluate Subgroups at t=0.50 with Wilson 95% Confidence Intervals
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

    # Crop status within FFHQ
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

    # Auxiliary 3-Class Task Evaluation
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

    # Load Exp2 and Exp3 comparison baselines
    exp2_comp = {}
    if EXP2_METRICS_PATH.exists():
        try:
            e2 = json.loads(EXP2_METRICS_PATH.read_text(encoding="utf-8"))
            exp2_comp = {
                "accuracy": e2.get("overall_metrics", {}).get("accuracy"),
                "roc_auc": e2.get("overall_metrics", {}).get("roc_auc"),
                "deepguard_fake_recall": e2.get("subgroups", {}).get("deepguard", {}).get("fake_recall"),
                "deepguard_real_recall": e2.get("subgroups", {}).get("deepguard", {}).get("real_recall"),
                "kaggle_fake_recall": e2.get("subgroups", {}).get("kaggle", {}).get("fake_recall"),
                "kaggle_real_recall": e2.get("subgroups", {}).get("kaggle", {}).get("real_recall")
            }
        except Exception:
            pass

    exp3_comp = {}
    if EXP3_METRICS_PATH.exists():
        try:
            e3 = json.loads(EXP3_METRICS_PATH.read_text(encoding="utf-8"))
            exp3_comp = {
                "accuracy": e3.get("overall_metrics_threshold_0.50", {}).get("accuracy"),
                "roc_auc": e3.get("overall_metrics_threshold_0.50", {}).get("roc_auc"),
                "deepguard_fake_recall": e3.get("subgroups_threshold_0.50", {}).get("deepguard_face_fake", {}).get("recall"),
                "deepguard_real_recall": e3.get("subgroups_threshold_0.50", {}).get("deepguard_face_real", {}).get("recall"),
                "kaggle_fake_recall": e3.get("subgroups_threshold_0.50", {}).get("kaggle_stylegan_fake", {}).get("recall"),
                "kaggle_real_recall": e3.get("subgroups_threshold_0.50", {}).get("kaggle_ffhq_real", {}).get("recall"),
                "aux_accuracy": e3.get("auxiliary_3class_evaluation", {}).get("overall_accuracy")
            }
        except Exception:
            pass

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
        "experiment": "V8-Exp4: Attenuated Multi-Task Auxiliary Loss",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_best": str(BEST_MODEL_PATH.relative_to(ROOT_DIR)),
        "checkpoint_latest": str(LATEST_MODEL_PATH.relative_to(ROOT_DIR)),
        "aux_loss_weight": AUX_LOSS_WEIGHT,
        "aux_loss_weight_alternative": AUX_LOSS_WEIGHT_ALTERNATIVE,
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
        "auxiliary_3class_evaluation": auxiliary_evaluation,
        "comparison_baselines": {
            "exp2": exp2_comp,
            "exp3": exp3_comp
        },
        "execution_time_sec": round(time.time() - start_time, 1)
    }

    save_json_atomic(final_metrics, METRICS_JSON_PATH)
    print(f"\nSaved Completed Validation Metrics: {METRICS_JSON_PATH}")

    # Build final Markdown Report
    report_md = f"""# TrustGuard AI V8-Exp4: Attenuated Multi-Task Auxiliary Loss — Final Evaluation Report

**Author / Team:** Senior ML Engineer, TrustGuard AI  
**Date:** {time.strftime("%Y-%m-%d", time.gmtime())}  
**Experiment ID:** `V8-Exp4`  
**Best Model Checkpoint:** `{BEST_MODEL_PATH.relative_to(ROOT_DIR)}`  
**Evaluation Dataset:** 1,000 Dev-Validation Pairs (`training/reports/dataset_manifest_v8_dev.json`)  
**Overall Acceptance Decision:** **{acceptance_status}**

---

## 1. Executive Summary

TrustGuard AI **V8-Exp4** evaluated the primary hypothesis that attenuating the auxiliary multi-task manipulation head loss weight from $\\lambda_{{\\text{{aux}}}} = 0.30$ to $\\mathbf{{\\lambda_{{\\text{{aux}}}} = 0.12}}$ relieves gradient interference on video-compression artifacts, thereby recovering DeepGuard Face Real recall to $\\ge 80.00\\%$ while preserving generative feature disentanglement (Kaggle FFHQ Real recall $\\ge 70.00\\%$).

---

## 2. Acceptance Criteria Evaluation ($t = 0.50$)

| Subgroup / Criterion | Required Target | Exp2 | Exp3 | Exp4 Actual | 95% Confidence Interval | Correct / Total | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGuard Face Fake Recall** | $\\ge 80.00\\%$ | 88.80% | 93.20% | **{dg_fake_res['recall']*100:.2f}%** | {dg_fake_res['recall_95ci']} | {dg_fake_res['correct_count']} / {dg_fake_res['sample_count']} | **{'PASS' if acceptance_checks['deepguard_face_fake_recall_ge_80']['pass'] else 'FAIL'}** |
| **DeepGuard Face Real Recall** | $\\ge 80.00\\%$ | 93.60% | 77.20% | **{dg_real_res['recall']*100:.2f}%** | {dg_real_res['recall_95ci']} | {dg_real_res['correct_count']} / {dg_real_res['sample_count']} | **{'PASS' if acceptance_checks['deepguard_face_real_recall_ge_80']['pass'] else 'FAIL'}** |
| **Kaggle StyleGAN Fake Recall** | $\\ge 80.00\\%$ | 85.60% | 83.20% | **{kg_fake_res['recall']*100:.2f}%** | {kg_fake_res['recall_95ci']} | {kg_fake_res['correct_count']} / {kg_fake_res['sample_count']} | **{'PASS' if acceptance_checks['kaggle_stylegan_fake_recall_ge_80']['pass'] else 'FAIL'}** |
| **Kaggle FFHQ Real Recall** | $\\ge 70.00\\%$ | 58.80% | 80.80% | **{kg_real_res['recall']*100:.2f}%** | {kg_real_res['recall_95ci']} | {kg_real_res['correct_count']} / {kg_real_res['sample_count']} | **{'PASS' if acceptance_checks['kaggle_ffhq_real_recall_ge_70']['pass'] else 'FAIL'}** |
| **Overall Experiment Acceptance** | **4 / 4 PASS** | **1 FAIL** | **1 FAIL** | **{'4 / 4 PASS' if all_criteria_passed else 'FAIL'}** | — | — | **{acceptance_status}** |

---

## 3. Primary Authenticity Detection Performance

- **Validation Accuracy:** `{acc_050*100:.2f}%` ({acc_050*1000:.0f} / 1,000)
- **ROC-AUC:** `{auc_050:.4f}`
- **Primary Binary Confusion Matrix ($t = 0.50$):**
```
                       Actual Fake (y=0)       Actual Real (y=1)
Predicted Fake (P<0.50)     TN = {tn_050:<5}                FN = {fn_050:<5}
Predicted Real (P>=0.50)    FP = {fp_050:<5}                TP = {tp_050:<5}
Total Samples                    500                     500
```

---

## 4. Auxiliary 3-Class Manipulation Head Performance

- **Overall Auxiliary Accuracy:** `{aux_accuracy*100:.2f}%` ({aux_accuracy*1000:.0f} / 1,000)
- **3-Class Confusion Matrix ($3 \\times 3$):**
```
                             Predicted 0         Predicted 1         Predicted 2
True Class 0 (Genuine)           {aux_cm[0][0]:<5}               {aux_cm[0][1]:<5}               {aux_cm[0][2]:<5}
True Class 1 (FaceSwap)          {aux_cm[1][0]:<5}               {aux_cm[1][1]:<5}               {aux_cm[1][2]:<5}
True Class 2 (StyleGAN)          {aux_cm[2][0]:<5}               {aux_cm[2][1]:<5}               {aux_cm[2][2]:<5}
```

---

## 5. Multi-Experiment Progression (Exp2 vs. Exp3 vs. Exp4)

| Metric | Exp2 (Class-Weighted) | Exp3 (Multi-Task $\\lambda=0.30$) | Exp4 Actual ($\\lambda=0.12$) | Exp4 vs Exp3 Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 81.70% | 83.60% | **{acc_050*100:.2f}%** | {acc_050*100 - 83.60:+.2f}% |
| **ROC-AUC** | 0.9044 | 0.9237 | **{auc_050:.4f}** | {auc_050 - 0.9237:+.4f} |
| **DeepGuard Fake Recall** | 88.80% | 93.20% | **{dg_fake_res['recall']*100:.2f}%** | {dg_fake_res['recall']*100 - 93.20:+.2f}% |
| **DeepGuard Real Recall** | 93.60% | 77.20% | **{dg_real_res['recall']*100:.2f}%** | {dg_real_res['recall']*100 - 77.20:+.2f}% |
| **Kaggle Fake Recall** | 85.60% | 83.20% | **{kg_fake_res['recall']*100:.2f}%** | {kg_fake_res['recall']*100 - 83.20:+.2f}% |
| **Kaggle Real Recall** | 58.80% | 80.80% | **{kg_real_res['recall']*100:.2f}%** | {kg_real_res['recall']*100 - 80.80:+.2f}% |
| **Acceptance Decision** | NOT PASSED | NOT PASSED | **{acceptance_status}** | — |

---

## 6. Safety & Governance Confirmation
- **Sealed Test Set Quarantine:** Unbroken. `dataset_manifest_v8_test_sealed.json` and directory `training/datasets/deepfake_v8_test_sealed/` were never accessed.
- **Production Serving Code:** Untouched. `app/services/predictor.py` and V4 production weights remain intact.
- **Dataset Preprocessing:** `training/prepare_v8_dataset.py` was not executed.
"""
    FINAL_REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"Saved Final Evaluation Report: {FINAL_REPORT_PATH}")

    lock.release()

if __name__ == "__main__":
    main()
