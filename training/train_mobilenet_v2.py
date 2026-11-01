from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# TRUSTGUARD AI - MOBILE NET V2 MODEL V2
# CNN + ANN DEEPFAKE DETECTOR
# ============================================================

print("=" * 80)
print("TRUSTGUARD AI - MOBILENETV2 CNN + ANN - MODEL V2")
print("=" * 80)


# ============================================================
# PROJECT PATHS
# ============================================================

# File location:
# D:\project\DeepFake-Detector-AI\training\train_mobilenet_v2.py

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "server" / "training" / "dataset_v2"
MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Stage 1
INITIAL_EPOCHS = 10

# Stage 2
FINE_TUNE_EPOCHS = 10

SEED = 42

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# MODEL OUTPUT FILES
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR / "deepfake_mobilenetv2_v2_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "deepfake_mobilenetv2_v2.keras"
)


# ============================================================
# PRINT PATHS
# ============================================================

print()
print("Project root      :", ROOT)
print("Dataset directory :", DATASET_DIR)
print("Train directory   :", TRAIN_DIR)
print("Validation dir    :", VAL_DIR)
print("Test directory    :", TEST_DIR)
print("Model directory   :", MODEL_DIR)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Required directory not found:\n{folder}"
        )


# ============================================================
# LOAD TRAIN DATA
# ============================================================

print()
print("=" * 80)
print("LOADING TRAINING DATA")
print("=" * 80)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

print("Classes:", train_ds.class_names)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print()
print("=" * 80)
print("LOADING VALIDATION DATA")
print("=" * 80)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Classes:", val_ds.class_names)


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Classes:", test_ds.class_names)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
        ),
    ],
    name="augmentation",
)


# ============================================================
# CNN - MOBILENETV2
# ============================================================

print()
print("=" * 80)
print("BUILDING MOBILENETV2 CNN")
print("=" * 80)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Start with the CNN frozen.
base_model.trainable = False


# ============================================================
# INPUT
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="image_input",
)


# ============================================================
# AUGMENTATION
# ============================================================

x = data_augmentation(inputs)


# ============================================================
# MOBILENETV2 PREPROCESSING
# ============================================================

x = preprocess_input(x)


# ============================================================
# CNN FEATURE EXTRACTION
# ============================================================

x = base_model(
    x,
    training=False,
)


# ============================================================
# FEATURE VECTOR
# ============================================================

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


# ============================================================
# ANN CLASSIFICATION HEAD
# ============================================================

x = layers.Dropout(
    0.40,
    name="dropout_1",
)(x)

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=regularizers.l2(0.001),
    name="ann_dense_128",
)(x)

x = layers.Dropout(
    0.30,
    name="dropout_2",
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="ann_output",
)(x)


# ============================================================
# COMPLETE MODEL
# ============================================================

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="TrustGuard_MobileNetV2_CNN_ANN_V2",
)


# ============================================================
# COMPILE - STAGE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
    ],
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ============================================================
# STAGE 1 - TRAIN ANN HEAD
# ============================================================

print()
print("=" * 80)
print("STAGE 1 - TRAINING CLASSIFICATION HEAD")
print("=" * 80)

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# STAGE 2 - FINE-TUNE TOP CNN LAYERS
# ============================================================

print()
print("=" * 80)
print("STAGE 2 - FINE-TUNING MOBILENETV2")
print("=" * 80)

base_model.trainable = True


# Freeze earlier MobileNetV2 layers.
# Train only the later feature-extraction layers.

fine_tune_from = max(
    0,
    len(base_model.layers) - 30,
)

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False


# Keep BatchNorm layers frozen for stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


print(
    "Total CNN layers:",
    len(base_model.layers),
)

print(
    "Fine-tuning from layer:",
    fine_tune_from,
)

print(
    "Trainable CNN layers:",
    sum(
        1
        for layer in base_model.layers
        if layer.trainable
    ),
)


# ============================================================
# RECOMPILE FOR FINE-TUNING
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
    ],
)


# ============================================================
# FINE-TUNE
# ============================================================

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 80)
print("SAVING V2 MODEL")
print("=" * 80)

model.save(FINAL_MODEL_PATH)

print("Best V2 model:")
print(BEST_MODEL_PATH)

print()
print("Final V2 model:")
print(FINAL_MODEL_PATH)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 80)
print("FINAL V2 TEST EVALUATION")
print("=" * 80)

test_results = model.evaluate(
    test_ds,
    verbose=1,
    return_dict=True,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("MODEL V2 TEST RESULTS")
print("=" * 80)

for metric_name, metric_value in test_results.items():
    print(
        f"{metric_name:12s}: "
        f"{metric_value:.6f}"
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 80)
print("TRUSTGUARD AI MODEL V2 TRAINING COMPLETE")
print("=" * 80)

print()
print("CNN:")
print("  MobileNetV2")
print("  ImageNet pretrained")
print("  Fine-tuned final CNN layers")

print()
print("ANN:")
print("  Dense(128, ReLU)")
print("  Dense(1, Sigmoid)")

print()
print("Classes:")
print("  0 = FAKE")
print("  1 = REAL")

print()
print("Best model:")
print(BEST_MODEL_PATH)

print()
print("Final model:")
print(FINAL_MODEL_PATH)

print()
print("=" * 80)