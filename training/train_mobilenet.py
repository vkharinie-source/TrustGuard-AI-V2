from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# TRUSTGUARD AI
# MOBILE NET V2 CNN + ANN DEEPFAKE DETECTOR
# ============================================================

print("=" * 80)
print("TRUSTGUARD AI - MOBILENETV2 CNN + ANN TRAINING")
print("=" * 80)


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is:
# D:\project\DeepFake-Detector-AI\training\train_mobilenet.py
#
# parents[0] = training
# parents[1] = DeepFake-Detector-AI

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "server" / "training" / "dataset"
MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# PRINT PATH INFORMATION
# ============================================================

print()
print("PROJECT ROOT      :", ROOT)
print("DATASET DIRECTORY :", DATASET_DIR)
print("TRAIN DIRECTORY   :", TRAIN_DIR)
print("VALIDATION DIR    :", VAL_DIR)
print("TEST DIRECTORY    :", TEST_DIR)
print("MODEL DIRECTORY   :", MODEL_DIR)


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATASET_DIR}"
    )

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING TRAINING DATASET")
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

print("Training classes:", train_ds.class_names)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING VALIDATION DATASET")
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

print("Validation classes:", val_ds.class_names)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING TEST DATASET")
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

print("Test classes:", test_ds.class_names)


# ============================================================
# PERFORMANCE OPTIMIZATION
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
    ],
    name="augmentation",
)


# ============================================================
# CNN PART - MOBILENETV2
# ============================================================

print()
print("=" * 80)
print("BUILDING CNN - MOBILENETV2")
print("=" * 80)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze the pretrained CNN.
base_model.trainable = False

print("CNN model: MobileNetV2")
print("ImageNet weights: Yes")
print("CNN trainable: False")


# ============================================================
# INPUT LAYER
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="image_input",
)


# ============================================================
# IMAGE AUGMENTATION
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
# CONVERT CNN FEATURE MAPS TO VECTOR
# ============================================================

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


# ============================================================
# REGULARIZATION
# ============================================================

x = layers.Dropout(
    0.40,
    name="dropout_1"
)(x)


# ============================================================
# ANN PART - DENSE LAYER
# ============================================================

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=regularizers.l2(0.001),
    name="ann_dense_128",
)(x)


# ============================================================
# SECOND DROPOUT
# ============================================================

x = layers.Dropout(
    0.30,
    name="dropout_2"
)(x)


# ============================================================
# ANN OUTPUT / CLASSIFICATION LAYER
# ============================================================

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
    name="TrustGuard_MobileNetV2_CNN_ANN",
)


# ============================================================
# COMPILE MODEL
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
# DISPLAY MODEL
# ============================================================

print()
print("=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

model.summary()


# ============================================================
# MODEL SAVE PATHS
# ============================================================

best_model_path = (
    MODEL_DIR / "deepfake_mobilenetv2_best.keras"
)

final_model_path = (
    MODEL_DIR / "deepfake_mobilenetv2.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Save the model with the best validation AUC
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    # Reduce learning rate when validation AUC stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),

    # Stop training when validation AUC stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ============================================================
# START TRAINING
# ============================================================

print()
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)

print("Epochs      :", EPOCHS)
print("Batch size  :", BATCH_SIZE)
print("Image size  :", IMG_SIZE)
print("Classes     :", ["fake", "real"])
print()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 80)
print("SAVING FINAL MODEL")
print("=" * 80)

model.save(final_model_path)

print("Final model saved to:")
print(final_model_path)

print()
print("Best model saved to:")
print(best_model_path)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

test_results = model.evaluate(
    test_ds,
    verbose=1,
    return_dict=True,
)


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print()
print("=" * 80)
print("TEST RESULTS")
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
print("TRAINING COMPLETED")
print("=" * 80)

print()
print("CNN:")
print("  MobileNetV2")
print("  Pretrained on ImageNet")
print("  Frozen during training")

print()
print("ANN:")
print("  Dense(128, ReLU)")
print("  Dense(1, Sigmoid)")

print()
print("Classification:")
print("  0 = FAKE")
print("  1 = REAL")

print()
print("Best model:")
print(best_model_path)

print()
print("Final model:")
print(final_model_path)

print()
print("=" * 80)
print("TRUSTGUARD AI TRAINING FINISHED")
print("=" * 80)