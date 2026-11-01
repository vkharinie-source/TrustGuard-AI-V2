from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# TRUSTGUARD AI - MODEL V3
# FINE-TUNE EXISTING MOBILENETV2 CNN + ANN
# ============================================================

print("=" * 80)
print("TRUSTGUARD AI - MODEL V3 FINE-TUNING")
print("=" * 80)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "server" / "training" / "dataset_v3"
MODEL_DIR = ROOT / "model"

OLD_MODEL_PATH = MODEL_DIR / "deepfake_mobilenetv2_best.keras"

BEST_MODEL_PATH = MODEL_DIR / "deepfake_mobilenetv2_v3_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "deepfake_mobilenetv2_v3.keras"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


# ============================================================
# CHECK PATHS
# ============================================================

if not OLD_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Existing V1 model not found:\n{OLD_MODEL_PATH}"
    )

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{folder}"
        )


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 80)
print("LOADING V3 DATASET")
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

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Classes:", train_ds.class_names)


# ============================================================
# PREFETCH
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# LOAD EXISTING V1 MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING EXISTING V1 MODEL")
print("=" * 80)

model = tf.keras.models.load_model(
    OLD_MODEL_PATH,
    compile=False,
)

print("Loaded:")
print(OLD_MODEL_PATH)

print()
print("Model name:")
print(model.name)


# ============================================================
# FIND MOBILENETV2 BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break

if base_model is None:
    raise RuntimeError(
        "MobileNetV2 base model could not be found."
    )

print()
print("CNN base model:")
print(base_model.name)


# ============================================================
# FREEZE MOST CNN LAYERS
# ============================================================

base_model.trainable = True

# Fine-tune only the last 30 layers.
fine_tune_from = max(
    0,
    len(base_model.layers) - 30,
)

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False


# Keep BatchNormalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


print()
print("Total MobileNetV2 layers:")
print(len(base_model.layers))

print("Fine-tuning from layer:")
print(fine_tune_from)

print(
    "Trainable CNN layers:",
    sum(
        1
        for layer in base_model.layers
        if layer.trainable
    ),
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5,
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
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ============================================================
# TRAIN V3
# ============================================================

print()
print("=" * 80)
print("STARTING V3 FINE-TUNING")
print("=" * 80)

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
print("SAVING V3 MODEL")
print("=" * 80)

model.save(FINAL_MODEL_PATH)

print("Best model:")
print(BEST_MODEL_PATH)

print()
print("Final model:")
print(FINAL_MODEL_PATH)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 80)
print("V3 FINAL TEST")
print("=" * 80)

results = model.evaluate(
    test_ds,
    verbose=1,
    return_dict=True,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("V3 TEST RESULTS")
print("=" * 80)

for name, value in results.items():
    print(
        f"{name:12s}: {value:.6f}"
    )


print()
print("=" * 80)
print("V3 TRAINING COMPLETE")
print("=" * 80)

print("CNN: MobileNetV2")
print("ANN: Dense classification head")

print()
print("Best V3 model:")
print(BEST_MODEL_PATH)

print()
print("Final V3 model:")
print(FINAL_MODEL_PATH)

print("=" * 80)