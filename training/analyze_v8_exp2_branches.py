import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

ROOT = Path.cwd()
MANIFEST = ROOT / "training/reports/dataset_manifest_v8_dev.json"
MODEL_PATH = ROOT / "models/deepfake_image_v8/deepfake_mobilenetv8_exp2_weighted_best.keras"
OUT = ROOT / "training/reports/v8_exp2_branch_ablation.json"
BATCH_SIZE = 16

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
records = manifest["records"] if isinstance(manifest, dict) else manifest
pairs = [r for r in records if r.get("split") == "validation"]

def load_image(path):
    raw = tf.io.read_file(str(ROOT / path))
    image = tf.image.decode_jpeg(raw, channels=3)
    return tf.image.resize(image, (224, 224))

crop_images, frame_images, labels = [], [], []
for r in pairs:
    crop_images.append(load_image(r["crop_path"]).numpy())
    frame_images.append(load_image(r["frame_path"]).numpy())
    labels.append(int(r["binary_label"]))

crop_images = np.asarray(crop_images, dtype=np.float32)
frame_images = np.asarray(frame_images, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int32)

print("Loading checkpoint for inference only...")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print("Model inputs:", model.input_names if hasattr(model, "input_names") else [x.name for x in model.inputs])

def evaluate(name, crop, frame):
    scores = model.predict(
        {"crop_input": crop, "frame_input": frame},
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    pred = (scores >= 0.5).astype(np.int32)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    result = {
        "test": name,
        "accuracy": float(accuracy_score(labels, pred)),
        "auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix_labels_0_1": cm.tolist(),
        "kaggle_ffhq_real_recall": float(np.mean(
            pred[[i for i, r in enumerate(pairs) if r["source_type"] == "kaggle_ffhq_real"]] == 1
        )),
    }
    print(
        f'{name}: accuracy={result["accuracy"]:.4f}, '
        f'AUC={result["auc"]:.4f}, '
        f'FFHQ-real recall={result["kaggle_ffhq_real_recall"]:.4f}'
    )
    return result

# Baseline: original two inputs
baseline = evaluate("both_original", crop_images, frame_images)

# Ablation: replace one stream with a neutral gray image.
# This is diagnostic only; it is not a valid deployment configuration.
gray = np.full_like(crop_images, 127.5, dtype=np.float32)
crop_only = evaluate("crop_with_gray_frame", crop_images, gray)
frame_only = evaluate("gray_crop_with_frame", gray, frame_images)

report = {
    "checkpoint": str(MODEL_PATH),
    "validation_records": len(pairs),
    "threshold": 0.5,
    "warning": "Branch ablations use artificial gray inputs and are diagnostic only, not deployment performance.",
    "results": [baseline, crop_only, frame_only],
}
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Saved:", OUT)
