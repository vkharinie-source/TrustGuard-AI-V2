import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "training/reports/dataset_manifest_v8_dev.json"
OUTPUT = ROOT / "training/reports/v8_exp2_checkpoint_comparison.json"

CHECKPOINTS = {
    "best": ROOT / "models/deepfake_image_v8/deepfake_mobilenetv8_exp2_weighted_best.keras",
    "latest": ROOT / "models/deepfake_image_v8/deepfake_mobilenetv8_exp2_weighted_latest.keras",
}
THRESHOLD = 0.5
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)


def load_image(path):
    raw = tf.io.read_file(str(ROOT / path))
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image.numpy()


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["split"] == "validation"]

    if len(records) != 1000:
        raise ValueError(f"Expected 1000 validation records, got {len(records)}")

    labels = np.asarray([r["binary_label"] for r in records], dtype=np.int32)
    crops = np.stack([load_image(r["crop_path"]) for r in records])
    frames = np.stack([load_image(r["frame_path"]) for r in records])

    print("Validation records:", len(records))
    print("Labels: 0=FAKE, 1=REAL")
    print("Images loaded. No training will run.")

    report = {
        "description": "Read-only comparison of existing V8 Exp2 checkpoints",
        "partition": "development validation only",
        "threshold": THRESHOLD,
        "label_convention": "0=FAKE, 1=REAL; score=P(REAL)",
        "validation_count": len(records),
        "checkpoints": {},
    }

    for name, checkpoint in CHECKPOINTS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        print(f"\nEvaluating {name}: {checkpoint.name}")
        model = tf.keras.models.load_model(str(checkpoint), compile=False)
        scores = model.predict(
            {"crop_input": crops, "frame_input": frames},
            batch_size=BATCH_SIZE,
            verbose=1,
        ).reshape(-1)

        predictions = (scores >= THRESHOLD).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()

        groups = defaultdict(list)
        for i, record in enumerate(records):
            groups[record["source_type"]].append(i)

        subgroup_results = {}
        for source, indices in sorted(groups.items()):
            idx = np.asarray(indices)
            y = labels[idx]
            pred = predictions[idx]
            source_scores = scores[idx]

            fake_mask = y == 0
            real_mask = y == 1

            subgroup_results[source] = {
                "count": int(len(idx)),
                "accuracy": round(float(accuracy_score(y, pred)), 4),
                "fake_recall": (
                    round(float(np.mean(pred[fake_mask] == 0)), 4)
                    if fake_mask.any() else None
                ),
                "real_recall": (
                    round(float(np.mean(pred[real_mask] == 1)), 4)
                    if real_mask.any() else None
                ),
                "mean_p_real": round(float(np.mean(source_scores)), 6),
            }

        result = {
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
            "confusion_matrix": {
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
            },
            "fake_recall": round(float(tn / (tn + fp)), 4) if tn + fp else None,
            "real_recall": round(float(tp / (tp + fn)), 4) if tp + fn else None,
            "subgroups": subgroup_results,
        }
        report["checkpoints"][name] = result

        print("Accuracy:", result["accuracy"])
        print("AUC:", result["roc_auc"])
        print("Fake recall:", result["fake_recall"])
        print("Real recall:", result["real_recall"])
        print("Kaggle FFHQ genuine recall:",
              subgroup_results["kaggle_ffhq_real"]["real_recall"])

        del model

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nComparison report saved to:", OUTPUT)
    print("No model weights or training files were modified.")


if __name__ == "__main__":
    main()
