import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

ROOT = Path(r"D:\project\DeepFake-Detector-AI")
MANIFEST = ROOT / "training/reports/dataset_manifest_v8_dev.json"
CHECKPOINT = ROOT / "models/deepfake_image_v8/deepfake_mobilenetv8_exp2_weighted_best.keras"
REPORT = ROOT / "training/reports/v8_exp2_validation_error_diagnostics.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
THRESHOLD = 0.50

def main():
    if REPORT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {REPORT}"
        )

    with MANIFEST.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = [
        r for r in manifest["records"]
        if r.get("split") == "validation"
    ]

    if not records:
        raise ValueError("No validation records found.")

    required = ("crop_path", "frame_path", "binary_label", "source_type")
    if any(any(key not in r for key in required) for r in records):
        raise ValueError("A validation record is missing a required field.")

    labels = np.asarray([int(r["binary_label"]) for r in records])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected labels 0=FAKE and 1=REAL.")

    crop_paths = [str(ROOT / r["crop_path"]) for r in records]
    frame_paths = [str(ROOT / r["frame_path"]) for r in records]

    def parse_eval(crop_path, frame_path, label):
        crop = tf.image.decode_jpeg(
            tf.io.read_file(crop_path), channels=3
        )
        crop = tf.image.resize(crop, IMAGE_SIZE)

        frame = tf.image.decode_jpeg(
            tf.io.read_file(frame_path), channels=3
        )
        frame = tf.image.resize(frame, IMAGE_SIZE)

        return {
            "crop_input": crop,
            "frame_input": frame
        }, label

    ds = tf.data.Dataset.from_tensor_slices(
        (crop_paths, frame_paths, labels.astype(np.float32))
    )
    ds = ds.map(parse_eval, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Loading existing Exp2 checkpoint...")
    model = tf.keras.models.load_model(str(CHECKPOINT), compile=False)

    print(f"Predicting {len(records)} validation samples...")
    scores = model.predict(ds, verbose=1).reshape(-1)

    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Invalid predictions or prediction count.")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected probabilities between 0 and 1.")

    predictions = (scores >= THRESHOLD).astype(int)
    rows = []

    for i, record in enumerate(records):
        actual = int(labels[i])
        predicted = int(predictions[i])
        score = float(scores[i])

        source = str(record["source_type"]).lower()
        if "deepguard" in source:
            group = "DeepGuard"
        elif "kaggle" in source:
            group = "Kaggle"
        else:
            group = "Other"

        rows.append({
            "validation_index": i,
            "source_type": record["source_type"],
            "group": group,
            "actual_label": actual,
            "actual_class": "REAL" if actual == 1 else "FAKE",
            "p_real": round(score, 8),
            "predicted_label": predicted,
            "predicted_class": "REAL" if predicted == 1 else "FAKE",
            "correct": actual == predicted,
            "error_type": (
                None if actual == predicted
                else "REAL_MISCLASSIFIED_AS_FAKE"
                if actual == 1
                else "FAKE_MISCLASSIFIED_AS_REAL"
            ),
            "distance_from_threshold": round(abs(score - THRESHOLD), 8)
        })

    errors = [r for r in rows if not r["correct"]]
    kaggle_real_errors = [
        r for r in errors
        if r["group"] == "Kaggle"
        and r["actual_label"] == 1
    ]
    kaggle_fake_errors = [
        r for r in errors
        if r["group"] == "Kaggle"
        and r["actual_label"] == 0
    ]

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    result = {
        "experiment": "V8 Exp2 validation error diagnostics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "partition": "development validation only",
        "threshold": THRESHOLD,
        "label_convention": "0=FAKE, 1=REAL; score=P(REAL)",
        "overall_confusion_matrix": {
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        },
        "validation_count": len(rows),
        "error_count": len(errors),
        "kaggle_real_misclassified_as_fake": len(kaggle_real_errors),
        "kaggle_fake_misclassified_as_real": len(kaggle_fake_errors),
        "errors": errors,
        "all_validation_predictions": rows,
        "safety_note": (
            "Validation-only diagnostic. No training, sealed-test access, "
            "production changes, or deployment."
        )
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nVALIDATION ERROR SUMMARY")
    print(f"Total validation samples: {len(rows)}")
    print(f"Total errors at threshold {THRESHOLD:.2f}: {len(errors)}")
    print(
        "Kaggle genuine images predicted FAKE: "
        f"{len(kaggle_real_errors)}"
    )
    print(
        "Kaggle fake images predicted REAL: "
        f"{len(kaggle_fake_errors)}"
    )
    print(f"Report saved: {REPORT}")

    print("\nFirst 10 Kaggle genuine-image errors:")
    for row in kaggle_real_errors[:10]:
        print(
            f"index={row['validation_index']}, "
            f"P(REAL)={row['p_real']:.4f}, "
            f"source={row['source_type']}, "
            f"crop={records[row['validation_index']]['crop_path']}"
        )

if __name__ == "__main__":
    main()
