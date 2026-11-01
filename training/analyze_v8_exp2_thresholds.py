import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(r"D:\project\DeepFake-Detector-AI")
MANIFEST = ROOT / "training" / "reports" / "dataset_manifest_v8_dev.json"
CHECKPOINT = ROOT / "models" / "deepfake_image_v8" / "deepfake_mobilenetv8_exp2_weighted_best.keras"
REPORT = ROOT / "training" / "reports" / "v8_exp2_threshold_sweep.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Development manifest not found: {MANIFEST}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Exp2 checkpoint not found: {CHECKPOINT}")
    if REPORT.exists():
        raise FileExistsError(
            f"Report already exists; refusing to overwrite it: {REPORT}"
        )

    # Read only the development manifest. Never load the sealed test manifest.
    with MANIFEST.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    val_pairs = [r for r in records if r.get("split") == "validation"]

    if not val_pairs:
        raise ValueError("No validation records found.")
    if any("crop_path" not in r or "frame_path" not in r
           or "binary_label" not in r or "source_type" not in r
           for r in val_pairs):
        raise ValueError("Validation records are missing required fields.")

    labels = np.asarray([int(r["binary_label"]) for r in val_pairs])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected labels: 0=FAKE, 1=REAL.")

    # Reproduce the original evaluation preprocessing: decode JPEG and resize.
    def parse_eval(crop_path, frame_path, label):
        crop = tf.io.decode_jpeg(tf.io.read_file(crop_path), channels=3)
        crop = tf.image.resize(crop, IMAGE_SIZE)
        frame = tf.io.decode_jpeg(tf.io.read_file(frame_path), channels=3)
        frame = tf.image.resize(frame, IMAGE_SIZE)
        return {"crop_input": crop, "frame_input": frame}, label

    crop_paths = [str(ROOT / r["crop_path"]) for r in val_pairs]
    frame_paths = [str(ROOT / r["frame_path"]) for r in val_pairs]

    ds = tf.data.Dataset.from_tensor_slices(
        (crop_paths, frame_paths, labels.astype(np.float32))
    )
    ds = ds.map(parse_eval, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"Loading existing checkpoint: {CHECKPOINT}")
    model = tf.keras.models.load_model(str(CHECKPOINT), compile=False)

    print(f"Predicting on {len(val_pairs)} validation records only...")
    scores = model.predict(ds, verbose=1).reshape(-1)

    if len(scores) != len(labels):
        raise ValueError("Prediction count does not match validation label count.")
    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected sigmoid probabilities between 0 and 1.")

    source = [str(r["source_type"]).lower() for r in val_pairs]
    groups = {
        "deepguard": np.asarray(["deepguard" in s for s in source]),
        "kaggle": np.asarray(["kaggle" in s for s in source]),
    }
    if any(mask.sum() == 0 for mask in groups.values()):
        raise ValueError("DeepGuard or Kaggle validation group is empty.")

    def evaluate(mask, threshold):
        y = labels[mask]
        pred = (scores[mask] >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = map(int, cm.ravel())
        fake_total = tn + fp
        real_total = fn + tp
        return {
            "samples": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "fake_recall": round(tn / fake_total, 4) if fake_total else None,
            "real_recall": round(tp / real_total, 4) if real_total else None,
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        }

    rows = []
    # Include the original 0.50 threshold and sweep in 0.01 increments.
    thresholds = sorted(set(np.round(np.arange(0.01, 1.00, 0.01), 2).tolist()
                            + [0.50]))

    for threshold in thresholds:
        overall = evaluate(np.ones(len(labels), dtype=bool), threshold)
        dg = evaluate(groups["deepguard"], threshold)
        kg = evaluate(groups["kaggle"], threshold)

        checks = {
            "deepguard_fake_recall_ge_0_60":
                dg["fake_recall"] is not None and dg["fake_recall"] >= 0.60,
            "deepguard_real_recall_ge_0_90":
                dg["real_recall"] is not None and dg["real_recall"] >= 0.90,
            "kaggle_fake_recall_ge_0_80":
                kg["fake_recall"] is not None and kg["fake_recall"] >= 0.80,
            "kaggle_real_recall_ge_0_70":
                kg["real_recall"] is not None and kg["real_recall"] >= 0.70,
        }
        rows.append({
            "threshold": float(threshold),
            "overall": overall,
            "deepguard": dg,
            "kaggle": kg,
            "acceptance_checks": checks,
            "all_four_pass": all(checks.values()),
        })

    baseline = next(r for r in rows if r["threshold"] == 0.50)
    passing = [r for r in rows if r["all_four_pass"]]

    # Highest overall accuracy among thresholds meeting all four targets.
    # If none pass, report the best target count, then accuracy.
    if passing:
        selected = max(passing, key=lambda r: r["overall"]["accuracy"])
        selection_reason = "Highest overall accuracy among thresholds passing all four targets."
    else:
        selected = max(
            rows,
            key=lambda r: (
                sum(r["acceptance_checks"].values()),
                r["overall"]["accuracy"]
            )
        )
        selection_reason = (
            "No threshold passes all four targets; shown threshold maximizes "
            "number of targets passed, then overall accuracy. Not a deployment selection."
        )

    result = {
        "experiment": "TrustGuard AI V8 Exp2 threshold sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "partition": "development validation only",
        "validation_samples": int(len(labels)),
        "label_convention": "0=FAKE, 1=REAL; sigmoid output=P(REAL)",
        "threshold_rule": "REAL when score >= threshold; otherwise FAKE",
        "validation_auc": round(float(roc_auc_score(labels, scores)), 4),
        "threshold_0_50_baseline": baseline,
        "all_four_target_passing_thresholds": passing,
        "selected_for_analysis_only": selected,
        "selection_reason": selection_reason,
        "threshold_results": rows,
        "safety_note": (
            "Validation analysis only. No training, no sealed-test access, "
            "no production-model changes, and no deployment decision."
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print("V8 EXP2 VALIDATION THRESHOLD SWEEP")
    print("=" * 72)
    print(f"Validation samples: {len(labels)}")
    print(f"Validation AUC: {result['validation_auc']:.4f}")
    print(f"Thresholds passing all four targets: {len(passing)}")
    print(f"Report saved: {REPORT}")
    print("\nThreshold 0.50 baseline:")
    print(json.dumps(baseline, indent=2))
    print("\nAnalysis candidate:")
    print(f"  Threshold: {selected['threshold']:.2f}")
    print(f"  Accuracy: {selected['overall']['accuracy']:.4f}")
    print(f"  Targets passed: {sum(selected['acceptance_checks'].values())}/4")
    print(f"  Reason: {selection_reason}")
    print("\nThis is validation analysis only, not approval to deploy.")

if __name__ == "__main__":
    main()
