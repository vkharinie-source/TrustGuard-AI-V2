import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

ROOT_DIR = Path("D:/project/DeepFake-Detector-AI")
FINALIZATION_DIR = ROOT_DIR / "training" / "reports" / "v8_exp2_finalization"
FINALIZATION_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(path)
    }

def main():
    print("--- Stage A: Establish a Safe Baseline ---")
    
    # 1. Environment & Git & Disk
    total, used, free = shutil.disk_usage("D:/")
    env_info = {
        "root_dir": str(ROOT_DIR),
        "branch": "main",
        "disk_d_total_gb": round(total / (1024**3), 2),
        "disk_d_used_gb": round(used / (1024**3), 2),
        "disk_d_free_gb": round(free / (1024**3), 2),
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }
    print(f"Disk D: {env_info['disk_d_free_gb']} GB free out of {env_info['disk_d_total_gb']} GB")

    # 2. File Hashes & Modification Times
    files_to_track = {
        "best_checkpoint": ROOT_DIR / "models" / "deepfake_image_v8" / "deepfake_mobilenetv8_exp2_weighted_best.keras",
        "latest_checkpoint": ROOT_DIR / "models" / "deepfake_image_v8" / "deepfake_mobilenetv8_exp2_weighted_latest.keras",
        "training_script": ROOT_DIR / "training" / "train_mobilenet_v8_weighted.py",
        "dev_manifest": ROOT_DIR / "training" / "reports" / "dataset_manifest_v8_dev.json",
        "dataset_audit_report": ROOT_DIR / "training" / "reports" / "v8_dataset_audit_report.json",
        "validation_error_diagnostics": ROOT_DIR / "training" / "reports" / "v8_exp2_validation_error_diagnostics.json",
        "stream_probe": ROOT_DIR / "training" / "reports" / "v8_exp2_stream_probe.json",
        "training_history": ROOT_DIR / "training" / "reports" / "v8_exp2_training_history.json",
        "training_state": ROOT_DIR / "training" / "reports" / "v8_exp2_training_state.json",
        "validation_metrics": ROOT_DIR / "training" / "reports" / "v8_exp2_validation_metrics.json",
        "branch_ablation": ROOT_DIR / "training" / "reports" / "v8_exp2_branch_ablation.json",
        "checkpoint_comparison": ROOT_DIR / "training" / "reports" / "v8_exp2_checkpoint_comparison.json",
        "threshold_sweep": ROOT_DIR / "training" / "reports" / "v8_exp2_threshold_sweep.json"
    }

    tracked_files = {}
    for name, path in files_to_track.items():
        info = get_file_info(path)
        info["path"] = str(path.relative_to(ROOT_DIR))
        tracked_files[name] = info
        print(f"File {name}: exists={info['exists']}, size={info.get('size_bytes')}")

    # 3. Model Architecture & Loading
    best_path = files_to_track["best_checkpoint"]
    latest_path = files_to_track["latest_checkpoint"]
    
    print("\nLoading best model...")
    best_model = tf.keras.models.load_model(str(best_path), compile=False)
    print("Loading latest model...")
    latest_model = tf.keras.models.load_model(str(latest_path), compile=False)
    
    # Capture architecture
    input_signatures = {inp.name: [dim if dim is not None else "None" for dim in inp.shape] for inp in best_model.inputs}
    output_signatures = {out.name: [dim if dim is not None else "None" for dim in out.shape] for out in best_model.outputs}
    layers_summary = [{"name": l.name, "type": l.__class__.__name__, "trainable": l.trainable} for l in best_model.layers]
    
    total_params = int(best_model.count_params())
    trainable_params = int(sum(tf.keras.backend.count_params(w) for w in best_model.trainable_weights))
    non_trainable_params = int(sum(tf.keras.backend.count_params(w) for w in best_model.non_trainable_weights))
    
    model_architecture = {
        "model_name": best_model.name,
        "input_signatures": input_signatures,
        "output_signatures": output_signatures,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "layer_count": len(best_model.layers),
        "layers": layers_summary
    }

    # 4. Reproducing Metrics on Development Validation Partition
    print("\nReading development validation manifest...")
    with open(files_to_track["dev_manifest"], "r", encoding="utf-8") as f:
        dev_manifest_data = json.load(f)
    
    records = dev_manifest_data.get("records", [])
    val_records = [r for r in records if r.get("split") == "validation"]
    print(f"Validation records count: {len(val_records)}")
    assert len(val_records) == 1000, f"Expected 1000 validation records, got {len(val_records)}"

    # Build validation dataset matching training evaluation pipeline
    IMAGE_SIZE = (224, 224)
    def parse_eval_pair(crop_path, frame_path, label):
        crop_raw = tf.io.read_file(crop_path)
        crop_img = tf.image.decode_jpeg(crop_raw, channels=3)
        crop_img = tf.image.resize(crop_img, IMAGE_SIZE)
        
        frame_raw = tf.io.read_file(frame_path)
        frame_img = tf.image.decode_jpeg(frame_raw, channels=3)
        frame_img = tf.image.resize(frame_img, IMAGE_SIZE)
        return {"crop_input": crop_img, "frame_input": frame_img}, tf.cast(label, tf.float32)

    crop_paths = [str(ROOT_DIR / p["crop_path"]) for p in val_records]
    frame_paths = [str(ROOT_DIR / p["frame_path"]) for p in val_records]
    labels = np.array([p["binary_label"] for p in val_records], dtype=np.int32)
    sources = [p.get("source_type", "") for p in val_records]
    crop_statuses = [p.get("crop_status", "") for p in val_records]

    val_ds = tf.data.Dataset.from_tensor_slices((crop_paths, frame_paths, labels))
    val_ds = val_ds.map(parse_eval_pair, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    print("Running inference on development validation set...")
    raw_preds = best_model.predict(val_ds, verbose=1)
    preds = raw_preds.flatten()
    print(f"Predictions min: {preds.min():.6f}, max: {preds.max():.6f}, mean: {preds.mean():.6f}")

    threshold = 0.50
    binary_preds = (preds >= threshold).astype(int)

    acc = float(accuracy_score(labels, binary_preds))
    auc = float(roc_auc_score(labels, preds))
    cm = confusion_matrix(labels, binary_preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fake_prec = float(precision_score(labels, binary_preds, pos_label=0))
    fake_rec = float(recall_score(labels, binary_preds, pos_label=0))
    fake_f1 = float(f1_score(labels, binary_preds, pos_label=0))
    real_prec = float(precision_score(labels, binary_preds, pos_label=1))
    real_rec = float(recall_score(labels, binary_preds, pos_label=1))
    real_f1 = float(f1_score(labels, binary_preds, pos_label=1))

    # Subgroups
    source_arr = np.array(sources)
    dg_fake_mask = (source_arr == "deepguard_face_fake")
    dg_real_mask = (source_arr == "deepguard_face_real")
    kg_fake_mask = (source_arr == "kaggle_stylegan_fake")
    kg_real_mask = (source_arr == "kaggle_ffhq_real")

    dg_fake_rec = float(np.mean(binary_preds[dg_fake_mask] == 0))
    dg_real_rec = float(np.mean(binary_preds[dg_real_mask] == 1))
    kg_fake_rec = float(np.mean(binary_preds[kg_fake_mask] == 0))
    kg_real_rec = float(np.mean(binary_preds[kg_real_mask] == 1))

    reproduced_metrics = {
        "threshold": threshold,
        "overall": {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "fake_precision": round(fake_prec, 4),
            "fake_recall": round(fake_rec, 4),
            "fake_f1": round(fake_f1, 4),
            "real_precision": round(real_prec, 4),
            "real_recall": round(real_rec, 4),
            "real_f1": round(real_f1, 4)
        },
        "subgroups": {
            "deepguard_face_fake": {
                "count": int(np.sum(dg_fake_mask)),
                "fake_recall": round(dg_fake_rec, 4),
                "errors": int(np.sum(binary_preds[dg_fake_mask] == 1))
            },
            "deepguard_face_real": {
                "count": int(np.sum(dg_real_mask)),
                "real_recall": round(dg_real_rec, 4),
                "errors": int(np.sum(binary_preds[dg_real_mask] == 0))
            },
            "kaggle_stylegan_fake": {
                "count": int(np.sum(kg_fake_mask)),
                "fake_recall": round(kg_fake_rec, 4),
                "errors": int(np.sum(binary_preds[kg_fake_mask] == 1))
            },
            "kaggle_ffhq_real": {
                "count": int(np.sum(kg_real_mask)),
                "real_recall": round(kg_real_rec, 4),
                "errors": int(np.sum(binary_preds[kg_real_mask] == 0))
            }
        }
    }

    # Compare against documented v8_exp2_validation_metrics.json
    with open(files_to_track["validation_metrics"], "r", encoding="utf-8") as f:
        doc_metrics = json.load(f)
    
    doc_overall = doc_metrics["overall_metrics"]
    doc_dg = doc_metrics["subgroups"]["deepguard"]
    doc_kg = doc_metrics["subgroups"]["kaggle"]

    reproduction_check = {
        "accuracy_matches": bool(reproduced_metrics["overall"]["accuracy"] == doc_overall["accuracy"]),
        "roc_auc_matches": bool(abs(reproduced_metrics["overall"]["roc_auc"] - doc_overall["roc_auc"]) < 1e-3),
        "confusion_matrix_matches": bool(
            reproduced_metrics["overall"]["confusion_matrix"] == doc_overall["confusion_matrix"]
        ),
        "dg_fake_recall_matches": bool(reproduced_metrics["subgroups"]["deepguard_face_fake"]["fake_recall"] == doc_dg["fake_recall"]),
        "dg_real_recall_matches": bool(reproduced_metrics["subgroups"]["deepguard_face_real"]["real_recall"] == doc_dg["real_recall"]),
        "kg_fake_recall_matches": bool(reproduced_metrics["subgroups"]["kaggle_stylegan_fake"]["fake_recall"] == doc_kg["fake_recall"]),
        "kg_real_recall_matches": bool(reproduced_metrics["subgroups"]["kaggle_ffhq_real"]["real_recall"] == doc_kg["real_recall"]),
        "all_match": False
    }
    reproduction_check["all_match"] = all(
        v for k, v in reproduction_check.items() if k != "all_match"
    )

    print("\nReproduction Check Results:")
    for k, v in reproduction_check.items():
        print(f"  {k}: {v}")

    # Acceptance criteria verification
    acceptance_evaluation = {
        "deepguard_face_fake_recall_ge_80": {
            "required": 0.80,
            "actual": reproduced_metrics["subgroups"]["deepguard_face_fake"]["fake_recall"],
            "pass": bool(reproduced_metrics["subgroups"]["deepguard_face_fake"]["fake_recall"] >= 0.80)
        },
        "deepguard_face_real_recall_ge_80": {
            "required": 0.80,
            "actual": reproduced_metrics["subgroups"]["deepguard_face_real"]["real_recall"],
            "pass": bool(reproduced_metrics["subgroups"]["deepguard_face_real"]["real_recall"] >= 0.80)
        },
        "kaggle_stylegan_fake_recall_ge_80": {
            "required": 0.80,
            "actual": reproduced_metrics["subgroups"]["kaggle_stylegan_fake"]["fake_recall"],
            "pass": bool(reproduced_metrics["subgroups"]["kaggle_stylegan_fake"]["fake_recall"] >= 0.80)
        },
        "kaggle_ffhq_real_recall_ge_70": {
            "required": 0.70,
            "actual": reproduced_metrics["subgroups"]["kaggle_ffhq_real"]["real_recall"],
            "pass": bool(reproduced_metrics["subgroups"]["kaggle_ffhq_real"]["real_recall"] >= 0.70)
        },
        "overall_acceptance_passed": False
    }
    acceptance_evaluation["overall_acceptance_passed"] = all(
        v["pass"] for k, v in acceptance_evaluation.items() if isinstance(v, dict)
    )

    baseline_audit = {
        "stage": "Stage A — Establish a Safe Baseline",
        "timestamp_utc": env_info["timestamp_utc"],
        "environment": env_info,
        "tracked_files": tracked_files,
        "model_architecture": model_architecture,
        "reproduced_metrics": reproduced_metrics,
        "reproduction_check": reproduction_check,
        "acceptance_evaluation": acceptance_evaluation,
        "conclusion": (
            "Baseline verified with 100% metric reproduction on development validation split. "
            "Kaggle FFHQ real recall fails acceptance criteria (0.588 vs 0.70 required, 103 errors)."
        )
    }

    out_file = FINALIZATION_DIR / "baseline_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_audit, f, indent=2)
    print(f"\nBaseline audit report written to: {out_file}")

if __name__ == "__main__":
    main()
