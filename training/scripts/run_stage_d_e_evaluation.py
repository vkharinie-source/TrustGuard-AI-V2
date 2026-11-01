import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

ROOT_DIR = Path("D:/project/DeepFake-Detector-AI")
FINALIZATION_DIR = ROOT_DIR / "training" / "reports" / "v8_exp2_finalization"
FINALIZATION_DIR.mkdir(parents=True, exist_ok=True)

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
    adjusted_std_error = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    lower = max(0.0, centre_adjusted_probability - adjusted_std_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_std_error)
    return (round(lower, 4), round(upper, 4))

def compute_metrics(labels: np.ndarray, preds: np.ndarray, threshold: float):
    binary_preds = (preds >= threshold).astype(int)
    cm = confusion_matrix(labels, binary_preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(labels, binary_preds))
    auc = float(roc_auc_score(labels, preds))

    # Recall for FAKE (0) = TN / (TN + FP)
    fake_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    # Recall for REAL (1) = TP / (TP + FN)
    real_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    # Specificity = TN / (TN + FP) = fake_rec
    specificity = fake_rec
    # Balanced Accuracy = (fake_rec + real_rec) / 2
    bal_acc = float((fake_rec + real_rec) / 2.0)

    fake_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    real_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    fake_f1 = float(2 * fake_prec * fake_rec / (fake_prec + fake_rec)) if (fake_prec + fake_rec) > 0 else 0.0
    real_f1 = float(2 * real_prec * real_rec / (real_prec + real_rec)) if (real_prec + real_rec) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "accuracy_95ci": list(wilson_score_interval(int(np.sum(binary_preds == labels)), len(labels))),
        "roc_auc": round(auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "specificity": round(specificity, 4),
        "fake_precision": round(fake_prec, 4),
        "fake_recall": round(fake_rec, 4),
        "fake_f1": round(fake_f1, 4),
        "real_precision": round(real_prec, 4),
        "real_recall": round(real_rec, 4),
        "real_f1": round(real_f1, 4),
        "confusion_matrix": {
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "convention": "0=FAKE (negative), 1=REAL (positive); TN=pred 0 actual 0, FP=pred 1 actual 0, FN=pred 0 actual 1, TP=pred 1 actual 1"
        },
        "errors": {
            "real_misclassified_as_fake_FN": fn,
            "fake_misclassified_as_real_FP": fp,
            "total_errors": fn + fp
        }
    }

def main():
    print("--- Stage D & E: Experiment Ledger & Validation Comparison ---")

    # Load dev validation manifest
    with open(ROOT_DIR / "training" / "reports" / "dataset_manifest_v8_dev.json", "r", encoding="utf-8") as f:
        dev_manifest = json.load(f)
    val_records = [r for r in dev_manifest["records"] if r.get("split") == "validation"]
    print(f"Validation samples: {len(val_records)}")

    # Load Exp2 model
    exp2_path = ROOT_DIR / "models" / "deepfake_image_v8" / "deepfake_mobilenetv8_exp2_weighted_best.keras"
    exp2_model = tf.keras.models.load_model(str(exp2_path), compile=False)

    # Load Exp0 model for comparison
    exp0_path = ROOT_DIR / "models" / "deepfake_image_v8" / "deepfake_mobilenetv8_exp0_best.keras"
    has_exp0 = exp0_path.exists()
    exp0_model = tf.keras.models.load_model(str(exp0_path), compile=False) if has_exp0 else None

    # Dataset builder
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
    sources = np.array([p.get("source_type", "") for p in val_records])
    crop_statuses = np.array([p.get("crop_status", "") for p in val_records])

    val_ds = tf.data.Dataset.from_tensor_slices((crop_paths, frame_paths, labels))
    val_ds = val_ds.map(parse_eval_pair, num_parallel_calls=tf.data.AUTOTUNE).batch(32)

    print("Generating predictions for Exp2...")
    exp2_preds = exp2_model.predict(val_ds, verbose=0).flatten()

    exp0_preds = None
    if exp0_model:
        print("Generating predictions for Exp0...")
        exp0_preds = exp0_model.predict(val_ds, verbose=0).flatten()

    # Subgroup helper
    def evaluate_all_subgroups(preds_arr, threshold):
        overall = compute_metrics(labels, preds_arr, threshold)
        subgroup_res = {}
        for src in ["deepguard_face_fake", "deepguard_face_real", "kaggle_stylegan_fake", "kaggle_ffhq_real"]:
            mask = (sources == src)
            src_lbls = labels[mask]
            src_preds = preds_arr[mask]
            src_bin = (src_preds >= threshold).astype(int)
            src_correct = int(np.sum(src_bin == src_lbls))
            src_total = len(src_lbls)
            target_class = src_lbls[0]

            if target_class == 0:
                rec = round(float(np.mean(src_bin == 0)), 4)
                rec_ci = wilson_score_interval(int(np.sum(src_bin == 0)), src_total)
            else:
                rec = round(float(np.mean(src_bin == 1)), 4)
                rec_ci = wilson_score_interval(int(np.sum(src_bin == 1)), src_total)

            subgroup_res[src] = {
                "sample_count": src_total,
                "correct_count": src_correct,
                "error_count": src_total - src_correct,
                "recall": rec,
                "recall_95ci": list(rec_ci),
                "mean_p_real": round(float(np.mean(src_preds)), 6)
            }

        # Crop status within FFHQ
        ffhq_mask = (sources == "kaggle_ffhq_real")
        ffhq_det_mask = ffhq_mask & (crop_statuses == "FACE_DETECTED")
        ffhq_fall_mask = ffhq_mask & (crop_statuses == "FALLBACK_CENTRAL_80")

        crop_status_res = {
            "FACE_DETECTED": {
                "count": int(np.sum(ffhq_det_mask)),
                "correct": int(np.sum((preds_arr[ffhq_det_mask] >= threshold) == 1)),
                "recall": round(float(np.mean((preds_arr[ffhq_det_mask] >= threshold) == 1)), 4),
                "recall_95ci": list(wilson_score_interval(int(np.sum((preds_arr[ffhq_det_mask] >= threshold) == 1)), int(np.sum(ffhq_det_mask))))
            },
            "FALLBACK_CENTRAL_80": {
                "count": int(np.sum(ffhq_fall_mask)),
                "correct": int(np.sum((preds_arr[ffhq_fall_mask] >= threshold) == 1)),
                "recall": round(float(np.mean((preds_arr[ffhq_fall_mask] >= threshold) == 1)), 4),
                "recall_95ci": list(wilson_score_interval(int(np.sum((preds_arr[ffhq_fall_mask] >= threshold) == 1)), int(np.sum(ffhq_fall_mask))))
            }
        }

        # Check authoritative acceptance criteria
        acceptance = {
            "deepguard_face_fake_recall_ge_80": {
                "target": 0.80,
                "actual": subgroup_res["deepguard_face_fake"]["recall"],
                "pass": bool(subgroup_res["deepguard_face_fake"]["recall"] >= 0.80)
            },
            "deepguard_face_real_recall_ge_80": {
                "target": 0.80,
                "actual": subgroup_res["deepguard_face_real"]["recall"],
                "pass": bool(subgroup_res["deepguard_face_real"]["recall"] >= 0.80)
            },
            "kaggle_stylegan_fake_recall_ge_80": {
                "target": 0.80,
                "actual": subgroup_res["kaggle_stylegan_fake"]["recall"],
                "pass": bool(subgroup_res["kaggle_stylegan_fake"]["recall"] >= 0.80)
            },
            "kaggle_ffhq_real_recall_ge_70": {
                "target": 0.70,
                "actual": subgroup_res["kaggle_ffhq_real"]["recall"],
                "pass": bool(subgroup_res["kaggle_ffhq_real"]["recall"] >= 0.70)
            }
        }
        all_passed = all(v["pass"] for v in acceptance.values())
        acceptance["overall_status"] = "PASSED" if all_passed else "NOT PASSED"

        return {
            "overall_metrics": overall,
            "subgroups": subgroup_res,
            "crop_status_within_ffhq": crop_status_res,
            "acceptance_criteria": acceptance
        }

    # Evaluate Exp2 at default threshold 0.50
    exp2_eval_050 = evaluate_all_subgroups(exp2_preds, 0.50)
    # Evaluate Exp2 at Youden J best threshold (0.43)
    exp2_eval_043 = evaluate_all_subgroups(exp2_preds, 0.43)

    # Evaluate Exp0 if present
    exp0_eval_050 = evaluate_all_subgroups(exp0_preds, 0.50) if exp0_preds is not None else None
    exp0_eval_calibrated = evaluate_all_subgroups(exp0_preds, 0.6458) if exp0_preds is not None else None

    # Validation comparison report
    validation_comparison = {
        "stage": "Stage E — Development-Validation Acceptance",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_partition": "Development Validation Set Only (1,000 samples, 500 Real / 500 Fake)",
        "models_evaluated": {
            "v8_exp2_weighted_default_threshold_0.50": {
                "description": "TrustGuard V8 Exp2 (Cost-Sensitive Loss w_fake=1.15, w_real=1.00) at default threshold 0.50",
                "checkpoint": "models/deepfake_image_v8/deepfake_mobilenetv8_exp2_weighted_best.keras",
                "results": exp2_eval_050
            },
            "v8_exp2_weighted_calibrated_threshold_0.43": {
                "description": "TrustGuard V8 Exp2 (Cost-Sensitive Loss w_fake=1.15, w_real=1.00) at exploratory calibrated threshold 0.43",
                "checkpoint": "models/deepfake_image_v8/deepfake_mobilenetv8_exp2_weighted_best.keras",
                "results": exp2_eval_043
            },
            "v8_exp0_baseline_uniform_threshold_0.50": {
                "description": "TrustGuard V8 Exp0 Baseline (Uniform Loss w_fake=1.00, w_real=1.00) at default threshold 0.50",
                "checkpoint": "models/deepfake_image_v8/deepfake_mobilenetv8_exp0_best.keras",
                "results": exp0_eval_050
            },
            "v8_exp0_baseline_uniform_calibrated_0.6458": {
                "description": "TrustGuard V8 Exp0 Baseline at documented calibrated threshold 0.6458",
                "checkpoint": "models/deepfake_image_v8/deepfake_mobilenetv8_exp0_best.keras",
                "results": exp0_eval_calibrated
            }
        },
        "side_by_side_summary": {
            "metric": [
                "Overall Accuracy",
                "Overall ROC-AUC",
                "DeepGuard Fake Recall (target >= 80%)",
                "DeepGuard Real Recall (target >= 80%)",
                "Kaggle StyleGAN Fake Recall (target >= 80%)",
                "Kaggle FFHQ Real Recall (target >= 70%)",
                "Final Acceptance Decision"
            ],
            "v8_exp0_uniform_t0.50": [
                exp0_eval_050["overall_metrics"]["accuracy"] if exp0_eval_050 else "N/A",
                exp0_eval_050["overall_metrics"]["roc_auc"] if exp0_eval_050 else "N/A",
                exp0_eval_050["subgroups"]["deepguard_face_fake"]["recall"] if exp0_eval_050 else "N/A",
                exp0_eval_050["subgroups"]["deepguard_face_real"]["recall"] if exp0_eval_050 else "N/A",
                exp0_eval_050["subgroups"]["kaggle_stylegan_fake"]["recall"] if exp0_eval_050 else "N/A",
                exp0_eval_050["subgroups"]["kaggle_ffhq_real"]["recall"] if exp0_eval_050 else "N/A",
                exp0_eval_050["acceptance_criteria"]["overall_status"] if exp0_eval_050 else "N/A"
            ],
            "v8_exp2_weighted_t0.50": [
                exp2_eval_050["overall_metrics"]["accuracy"],
                exp2_eval_050["overall_metrics"]["roc_auc"],
                exp2_eval_050["subgroups"]["deepguard_face_fake"]["recall"],
                exp2_eval_050["subgroups"]["deepguard_face_real"]["recall"],
                exp2_eval_050["subgroups"]["kaggle_stylegan_fake"]["recall"],
                exp2_eval_050["subgroups"]["kaggle_ffhq_real"]["recall"],
                exp2_eval_050["acceptance_criteria"]["overall_status"]
            ],
            "v8_exp2_weighted_calibrated_t0.43": [
                exp2_eval_043["overall_metrics"]["accuracy"],
                exp2_eval_043["overall_metrics"]["roc_auc"],
                exp2_eval_043["subgroups"]["deepguard_face_fake"]["recall"],
                exp2_eval_043["subgroups"]["deepguard_face_real"]["recall"],
                exp2_eval_043["subgroups"]["kaggle_stylegan_fake"]["recall"],
                exp2_eval_043["subgroups"]["kaggle_ffhq_real"]["recall"],
                exp2_eval_043["acceptance_criteria"]["overall_status"]
            ]
        },
        "acceptance_conclusion": {
            "status": "NOT PASSED",
            "failing_criterion": "kaggle_ffhq_real_recall_ge_70 (actual: 0.588 at t=0.50, 0.644 at t=0.43; required: >= 0.700)",
            "gap": "-11.2 percentage points at default threshold; -5.6 percentage points at best exploratory threshold",
            "recommendation": (
                "Do not promote Exp2 to production or sealed-test evaluation. "
                "Keep sealed test quarantined. "
                "The class weighting strategy in Exp2 has reached its Pareto frontier. "
                "Follow the project experiment plan to proceed to Exp3 (Multi-Task Auxiliary Manipulation Head) "
                "to separate generative synthesis features from boundary face-swap features without global loss distortion."
            )
        }
    }

    out_comp = FINALIZATION_DIR / "validation_comparison.json"
    with open(out_comp, "w", encoding="utf-8") as f:
        json.dump(validation_comparison, f, indent=2)
    print(f"Validation comparison written to: {out_comp}")

    # Experiment ledger
    experiment_ledger = {
        "project": "TrustGuard AI V8 Controlled Experiments",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT_DIR),
        "experiments": [
            {
                "experiment_id": "V8-Exp0",
                "name": "Dual-Stream MobileNetV2 Baseline Control",
                "hypothesis": "Shared MobileNetV2 dual-stream (crop + frame) with uniform loss (1.0:1.0) establishes baseline cross-domain performance.",
                "loss_configuration": "Binary Cross-Entropy (w_fake=1.00, w_real=1.00)",
                "script": "training/train_mobilenet_v8_baseline.py",
                "seed": 42,
                "batch_size": 32,
                "data_split": "dataset_manifest_v8_dev.json (4,000 train / 1,000 val)",
                "checkpoint": "models/deepfake_image_v8/deepfake_mobilenetv8_exp0_best.keras",
                "key_metrics_val": {
                    "accuracy": 0.770,
                    "roc_auc": 0.8402,
                    "deepguard_fake_recall": 0.840,
                    "deepguard_real_recall": 0.756,
                    "kaggle_fake_recall": 0.760,
                    "kaggle_real_recall": 0.724
                },
                "decision": "COMPLETED. Failed Kaggle fake recall (0.760 < 0.800 target). Led to Exp1 and Exp2."
            },
            {
                "experiment_id": "V8-Exp1",
                "name": "Tri-Stream High-Pass Frequency Residual Branch",
                "hypothesis": "High-pass spatial residual branch isolates sensor noise and deconvolution patterns to catch generative fakes.",
                "loss_configuration": "Binary Cross-Entropy (w_fake=1.00, w_real=1.00)",
                "script": "training/train_mobilenet_v8_frequency.py",
                "seed": 42,
                "batch_size": 32,
                "data_split": "dataset_manifest_v8_dev.json (4,000 train / 1,000 val)",
                "checkpoint": "models/deepfake_image_v8/deepfake_mobilenetv8_exp1_frequency_best.keras",
                "key_metrics_val": {
                    "accuracy": 0.758,
                    "roc_auc": 0.8250,
                    "deepguard_fake_recall": 0.864,
                    "deepguard_real_recall": 0.892,
                    "kaggle_fake_recall": 0.792,
                    "kaggle_real_recall": 0.484
                },
                "decision": "REJECTED. Frequency residual branch drastically degraded Kaggle genuine real recall (0.484 << 0.700 target)."
            },
            {
                "experiment_id": "V8-Exp2",
                "name": "Dual-Stream Cost-Sensitive Class Weighting (w_fake=1.15, w_real=1.00)",
                "hypothesis": "Moderate penalty on missed fakes (w_fake=1.15) shifts borderline StyleGAN fakes (P in [0.50, 0.60]) below decision threshold.",
                "loss_configuration": "Binary Cross-Entropy (w_fake=1.15, w_real=1.00)",
                "script": "training/train_mobilenet_v8_weighted.py",
                "seed": 42,
                "batch_size": 32,
                "data_split": "dataset_manifest_v8_dev.json (4,000 train / 1,000 val)",
                "checkpoint": "models/deepfake_image_v8/deepfake_mobilenetv8_exp2_weighted_best.keras",
                "key_metrics_val": {
                    "accuracy": 0.817,
                    "roc_auc": 0.9044,
                    "deepguard_fake_recall": 0.888,
                    "deepguard_real_recall": 0.936,
                    "kaggle_fake_recall": 0.856,
                    "kaggle_real_recall": 0.588
                },
                "decision": (
                    "FINALIZED & EVALUATED. Kaggle fake recall successfully passed target (0.856 >= 0.800), "
                    "but Kaggle real recall failed acceptance (0.588 < 0.700, 103 errors). "
                    "Threshold sweep confirmed no static threshold satisfies all four criteria. "
                    "Acceptance NOT PASSED. Sealed test preserved untouched. Retraining stopped pending Exp3 authorization."
                )
            }
        ],
        "planned_next_experiment": {
            "experiment_id": "V8-Exp3",
            "name": "Auxiliary Multi-Task Manipulation Head",
            "script": "training/train_mobilenet_v8_multitask.py",
            "hypothesis": (
                "Shared 5,120-d backbone with dual heads (Primary Binary + 3-Class Auxiliary: Genuine / Face-Swap / StyleGAN) "
                "prevents facial synthesis gradients from conflicting with facial boundary gradients, eliminating cross-domain degradation."
            ),
            "status": "Prepared in repository; awaiting authorized execution."
        }
    }

    out_ledger = FINALIZATION_DIR / "experiment_ledger.json"
    with open(out_ledger, "w", encoding="utf-8") as f:
        json.dump(experiment_ledger, f, indent=2)
    print(f"Experiment ledger written to: {out_ledger}")

if __name__ == "__main__":
    main()
