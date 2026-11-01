"""
TrustGuard AI - Video Deepfake Detection Training V1
Dataset: FaceForensics++ & Celeb-DF Video Dataset (training/datasets/video_deepfake/1000_videos)
Modality: Frame-level Facial CNN Feature Extraction (MobileNetV2) + Temporal Video Aggregation
"""

import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import joblib

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from train_utils import compute_metrics, draw_confusion_matrix

BASE = Path(__file__).resolve().parent
DATASET_DIR = BASE / "datasets" / "video_deepfake" / "1000_videos"
MODEL_DIR = BASE.parent / "models" / "video_deepfake_v1"
REPORT_DIR = BASE / "reports"
RANDOM_SEED = 42

IMG_SIZE = (128, 128)
BATCH_SIZE = 64

def load_split_paths(split_name, max_per_class=1200):
    split_dir = DATASET_DIR / split_name
    fake_paths = sorted(list((split_dir / "fake").glob("*.png")))
    real_paths = sorted(list((split_dir / "real").glob("*.png")))

    rng = np.random.default_rng(RANDOM_SEED)
    if len(fake_paths) > max_per_class:
        fake_paths = list(rng.choice(fake_paths, size=max_per_class, replace=False))
    if len(real_paths) > max_per_class:
        real_paths = list(rng.choice(real_paths, size=max_per_class, replace=False))

    paths = fake_paths + real_paths
    labels = [1] * len(fake_paths) + [0] * len(real_paths)
    
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    paths = [paths[i] for i in indices]
    labels = np.array([labels[i] for i in indices], dtype=int)
    
    return paths, labels

def extract_features(paths, base_model):
    """Extract 1280-dimensional MobileNetV2 embeddings in batches."""
    features_list = []
    total = len(paths)
    for i in range(0, total, BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_imgs = []
        for p in batch_paths:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB").resize(IMG_SIZE)
                batch_imgs.append(np.array(img_rgb, dtype=np.float32))
        
        batch_arr = preprocess_input(np.array(batch_imgs))
        batch_feats = base_model.predict(batch_arr, verbose=0)
        features_list.append(batch_feats)
    return np.vstack(features_list)

def train():
    print("=" * 60)
    print("VIDEO DEEPFAKE DETECTION - TRAINING V1")
    print("=" * 60)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    print(f"Loading dataset from: {DATASET_DIR}")
    train_paths, y_train = load_split_paths("train", max_per_class=1500)
    val_paths, y_val = load_split_paths("validation", max_per_class=500)
    test_paths, y_test = load_split_paths("test", max_per_class=500)

    print(f"Train samples: {len(train_paths)} (Fake: {np.sum(y_train==1)}, Real: {np.sum(y_train==0)})")
    print(f"Val samples:   {len(val_paths)} (Fake: {np.sum(y_val==1)}, Real: {np.sum(y_val==0)})")
    print(f"Test samples:  {len(test_paths)} (Fake: {np.sum(y_test==1)}, Real: {np.sum(y_test==0)})")

    print("\nLoading MobileNetV2 feature extractor (ImageNet weights, Global Pooling)...")
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base_model.trainable = False

    print("Extracting deep CNN representations for Training split...")
    t0 = time.time()
    X_train = extract_features(train_paths, base_model)
    print(f"  Train features shape: {X_train.shape} in {time.time()-t0:.1f}s")

    print("Extracting deep CNN representations for Validation split...")
    t0 = time.time()
    X_val = extract_features(val_paths, base_model)
    print(f"  Val features shape:   {X_val.shape} in {time.time()-t0:.1f}s")

    print("Extracting deep CNN representations for Test split...")
    t0 = time.time()
    X_test = extract_features(test_paths, base_model)
    print(f"  Test features shape:  {X_test.shape} in {time.time()-t0:.1f}s")

    # Evaluate Candidate Classifiers
    candidates = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "SGDClassifier_LogLoss": SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1500, random_state=RANDOM_SEED
        )
    }

    best_name = None
    best_model = None
    best_val_f1 = -1.0
    val_candidate_metrics = {}

    print("\nEvaluating model candidates on Validation split:")
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None
        val_preds = clf.predict(X_val)
        m = compute_metrics(y_val, val_preds, val_probs)
        val_candidate_metrics[name] = m
        print(f"  [{name}] Val Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")

        if m["f1_score"] > best_val_f1:
            best_val_f1 = m["f1_score"]
            best_name = name
            best_model = clf

    print(f"\nBest Candidate: {best_name} (Validation F1 = {best_val_f1:.4f})")

    # Tune decision threshold on validation set
    selected_threshold = 0.5
    best_threshold_f1 = best_val_f1
    if hasattr(best_model, "predict_proba"):
        val_probs_best = best_model.predict_proba(X_val)[:, 1]
        for t in np.arange(0.30, 0.75, 0.05):
            preds_t = (val_probs_best >= t).astype(int)
            f1_t = f1_score(y_val, preds_t, pos_label=1, zero_division=0)
            if f1_t > best_threshold_f1:
                best_threshold_f1 = f1_t
                selected_threshold = float(round(t, 2))

    print(f"Selected Decision Threshold: {selected_threshold} (Tuned Val F1: {best_threshold_f1:.4f})")

    # Final evaluation on held-out Test split
    test_probs = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    test_preds = (test_probs >= selected_threshold).astype(int) if test_probs is not None else best_model.predict(X_test)

    test_metrics = compute_metrics(y_test, test_preds, test_probs)
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION")
    print("=" * 50)
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-Score : {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] else "ROC-AUC  : N/A")
    print(f"PR-AUC   : {test_metrics['pr_auc']:.4f}" if test_metrics['pr_auc'] else "PR-AUC   : N/A")
    print(f"False Positives: {test_metrics['false_positives']} | False Negatives: {test_metrics['false_negatives']}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    clf_path = MODEL_DIR / "video_classifier.pkl"
    meta_path = MODEL_DIR / "metadata.json"
    joblib.dump(best_model, clf_path)

    # Build and save end-to-end Keras model (MobileNetV2 + Dense classifier)
    print("\nAssembling full end-to-end Keras video model...")
    inputs = layers.Input(shape=(128, 128, 3), name="video_frame_input")
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    
    # Linear projection matching best model weights
    coef = best_model.coef_[0].astype(np.float32)
    intercept = best_model.intercept_[0].astype(np.float32)
    
    dense = layers.Dense(1, activation="sigmoid", name="deepfake_probability")
    outputs = dense(x)
    
    end_to_end_model = models.Model(inputs=inputs, outputs=outputs, name="video_deepfake_detector")
    dense.set_weights([coef.reshape(-1, 1), np.array([intercept], dtype=np.float32)])
    
    keras_path = MODEL_DIR / "video_deepfake_model.keras"
    end_to_end_model.save(keras_path)
    print(f"End-to-end Keras model saved: {keras_path}")

    # Metadata
    metadata = {
        "module": "video_deepfake",
        "version": "v1",
        "dataset": "FaceForensics++ & Celeb-DF Combined (nanduncs/1000-videos-split)",
        "backbone": "MobileNetV2(ImageNet, pooling=avg, 128x128)",
        "classifier": best_name,
        "selected_threshold": selected_threshold,
        "label_mapping": {"0": "real_authentic", "1": "fake_deepfake"},
        "random_seed": RANDOM_SEED
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Confusion matrix image
    cm_img_path = REPORT_DIR / "video_deepfake_v1_cm.png"
    draw_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=["Authentic Video", "Deepfake Video"],
        title="Video Deepfake Detection V1 - Test Confusion Matrix",
        out_path=cm_img_path
    )

    # JSON Report
    report = {
        "module_name": "Video Deepfake Detection",
        "version": "training_v1",
        "dataset_name": "FaceForensics++ & Celeb-DF Combined Deepfake Dataset",
        "dataset_path": str(DATASET_DIR),
        "modality": "Frame-Level Facial CNN Feature Extraction & Temporal Video Aggregation",
        "original_sample_count": 16433,
        "usable_sample_count": len(train_paths) + len(val_paths) + len(test_paths),
        "split_counts": {
            "training": len(train_paths),
            "validation": len(val_paths),
            "test": len(test_paths)
        },
        "label_mapping": {"0": "real_authentic", "1": "fake_deepfake"},
        "model_configuration": {
            "feature_extractor": "MobileNetV2(input_shape=(128,128,3), weights='imagenet', pooling='avg')",
            "classifier": best_name,
            "decision_threshold": selected_threshold
        },
        "validation_candidate_metrics": val_candidate_metrics,
        "validation_best_metrics": val_candidate_metrics[best_name],
        "final_test_metrics": test_metrics,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "false_positive_count": test_metrics["false_positives"],
        "false_negative_count": test_metrics["false_negatives"],
        "model_output_path": str(MODEL_DIR),
        "keras_model_file": str(keras_path),
        "classifier_file": str(clf_path),
        "confusion_matrix_image_path": str(cm_img_path),
        "training_status": "SUCCESS",
        "video_aggregation_mechanism": (
            "During inference on a full video (.mp4/.mov), OpenCV samples frames across time intervals. "
            "Individual facial frame scores are aggregated via temporal percentile and peak manipulation scoring."
        )
    }

    report_path_v1 = REPORT_DIR / "video_deepfake_v1_report.json"
    report_path_main = REPORT_DIR / "video_report.json"
    with open(report_path_v1, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(report_path_main, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts saved successfully:")
    print(f"  Keras Model: {keras_path}")
    print(f"  Classifier:  {clf_path}")
    print(f"  Report:      {report_path_v1}")
    print(f"  CM Image:    {cm_img_path}")
    return report

if __name__ == "__main__":
    train()
