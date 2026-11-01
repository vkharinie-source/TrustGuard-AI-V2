"""
TrustGuard AI - Phishing URL Detection Training Pipeline V5
Module: Phishing URL Detection
Datasets: training/datasets/phishing_urls/training_v5.parquet and preserved Testing.parquet

Pipeline:
1. Self-contained URLFeatureExtractor extracting 32 lexical, structural, brand, and entropy features.
2. RobustScaler for continuous numerical signals.
3. TF-IDF character (3-5 char_wb) and word (1-2) n-gram FeatureUnion.
4. Regularized, balanced LogisticRegression classifier.
5. Calibrated threshold selected purely on validation data.
6. Held-out test evaluation on preserved Testing.parquet (3,772 samples).
"""

import sys
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.url_feature_extractor import URLFeatureExtractor

DATA_DIR = BASE_DIR / "datasets" / "phishing_urls"
TRAIN_PARQUET = DATA_DIR / "training_v5.parquet"
TEST_PARQUET = DATA_DIR / "Testing.parquet"
OUT_MODEL_DIR = ROOT_DIR / "models" / "phishing_urls_v5"
REPORT_DIR = BASE_DIR / "reports"
RANDOM_SEED = 42


def get_domain(u: str) -> str:
    try:
        if not u.startswith("http"):
            u = "http://" + u
        host = urlparse(u).hostname or ""
        return host.lower().removeprefix("www.")
    except Exception:
        return ""


def build_pipeline():
    engineered_pipe = Pipeline([
        ("extractor", URLFeatureExtractor()),
        ("scaler", RobustScaler())
    ])

    text_pipe = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=35000,
            sublinear_tf=True
        )),
        ("word", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_features=15000,
            sublinear_tf=True
        ))
    ])

    union = FeatureUnion([
        ("engineered", engineered_pipe),
        ("text", text_pipe)
    ])

    pipeline = Pipeline([
        ("features", union),
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=2.5,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ))
    ])

    return pipeline


def train():
    print("=" * 60)
    print("PHISHING URL DETECTION - TRAINING V5 (INTEGRATED LEXICAL + NLP PIPELINE)")
    print("=" * 60)

    if not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists():
        raise FileNotFoundError(f"Missing parquet files in {DATA_DIR}")

    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    label_map = {"legitimate": 0, "phishing": 1}
    y_train_full = train_df["status"].astype(str).str.lower().map(label_map).astype(int)
    y_test = test_df["status"].astype(str).str.lower().map(label_map).astype(int)

    # Stratified 85% Train / 15% Validation split
    train_urls, val_urls, y_train, y_val = train_test_split(
        train_df["url"], y_train_full, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train_full
    )

    print(f"Dataset Splits: Train = {len(train_urls)} | Validation = {len(val_urls)} | Preserved Test = {len(test_df)}")
    print(f"Train Class Counts: {y_train.value_counts().to_dict()}")
    print(f"Val Class Counts:   {y_val.value_counts().to_dict()}")
    print(f"Test Class Counts:  {y_test.value_counts().to_dict()}")

    pipeline = build_pipeline()

    print("\nFitting integrated feature union + classifier on training set...")
    pipeline.fit(train_urls, y_train)

    # Verify class ordering
    clf = pipeline.named_steps["classifier"]
    classes = [int(c) for c in clf.classes_]
    print(f"Model classes: {classes}")
    assert classes == [0, 1], f"Expected classes [0, 1], got {classes}"
    phish_col = classes.index(1)

    # Validation evaluation & decision threshold sweep
    val_probs = pipeline.predict_proba(val_urls)[:, phish_col]

    print("\n--- Decision Threshold Calibration on Validation Split ---")
    best_thresh = 0.50
    best_val_f1 = 0.0
    threshold_logs = []

    for t in np.arange(0.30, 0.70, 0.02):
        t = round(float(t), 2)
        preds = (val_probs >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        threshold_logs.append({
            "threshold": t, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = t

    print(f"Selected Decision Threshold: {best_thresh} (Validation F1 = {best_val_f1:.4f})")

    val_preds = (val_probs >= best_thresh).astype(int)
    val_metrics = {
        "accuracy": float(accuracy_score(y_val, val_preds)),
        "precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "recall": float(recall_score(y_val, val_preds, zero_division=0)),
        "f1_score": float(f1_score(y_val, val_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, val_probs)),
        "pr_auc": float(average_precision_score(y_val, val_probs)),
        "confusion_matrix": confusion_matrix(y_val, val_preds).tolist()
    }
    print(f"Validation Metrics at Threshold {best_thresh}:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {val_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {val_metrics['pr_auc']:.4f}")
    print(f"  Confusion Matrix: {val_metrics['confusion_matrix']}")

    # Final independent test evaluation on strictly held-out Testing.parquet
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (Preserved Testing.parquet)")
    print("=" * 60)
    test_probs = pipeline.predict_proba(test_df["url"])[:, phish_col]
    test_preds = (test_probs >= best_thresh).astype(int)

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, test_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "pr_auc": float(average_precision_score(y_test, test_probs)),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "phishing_false_negative_rate": float(fn / (fn + tp)),
        "legitimate_false_positive_rate": float(fp / (fp + tn)),
        "classification_report": classification_report(
            y_test, test_preds, target_names=["legitimate", "phishing"], output_dict=True, zero_division=0
        )
    }

    print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall   : {test_metrics['recall']:.4f} (Phishing FN: {fn} / {fn+tp} = {test_metrics['phishing_false_negative_rate']:.2%})")
    print(f"Test F1 Score : {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC   : {test_metrics['pr_auc']:.4f}")
    print(f"Legit False Positive Rate: {test_metrics['legitimate_false_positive_rate']:.2%} ({fp} / {fp+tn})")
    print("Confusion Matrix:\n", cm)

    # Save artifacts into models/phishing_urls_v5
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = OUT_MODEL_DIR / "phishing_urls_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"\nSaved pipeline to: {pipeline_path}")

    features_meta = {
        "feature_type": "FeatureUnion(URLFeatureExtractor(32) + TfidfVectorizer_char_wb(3-5) + TfidfVectorizer_word(1-2))",
        "engineered_feature_count": 32,
        "input_format": "raw_url_string",
        "classifier": "LogisticRegression(C=2.5, class_weight='balanced')",
        "classes": classes,
        "calibrated_decision_threshold": best_thresh
    }
    with open(OUT_MODEL_DIR / "url_features.json", "w", encoding="utf-8") as f:
        json.dump(features_meta, f, indent=2)

    meta = {
        "module": "phishing_urls",
        "version": "v5",
        "model_type": "Unified URL FeatureUnion (32-lexical + char_wb 3-5 + word 1-2) + LogisticRegression",
        "dataset": "training_v5.parquet and preserved Testing.parquet",
        "validation_accuracy": val_metrics["accuracy"],
        "validation_recall": val_metrics["recall"],
        "validation_f1": val_metrics["f1_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1_score"],
        "test_roc_auc": test_metrics["roc_auc"],
        "phishing_false_negatives": test_metrics["false_negatives"],
        "decision_threshold": best_thresh,
        "classes": classes
    }
    with open(OUT_MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save comprehensive machine-readable report
    full_report = {
        "module_name": "Phishing URL Detection V5",
        "version": "v5",
        "dataset_name": "training_v5.parquet and Testing.parquet",
        "split_counts": {
            "training": len(train_urls),
            "validation": len(val_urls),
            "preserved_test": len(test_df)
        },
        "selected_threshold": best_thresh,
        "validation_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "training_status": "SUCCESS"
    }

    # Save confusion matrix image
    cm_img_path = REPORT_DIR / "phishing_urls_v5_cm.png"
    try:
        from train_utils import draw_confusion_matrix
        draw_confusion_matrix(
            test_metrics["confusion_matrix"],
            class_names=["Legitimate URL", "Phishing URL"],
            title="Phishing URL Detection V5 - Test Confusion Matrix",
            out_path=cm_img_path
        )
        print(f"Saved confusion matrix image to: {cm_img_path}")
    except Exception as e:
        print(f"Could not render CM image: {e}")

    report_path = REPORT_DIR / "website_detection_v5_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Saved metrics report to: {report_path}")
    return full_report


if __name__ == "__main__":
    train()

