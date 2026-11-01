"""
Phishing URL Detection - Training V4
Module: Phishing URL Detection
Datasets: training/datasets/phishing_urls/Training.parquet and Testing.parquet

Uses an end-to-end NLP/Lexical FeatureUnion directly on the raw URL string:
1) Character-level n-grams (3 to 6) capturing obfuscation, IP patterns, punycode, subdomains, extensions.
2) Word-level n-grams capturing phishing terms and brand impersonation tokens.
Model: Regularized Logistic Regression with balanced weighting.
Evaluated on 15% validation split and preserved untouched Testing.parquet (3,772 samples).
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from train_utils import compute_metrics, draw_confusion_matrix

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "datasets" / "phishing_urls"
TRAIN_PARQUET = DATA_DIR / "Training.parquet"
TEST_PARQUET = DATA_DIR / "Testing.parquet"
MODEL_DIR = BASE.parent / "models" / "phishing_urls_training_v3"
REPORT_DIR = BASE / "reports"
RANDOM_SEED = 42

def train():
    print("=" * 60)
    print("PHISHING URL DETECTION - TRAINING V4 (End-to-End Direct URL NLP Pipeline)")
    print("=" * 60)

    if not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists():
        raise FileNotFoundError(f"Missing parquet files in {DATA_DIR}")

    print("Loading Parquet datasets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    label_col = "status"
    label_map = {"legitimate": 0, "phishing": 1}

    y_train_full = train_df[label_col].astype(str).str.lower().map(label_map).astype(int)
    y_test = test_df[label_col].astype(str).str.lower().map(label_map).astype(int)

    # Stratified 85% train, 15% validation from Training.parquet. Testing.parquet is strictly held-out.
    train_urls, val_urls, y_train, y_val = train_test_split(
        train_df["url"], y_train_full, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train_full
    )

    print(f"Splits -> Train: {len(train_urls)} | Validation: {len(val_urls)} | Preserved Test: {len(test_df)}")

    union = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=3,
            max_features=35000,
            sublinear_tf=True
        )),
        ("word", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=3,
            max_features=15000,
            sublinear_tf=True
        ))
    ])

    pipeline = Pipeline([
        ("features", union),
        ("classifier", LogisticRegression(
            max_iter=1500,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ))
    ])

    print("\nFitting URL pipeline...")
    pipeline.fit(train_urls, y_train)

    # Validation evaluation
    val_probs = pipeline.predict_proba(val_urls)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)
    val_metrics = compute_metrics(y_val, val_preds, val_probs)
    print("\n--- VALIDATION METRICS ---")
    print(f"Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall:    {val_metrics['recall']:.4f}")
    print(f"F1 Score:  {val_metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {val_metrics['roc_auc']:.4f}")

    # Final test evaluation on held-out Testing.parquet
    test_probs = pipeline.predict_proba(test_df["url"])[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)
    test_metrics = compute_metrics(y_test, test_preds, test_probs)
    print("\n--- FINAL TEST METRICS (Preserved Testing.parquet) ---")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))

    # Save model artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_DIR / "phishing_urls_pipeline.pkl")
    with open(MODEL_DIR / "url_features.json", "w", encoding="utf-8") as f:
        json.dump(["raw_url"], f, indent=2)

    meta = {
        "module": "phishing_urls",
        "version": "v4",
        "model_type": "Direct URL FeatureUnion (char_wb 3-6 + word 1-2) + LogisticRegression",
        "dataset": "Training.parquet and Testing.parquet",
        "validation_accuracy": val_metrics["accuracy"],
        "validation_f1": val_metrics["f1_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_score"],
        "decision_threshold": 0.50
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = {
        "dataset_name": "Training.parquet and Testing.parquet",
        "model_output_path": str(MODEL_DIR),
        "split_counts": {
            "training": len(train_urls),
            "validation": len(val_urls),
            "test": len(test_df)
        },
        "validation_best_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "training_status": "SUCCESS"
    }
    with open(REPORT_DIR / "phishing_urls_training_v3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nModel saved to:", MODEL_DIR / "phishing_urls_pipeline.pkl")
    print("Report saved to:", REPORT_DIR / "phishing_urls_training_v3_report.json")

if __name__ == "__main__":
    train()
