"""
Phishing URL Detection - Training V3
Module: Phishing URL Detection
Datasets: training/datasets/phishing_urls/Training.parquet and Testing.parquet
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score

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
    print("PHISHING URL DETECTION - TRAINING V3")
    print("=" * 60)

    if not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists():
        raise FileNotFoundError(f"Missing parquet files in {DATA_DIR}")

    print("Loading Parquet datasets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    label_col = "status"
    label_map = {"legitimate": 0, "phishing": 1}

    for name, df in [("Training", train_df), ("Testing", test_df)]:
        if label_col not in df.columns:
            raise ValueError(f"'{label_col}' missing in {name}.parquet")
        unexpected = set(df[label_col].astype(str).str.lower().unique()) - set(label_map.keys())
        if unexpected:
            raise ValueError(f"Unexpected labels in {name}: {unexpected}")

    y_train_full = train_df[label_col].astype(str).str.lower().map(label_map).astype(int)
    y_test = test_df[label_col].astype(str).str.lower().map(label_map).astype(int)

    # Identify numeric features
    feature_cols = [
        col for col in train_df.columns
        if col not in ["url", label_col] and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    missing_in_test = set(feature_cols) - set(test_df.columns)
    if missing_in_test:
        raise ValueError(f"Testing set missing features: {sorted(missing_in_test)}")

    print(f"Total features identified: {len(feature_cols)}")
    print(f"Training.parquet rows: {len(train_df)} | Testing.parquet rows: {len(test_df)}")

    X_train_full = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Split Training.parquet into 85% train and 15% validation. Testing.parquet is strictly held-out.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train_full
    )

    print(f"Splits -> Train: {len(X_train)} | Validation: {len(X_val)} | Preserved Test: {len(X_test)}")
    print(f"Train label distribution: {y_train.value_counts().to_dict()}")
    print(f"Val label distribution:   {y_val.value_counts().to_dict()}")
    print(f"Test label distribution:  {y_test.value_counts().to_dict()}")

    # Imputer fit strictly on train split
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # Candidates: Random Forest, Extra Trees, HistGradientBoosting
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, random_state=RANDOM_SEED
        )
    }

    best_name = None
    best_model = None
    best_val_f1 = -1.0
    val_candidate_metrics = {}

    print("\nEvaluating model candidates on Validation split:")
    for name, clf in candidates.items():
        clf.fit(X_train_imp, y_train)
        if hasattr(clf, "predict_proba"):
            val_probs = clf.predict_proba(X_val_imp)[:, 1]
        else:
            val_probs = None
        val_preds = clf.predict(X_val_imp)
        m = compute_metrics(y_val, val_preds, val_probs)
        val_candidate_metrics[name] = m
        print(f"  [{name}] Val Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']}")

        if m["f1_score"] > best_val_f1:
            best_val_f1 = m["f1_score"]
            best_name = name
            best_model = clf

    print(f"\nBest Candidate: {best_name} (Validation F1 = {best_val_f1:.4f})")

    # Threshold selection on validation set
    selected_threshold = 0.5
    best_threshold_f1 = best_val_f1
    if hasattr(best_model, "predict_proba"):
        val_probs_best = best_model.predict_proba(X_val_imp)[:, 1]
        for t in np.arange(0.30, 0.75, 0.05):
            preds_t = (val_probs_best >= t).astype(int)
            f1_t = f1_score(y_val, preds_t, pos_label=1, zero_division=0)
            if f1_t > best_threshold_f1:
                best_threshold_f1 = f1_t
                selected_threshold = float(round(t, 2))

    print(f"Selected Decision Threshold: {selected_threshold} (Tuned Val F1: {best_threshold_f1:.4f})")

    # Final evaluation on held-out Testing.parquet
    if hasattr(best_model, "predict_proba"):
        test_probs = best_model.predict_proba(X_test_imp)[:, 1]
        test_preds = (test_probs >= selected_threshold).astype(int)
    else:
        test_probs = None
        test_preds = best_model.predict(X_test_imp)

    test_metrics = compute_metrics(y_test, test_preds, test_probs)
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION (Testing.parquet)")
    print("=" * 50)
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-Score : {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] else "ROC-AUC  : N/A")
    print(f"PR-AUC   : {test_metrics['pr_auc']:.4f}" if test_metrics['pr_auc'] else "PR-AUC   : N/A")
    print(f"False Positives: {test_metrics['false_positives']} | False Negatives: {test_metrics['false_negatives']}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Save pipeline and artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full_pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", best_model)
    ])

    pipeline_path = MODEL_DIR / "phishing_urls_pipeline.pkl"
    model_path = MODEL_DIR / "phishing_urls_model.pkl"
    feat_path = MODEL_DIR / "url_features.json"
    meta_path = MODEL_DIR / "metadata.json"

    joblib.dump(full_pipeline, pipeline_path)
    joblib.dump(best_model, model_path)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "module": "phishing_urls",
        "version": "v3",
        "best_algorithm": best_name,
        "selected_threshold": selected_threshold,
        "label_mapping": label_map,
        "feature_count": len(feature_cols),
        "random_seed": RANDOM_SEED
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Confusion matrix image
    cm_img_path = REPORT_DIR / "phishing_urls_training_v3_cm.png"
    draw_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=["Legitimate URL", "Phishing URL"],
        title="Phishing URL Detection V3 - Test Confusion Matrix",
        out_path=cm_img_path
    )

    # JSON Report
    report = {
        "module_name": "Phishing URL Detection",
        "version": "training_v3",
        "dataset_name": "Training.parquet and Testing.parquet",
        "dataset_path": str(DATA_DIR),
        "original_sample_count": len(train_df) + len(test_df),
        "usable_sample_count": len(train_df) + len(test_df),
        "missing_removed_count": 0,
        "duplicate_removed_count": 0,
        "class_distribution": {
            "training_set": train_df[label_col].value_counts().to_dict(),
            "testing_set": test_df[label_col].value_counts().to_dict()
        },
        "label_mapping": label_map,
        "split_counts": {
            "training": len(X_train),
            "validation": len(X_val),
            "test": len(X_test)
        },
        "splitting_strategy": "Training.parquet split into 85% train / 15% validation; Testing.parquet preserved as held-out test set",
        "model_and_preprocessing_configuration": {
            "selected_model": best_name,
            "decision_threshold": selected_threshold,
            "imputer": "SimpleImputer(strategy='median')",
            "numeric_feature_count": len(feature_cols),
            "candidate_models_evaluated": list(candidates.keys())
        },
        "validation_candidate_metrics": val_candidate_metrics,
        "validation_best_metrics": val_candidate_metrics[best_name],
        "final_test_metrics": test_metrics,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "false_positive_count": test_metrics["false_positives"],
        "false_negative_count": test_metrics["false_negatives"],
        "model_output_path": str(MODEL_DIR),
        "confusion_matrix_image_path": str(cm_img_path),
        "training_status": "SUCCESS",
        "limitations_and_observations": (
            "Model operates on precomputed numerical URL and DNS/lexical features. "
            "Real-time evaluation requires feature extractor to extract identical 87 features from input URLs."
        )
    }

    report_path = REPORT_DIR / "phishing_urls_training_v3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts saved successfully:")
    print(f"  Pipeline: {pipeline_path}")
    print(f"  Report:   {report_path}")
    print(f"  CM Image: {cm_img_path}")
    return report

if __name__ == "__main__":
    train()
