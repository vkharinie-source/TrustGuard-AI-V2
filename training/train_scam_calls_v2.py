"""
Scam Call Detection - Training V2
Module: Scam Call Transcript Classification
Datasets: training/datasets/scam_calls/prepared/train.csv, validation.csv, test.csv
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from train_utils import compute_metrics, draw_confusion_matrix

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "datasets" / "scam_calls" / "prepared"
MODEL_DIR = BASE.parent / "models" / "scam_calls_training_v2"
REPORT_DIR = BASE / "reports"
RANDOM_SEED = 42

def load_and_clean_split(filepath):
    df = pd.read_csv(filepath)
    orig_len = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    valid_mask = df["label"].isin([0, 1])
    df = df[valid_mask].copy()
    return df, orig_len

def train():
    print("=" * 60)
    print("SCAM CALL DETECTION - TRAINING V2")
    print("=" * 60)

    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"
    test_path = DATA_DIR / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Prepared split not found: {p}")

    print("Loading prepared splits...")
    train_df, orig_train_len = load_and_clean_split(train_path)
    val_df, orig_val_len = load_and_clean_split(val_path)
    test_df, orig_test_len = load_and_clean_split(test_path)

    total_original = orig_train_len + orig_val_len + orig_test_len
    total_usable = len(train_df) + len(val_df) + len(test_df)

    print(f"Original samples: {total_original} | Usable: {total_usable}")
    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Val labels:   {val_df['label'].value_counts().to_dict()}")
    print(f"Test labels:  {test_df['label'].value_counts().to_dict()}")

    # Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True
    )

    print("\nFitting vectorizer on training split...")
    X_train_vec = vectorizer.fit_transform(train_df["text"])
    X_val_vec = vectorizer.transform(val_df["text"])
    X_test_vec = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_val = val_df["label"]
    y_test = test_df["label"]

    # Candidates: Logistic Regression, MultinomialNB, SGDClassifier
    candidates = {
        "LogisticRegression_Balanced": LogisticRegression(
            C=1.0, max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "SGDClassifier_Balanced": SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-5, class_weight="balanced",
            max_iter=1500, random_state=RANDOM_SEED
        )
    }

    best_name = None
    best_model = None
    best_val_f1 = -1.0
    val_candidate_metrics = {}

    print("\nEvaluating model candidates on Validation split:")
    for name, clf in candidates.items():
        clf.fit(X_train_vec, y_train)
        if hasattr(clf, "predict_proba"):
            val_probs = clf.predict_proba(X_val_vec)[:, 1]
        else:
            val_probs = None
        val_preds = clf.predict(X_val_vec)
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
        val_probs_best = best_model.predict_proba(X_val_vec)[:, 1]
        for t in np.arange(0.30, 0.75, 0.05):
            preds_t = (val_probs_best >= t).astype(int)
            f1_t = f1_score(y_val, preds_t, pos_label=1, zero_division=0)
            if f1_t > best_threshold_f1:
                best_threshold_f1 = f1_t
                selected_threshold = float(round(t, 2))

    print(f"Selected Decision Threshold: {selected_threshold} (Tuned Val F1: {best_threshold_f1:.4f})")

    # Final evaluation on untouched Test split
    if hasattr(best_model, "predict_proba"):
        test_probs = best_model.predict_proba(X_test_vec)[:, 1]
        test_preds = (test_probs >= selected_threshold).astype(int)
    else:
        test_probs = None
        test_preds = best_model.predict(X_test_vec)

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

    # Save pipeline and artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_model)
    ])

    pipeline_path = MODEL_DIR / "scam_calls_pipeline.pkl"
    model_path = MODEL_DIR / "scam_calls_model.pkl"
    vec_path = MODEL_DIR / "scam_calls_vectorizer.pkl"
    meta_path = MODEL_DIR / "metadata.json"

    joblib.dump(full_pipeline, pipeline_path)
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)

    metadata = {
        "module": "scam_calls",
        "modality": "text_transcript",
        "version": "v2",
        "best_algorithm": best_name,
        "selected_threshold": selected_threshold,
        "label_mapping": {"0": "normal", "1": "scam"},
        "vectorizer": "TF-IDF (max_features=30000, ngram=(1,2), sublinear_tf=True)",
        "random_seed": RANDOM_SEED
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Confusion matrix image
    cm_img_path = REPORT_DIR / "scam_calls_training_v2_cm.png"
    draw_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=["Normal Call", "Scam Call"],
        title="Scam Call Detection V2 - Test Confusion Matrix",
        out_path=cm_img_path
    )

    # JSON Report
    report = {
        "module_name": "Scam Call Transcript Detection",
        "modality": "Text / Transcript classification (not raw audio)",
        "version": "training_v2",
        "dataset_name": "prepared/train.csv, validation.csv, test.csv",
        "dataset_path": str(DATA_DIR),
        "original_sample_count": total_original,
        "usable_sample_count": total_usable,
        "missing_removed_count": total_original - total_usable,
        "duplicate_removed_count": 0,
        "class_distribution": {
            "training": train_df["label"].value_counts().to_dict(),
            "validation": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict()
        },
        "label_mapping": {"0": "normal", "1": "scam"},
        "split_counts": {
            "training": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "splitting_strategy": "Official prepared split (train.csv, validation.csv, test.csv) preserved strictly",
        "model_and_preprocessing_configuration": {
            "selected_model": best_name,
            "decision_threshold": selected_threshold,
            "vectorizer": "TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True)",
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
            "Model is trained strictly on transcripts/text dialogues and does not analyze acoustic signals or voice timbre. "
            "Real-world audio calls require an automatic speech recognition (ASR) pipeline before evaluation."
        )
    }

    report_path = REPORT_DIR / "scam_calls_training_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts saved successfully:")
    print(f"  Pipeline: {pipeline_path}")
    print(f"  Report:   {report_path}")
    print(f"  CM Image: {cm_img_path}")
    return report

if __name__ == "__main__":
    train()
