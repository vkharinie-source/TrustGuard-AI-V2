"""
Fake News Detection - Training V2
Module: Fake News Detection
Datasets: training/datasets/fake_news/Fake.csv and True.csv
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from train_utils import compute_metrics, draw_confusion_matrix

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "datasets" / "fake_news"
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"
MODEL_DIR = BASE.parent / "models" / "fake_news_training_v2"
REPORT_DIR = BASE / "reports"
RANDOM_SEED = 42

def train():
    print("=" * 60)
    print("FAKE NEWS DETECTION - TRAINING V2")
    print("=" * 60)

    if not FAKE_PATH.exists() or not TRUE_PATH.exists():
        raise FileNotFoundError(f"Missing datasets in {DATA_DIR}")

    print(f"Loading Fake news: {FAKE_PATH.name}")
    fake_df = pd.read_csv(FAKE_PATH)
    fake_df["label"] = 1  # 1 = Fake

    print(f"Loading True news: {TRUE_PATH.name}")
    true_df = pd.read_csv(TRUE_PATH)
    true_df["label"] = 0  # 0 = Real / True

    original_fake = len(fake_df)
    original_true = len(true_df)
    original_count = original_fake + original_true
    print(f"Original counts -> Fake: {original_fake}, True: {original_true}, Total: {original_count}")

    # Combine title and text for rich context
    combined = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Check nulls
    missing_count = int(combined[["title", "text"]].isnull().any(axis=1).sum())
    combined["title"] = combined["title"].fillna("").astype(str).str.strip()
    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    
    combined["full_text"] = combined["title"] + " " + combined["text"]
    combined["full_text"] = combined["full_text"].str.strip()
    combined = combined[combined["full_text"] != ""].copy()

    # Deduplicate by text content to prevent split leakage
    initial_clean = len(combined)
    combined = combined.drop_duplicates(subset=["full_text"]).reset_index(drop=True)
    duplicate_count = initial_clean - len(combined)
    usable_count = len(combined)

    print(f"Usable samples after cleaning and deduplication: {usable_count} (dropped {duplicate_count} duplicates)")
    class_dist = combined["label"].value_counts().to_dict()
    print(f"Class distribution (0=Real, 1=Fake): {class_dist}")

    # Split 70% Train, 15% Validation, 15% Test
    X = combined["full_text"]
    y = combined["label"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=RANDOM_SEED, stratify=y_train_val
    )

    print(f"Splits -> Train: {len(X_train)} ({(len(X_train)/usable_count)*100:.1f}%) | "
          f"Validation: {len(X_val)} ({(len(X_val)/usable_count)*100:.1f}%) | "
          f"Test: {len(X_test)} ({(len(X_test)/usable_count)*100:.1f}%)")

    # Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    print("\nFitting vectorizer on training data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Model candidates
    candidates = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "SGDClassifier_LogLoss": SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-5, max_iter=1500, random_state=RANDOM_SEED
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

    # Save complete pipeline and standalone artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_model)
    ])

    pipeline_path = MODEL_DIR / "fake_news_pipeline.pkl"
    model_path = MODEL_DIR / "fake_news_model.pkl"
    vec_path = MODEL_DIR / "fake_news_vectorizer.pkl"
    meta_path = MODEL_DIR / "metadata.json"

    joblib.dump(full_pipeline, pipeline_path)
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)

    metadata = {
        "module": "fake_news",
        "version": "v2",
        "best_algorithm": best_name,
        "selected_threshold": selected_threshold,
        "label_mapping": {"0": "real_true", "1": "fake"},
        "vectorizer": "TF-IDF (max_features=40000, ngram=(1,2), sublinear_tf=True)",
        "random_seed": RANDOM_SEED
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Confusion matrix image
    cm_img_path = REPORT_DIR / "fake_news_training_v2_cm.png"
    draw_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=["Real News", "Fake News"],
        title="Fake News Detection V2 - Test Confusion Matrix",
        out_path=cm_img_path
    )

    # JSON Report
    report = {
        "module_name": "Fake News Detection",
        "version": "training_v2",
        "dataset_name": "Fake.csv and True.csv",
        "dataset_path": str(DATA_DIR),
        "original_sample_count": original_count,
        "usable_sample_count": usable_count,
        "missing_removed_count": missing_count,
        "duplicate_removed_count": duplicate_count,
        "class_distribution": class_dist,
        "label_mapping": {"0": "real_true", "1": "fake"},
        "split_counts": {
            "training": len(X_train),
            "validation": len(X_val),
            "test": len(X_test)
        },
        "splitting_strategy": "Stratified 70/15/15 split on combined unique articles, seed 42",
        "model_and_preprocessing_configuration": {
            "selected_model": best_name,
            "decision_threshold": selected_threshold,
            "features": "Concatenated title and text fields",
            "vectorizer": "TfidfVectorizer(max_features=40000, ngram_range=(1,2), sublinear_tf=True, stop_words='english')",
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
            "The dataset exhibits distinct stylistic cues (e.g. publication headers like Reuters for true news). "
            "Real-world generalization across evolving political and international topics may degrade if new events use novel terminology."
        )
    }

    report_path = REPORT_DIR / "fake_news_training_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts saved successfully:")
    print(f"  Pipeline: {pipeline_path}")
    print(f"  Report:   {report_path}")
    print(f"  CM Image: {cm_img_path}")
    return report

if __name__ == "__main__":
    train()
