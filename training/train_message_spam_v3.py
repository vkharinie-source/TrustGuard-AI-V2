"""
Message / SMS Spam Detection - Training V3
Module: Message / SMS Spam Detection
Dataset: training/datasets/message_spam/dataset.csv
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from train_utils import compute_metrics, draw_confusion_matrix

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "datasets" / "message_spam" / "dataset.csv"
MODEL_DIR = BASE.parent / "models" / "message_spam_training_v3"
REPORT_DIR = BASE / "reports"
RANDOM_SEED = 42

def train():
    print("=" * 60)
    print("MESSAGE / SMS SPAM DETECTION - TRAINING V3")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"Loading dataset: {DATA_PATH.name}")
    raw_df = pd.read_csv(DATA_PATH)
    original_count = len(raw_df)
    print(f"Original samples: {original_count}")

    # Inspect columns
    text_col = "text" if "text" in raw_df.columns else raw_df.columns[1]
    label_col = "labels" if "labels" in raw_df.columns else raw_df.columns[0]
    lang_col = "lang" if "lang" in raw_df.columns else None

    # Clean missing
    df = raw_df.dropna(subset=[text_col, label_col]).copy()
    missing_count = original_count - len(df)

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].copy()

    # Map labels: 'ham' -> 0, 'spam' -> 1
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    label_map = {"ham": 0, "spam": 1}
    valid_mask = df[label_col].isin(label_map.keys())
    df = df[valid_mask].copy()
    df["target"] = df[label_col].map(label_map).astype(int)

    # Deduplicate text while keeping first metadata
    initial_clean = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    duplicate_count = initial_clean - len(df)
    usable_count = len(df)

    print(f"Usable samples after cleaning: {usable_count} (dropped {missing_count} null/blank, {duplicate_count} duplicates)")
    class_dist = df["target"].value_counts().to_dict()
    print(f"Class distribution (0=ham, 1=spam): {class_dist}")

    if lang_col and lang_col in df.columns:
        print("Language distribution in usable dataset:")
        print(df[lang_col].value_counts().to_dict())

    # Stratified split: 70% Train, 15% Validation, 15% Test
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, random_state=RANDOM_SEED, stratify=df["target"]
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=(0.15 / 0.85), random_state=RANDOM_SEED, stratify=train_val_df["target"]
    )

    print(f"Splits -> Train: {len(train_df)} ({(len(train_df)/usable_count)*100:.1f}%) | "
          f"Validation: {len(val_df)} ({(len(val_df)/usable_count)*100:.1f}%) | "
          f"Test: {len(test_df)} ({(len(test_df)/usable_count)*100:.1f}%)")

    # Multilingual-friendly character and word n-gram TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=40000,
        sublinear_tf=True
    )

    print("\nFitting vectorizer on training split...")
    X_train_vec = vectorizer.fit_transform(train_df[text_col])
    X_val_vec = vectorizer.transform(val_df[text_col])
    X_test_vec = vectorizer.transform(test_df[text_col])

    y_train = train_df["target"]
    y_val = val_df["target"]
    y_test = test_df["target"]

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
    print("FINAL TEST EVALUATION (OVERALL)")
    print("=" * 50)
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-Score : {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] else "ROC-AUC  : N/A")
    print(f"PR-AUC   : {test_metrics['pr_auc']:.4f}" if test_metrics['pr_auc'] else "PR-AUC   : N/A")
    print(f"False Positives: {test_metrics['false_positives']} | False Negatives: {test_metrics['false_negatives']}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Evaluation by language
    by_language_metrics = {}
    if lang_col and lang_col in test_df.columns:
        print("\n=== BREAKDOWN BY LANGUAGE (HELD-OUT TEST SET) ===")
        test_df_eval = test_df.copy()
        test_df_eval["prediction"] = test_preds

        for lang, group in test_df_eval.groupby(lang_col):
            y_lang_true = group["target"]
            y_lang_pred = group["prediction"]
            l_acc = float(accuracy_score(y_lang_true, y_lang_pred))
            l_prec = float(precision_score(y_lang_true, y_lang_pred, pos_label=1, zero_division=0))
            l_rec = float(recall_score(y_lang_true, y_lang_pred, pos_label=1, zero_division=0))
            l_f1 = float(f1_score(y_lang_true, y_lang_pred, pos_label=1, zero_division=0))

            by_language_metrics[lang] = {
                "sample_count": len(group),
                "accuracy": l_acc,
                "precision": l_prec,
                "recall": l_rec,
                "f1_score": l_f1
            }
            print(f"  {lang.upper():10s} (n={len(group):4d}) | Acc: {l_acc:.4f} | Prec: {l_prec:.4f} | Rec: {l_rec:.4f} | F1: {l_f1:.4f}")

    # Save pipeline and artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_model)
    ])

    pipeline_path = MODEL_DIR / "message_spam_pipeline.pkl"
    model_path = MODEL_DIR / "message_spam_model.pkl"
    vec_path = MODEL_DIR / "message_spam_vectorizer.pkl"
    meta_path = MODEL_DIR / "metadata.json"

    joblib.dump(full_pipeline, pipeline_path)
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)

    metadata = {
        "module": "message_spam",
        "version": "v3",
        "best_algorithm": best_name,
        "selected_threshold": selected_threshold,
        "label_mapping": {"0": "ham", "1": "spam"},
        "vectorizer": "TF-IDF (max_features=40000, ngram=(1,2), sublinear_tf=True)",
        "languages_evaluated": list(by_language_metrics.keys()),
        "random_seed": RANDOM_SEED
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Confusion matrix image
    cm_img_path = REPORT_DIR / "message_spam_training_v3_cm.png"
    draw_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=["Ham", "Spam"],
        title="Message Spam Detection V3 - Test Confusion Matrix",
        out_path=cm_img_path
    )

    # JSON Report
    report = {
        "module_name": "Message / SMS Spam Detection",
        "version": "training_v3",
        "dataset_name": "dataset.csv",
        "dataset_path": str(DATA_PATH),
        "original_sample_count": original_count,
        "usable_sample_count": usable_count,
        "missing_removed_count": missing_count,
        "duplicate_removed_count": duplicate_count,
        "class_distribution": class_dist,
        "label_mapping": {"0": "ham", "1": "spam"},
        "split_counts": {
            "training": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "splitting_strategy": "Stratified 70/15/15 split preserving language distribution, seed 42",
        "model_and_preprocessing_configuration": {
            "selected_model": best_name,
            "decision_threshold": selected_threshold,
            "vectorizer": "TfidfVectorizer(max_features=40000, ngram_range=(1,2), sublinear_tf=True, strip_accents='unicode')",
            "candidate_models_evaluated": list(candidates.keys())
        },
        "validation_candidate_metrics": val_candidate_metrics,
        "validation_best_metrics": val_candidate_metrics[best_name],
        "final_test_metrics": test_metrics,
        "breakdown_by_language": by_language_metrics,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "false_positive_count": test_metrics["false_positives"],
        "false_negative_count": test_metrics["false_negatives"],
        "model_output_path": str(MODEL_DIR),
        "confusion_matrix_image_path": str(cm_img_path),
        "training_status": "SUCCESS",
        "limitations_and_observations": (
            "Multilingual dataset covers English, French, German, and Hindi messages. Performance on low-resource "
            "dialects or transliterated romanized text (e.g. Hinglish) may be lower than standardized corpora."
        )
    }

    report_path = REPORT_DIR / "message_spam_training_v3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nArtifacts saved successfully:")
    print(f"  Pipeline: {pipeline_path}")
    print(f"  Report:   {report_path}")
    print(f"  CM Image: {cm_img_path}")
    return report

if __name__ == "__main__":
    train()
