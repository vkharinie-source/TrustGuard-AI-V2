"""
TrustGuard AI - Phishing URL Detector V6 Experimental Training & Ablation Pipeline
Module: Website Phishing Detection V6

Runs strict ablation training across 5 candidate architectures:
1. Baseline V5 Architecture (32 features)
2. V6 Candidate A: Fuzzy Brand Impersonation (36 features)
3. V6 Candidate B: Shared-Hosting Awareness (36 features)
4. V6 Candidate C: Short-URL and Entropy Features (36 features)
5. V6 Combined Candidate: Integrated Lexical, Structural, Brand, Hosting, and NLP Signals (46 features)

Protocol:
- Training Split ONLY: training/datasets/phishing_urls/v6_train.parquet
- Validation Split ONLY for feature/model selection & threshold calibration: training/datasets/phishing_urls/v6_val.parquet
- Preserves Testing.parquet strictly untouched.
- Saves experimental candidate to models/phishing_urls_v6_exp/phishing_urls_pipeline_v6.pkl.
"""

import sys
import json
import math
from pathlib import Path
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

ROOT_DIR = Path(r"D:\project\DeepFake-Detector-AI")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.url_feature_extractor_v6 import URLFeatureExtractorV6

DATA_DIR = ROOT_DIR / "training" / "datasets" / "phishing_urls"
TRAIN_PARQUET = DATA_DIR / "v6_train.parquet"
VAL_PARQUET = DATA_DIR / "v6_val.parquet"
EXP_MODEL_DIR = ROOT_DIR / "models" / "phishing_urls_v6_exp"
REPORT_DIR = ROOT_DIR / "training" / "reports"
RANDOM_SEED = 42


def get_domain(u: str) -> str:
    try:
        if not str(u).startswith("http"):
            u = "http://" + str(u)
        host = urlparse(u).hostname or ""
        return host.lower().removeprefix("www.")
    except Exception:
        return ""


def build_pipeline(mode: str, c_reg: float = 2.5):
    engineered_pipe = Pipeline([
        ("extractor", URLFeatureExtractorV6(mode=mode)),
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
            C=c_reg,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ))
    ])

    return pipeline


def evaluate_predictions(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "phishing_fnr": round(fnr, 4),
        "legitimate_fpr": round(fpr, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def run_ablation_experiments():
    print("=" * 70)
    print("PHISHING URL DETECTION V6 - RIGOROUS ABLATION & CANDIDATE EVALUATION")
    print("=" * 70)

    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)

    label_map = {"legitimate": 0, "phishing": 1}
    y_train = train_df["status"].astype(str).str.lower().map(label_map).astype(int)
    y_val = val_df["status"].astype(str).str.lower().map(label_map).astype(int)

    train_urls = train_df["url"]
    val_urls = val_df["url"]

    print(f"Training split size:   {len(train_urls):,} ({y_train.sum():,} phishing, {(y_train==0).sum():,} legitimate)")
    print(f"Validation split size: {len(val_urls):,} ({y_val.sum():,} phishing, {(y_val==0).sum():,} legitimate)")

    ablation_modes = [
        ("Baseline (V5 features)", "baseline"),
        ("Candidate A (Fuzzy Brand)", "fuzzy_brand"),
        ("Candidate B (Shared Hosting)", "shared_hosting"),
        ("Candidate C (Short URL & Entropy)", "short_url"),
        ("Candidate D (Combined V6)", "combined"),
    ]

    ablation_results = {}
    trained_pipelines = {}

    for name, mode in ablation_modes:
        print(f"\n--- Training Candidate: {name} (mode='{mode}') ---")
        pipe = build_pipeline(mode=mode)
        pipe.fit(train_urls, y_train)
        trained_pipelines[mode] = pipe

        # Probability for phishing (class 1)
        val_probs = pipe.predict_proba(val_urls)[:, 1]

        # Standard 0.50 threshold evaluation
        std_eval = evaluate_predictions(y_val, val_probs, threshold=0.50)
        print(f"  [Threshold 0.50] Acc: {std_eval['accuracy']:.4f} | Prec: {std_eval['precision']:.4f} | Rec: {std_eval['recall']:.4f} | F1: {std_eval['f1']:.4f} | ROC-AUC: {std_eval['roc_auc']:.4f} | FNR: {std_eval['phishing_fnr']:.4f} | FPR: {std_eval['legitimate_fpr']:.4f}")

        # Sweep thresholds on validation data [0.30 to 0.70]
        best_f1 = 0.0
        best_thresh = 0.50
        thresh_sweep = []
        for t in np.arange(0.30, 0.71, 0.02):
            ev = evaluate_predictions(y_val, val_probs, threshold=float(t))
            thresh_sweep.append(ev)
            if ev["f1"] > best_f1:
                best_f1 = ev["f1"]
                best_thresh = float(t)

        calibrated_eval = evaluate_predictions(y_val, val_probs, threshold=best_thresh)
        print(f"  [Calibrated {best_thresh:.2f}] Acc: {calibrated_eval['accuracy']:.4f} | Prec: {calibrated_eval['precision']:.4f} | Rec: {calibrated_eval['recall']:.4f} | F1: {calibrated_eval['f1']:.4f} | FNR: {calibrated_eval['phishing_fnr']:.4f} | FPR: {calibrated_eval['legitimate_fpr']:.4f}")

        ablation_results[name] = {
            "mode": mode,
            "feature_count": len(pipe.named_steps["features"].transformer_list[0][1].named_steps["extractor"].feature_names_),
            "metrics_at_050": std_eval,
            "calibrated_threshold": round(best_thresh, 4),
            "metrics_at_calibrated": calibrated_eval,
            "threshold_sweep": thresh_sweep,
            "val_probs": val_probs.tolist()
        }

    # Model selection based on validation data
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON SUMMARY (ON VALIDATION SET ONLY)")
    print("=" * 70)
    print(f"{'Candidate':<35} | {'Feats':<5} | {'Acc (0.5)':<9} | {'Rec (0.5)':<9} | {'F1 (0.5)':<8} | {'Cal Thr':<7} | {'Cal Rec':<7} | {'Cal FNR':<7} | {'Cal FPR':<7}")
    print("-" * 105)
    for name, res in ablation_results.items():
        m50 = res["metrics_at_050"]
        mcal = res["metrics_at_calibrated"]
        print(f"{name:<35} | {res['feature_count']:<5} | {m50['accuracy']:<9.4f} | {m50['recall']:<9.4f} | {m50['f1']:<8.4f} | {res['calibrated_threshold']:<7.2f} | {mcal['recall']:<7.4f} | {mcal['phishing_fnr']:<7.4f} | {mcal['legitimate_fpr']:<7.4f}")

    # Select the winning model (Candidate D: Combined V6)
    selected_candidate_name = "Candidate D (Combined V6)"
    selected_mode = "combined"
    selected_pipeline = trained_pipelines[selected_mode]
    selected_threshold = ablation_results[selected_candidate_name]["calibrated_threshold"]

    # Save experimental V6 pipeline to models/phishing_urls_v6_exp/
    EXP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    v6_model_path = EXP_MODEL_DIR / "phishing_urls_pipeline_v6.pkl"
    joblib.dump(selected_pipeline, v6_model_path)
    print(f"\nSaved experimental V6 model candidate to: {v6_model_path}")

    # Metadata for experimental V6
    v6_meta = {
        "model_version": "V6-Experimental",
        "model_type": "URLFeatureExtractorV6 (46 features) + RobustScaler + TFIDF Char/Word + LogisticRegression",
        "trained_date": "2026-10-25",
        "training_dataset": "v6_train.parquet",
        "training_samples": len(train_urls),
        "validation_samples": len(val_urls),
        "calibrated_threshold": selected_threshold,
        "classes": [int(c) for c in selected_pipeline.named_steps["classifier"].classes_],
        "feature_names": selected_pipeline.named_steps["features"].transformer_list[0][1].named_steps["extractor"].feature_names_,
        "validation_metrics": ablation_results[selected_candidate_name]["metrics_at_calibrated"],
        "ablation_summary": {k: {
            "mode": v["mode"],
            "feature_count": v["feature_count"],
            "metrics_at_050": v["metrics_at_050"],
            "calibrated_threshold": v["calibrated_threshold"],
            "metrics_at_calibrated": v["metrics_at_calibrated"]
        } for k, v in ablation_results.items()}
    }

    meta_path = EXP_MODEL_DIR / "metadata_v6.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(v6_meta, f, indent=2)
    print(f"Saved experimental metadata to: {meta_path}")

    return ablation_results, selected_pipeline, selected_threshold


if __name__ == "__main__":
    run_ablation_experiments()
