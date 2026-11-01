"""
TrustGuard AI - Final Independent Evaluation of Selected V6 Phishing Detector
Evaluates models/phishing_urls_v6_exp/phishing_urls_pipeline_v6.pkl ONCE on the untouched
independent test set: training/datasets/phishing_urls/v6_test_new.parquet

SAFEGUARDS:
- Uses untouched v6_test_new.parquet (1,305 samples).
- Does not modify or tune any threshold (uses calibrated threshold 0.46 selected on validation).
- Never accesses or modifies Testing.parquet.
- Preserves V5 production model byte-for-byte.
- Evaluates domain-grouped performance and targeted URL category performance.
"""

import sys
import json
import math
from pathlib import Path
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

ROOT_DIR = Path(r"D:\project\DeepFake-Detector-AI")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.url_feature_extractor_v6 import URLFeatureExtractorV6

DATA_DIR = ROOT_DIR / "training" / "datasets" / "phishing_urls"
TEST_NEW_PARQUET = DATA_DIR / "v6_test_new.parquet"
MODEL_PATH = ROOT_DIR / "models" / "phishing_urls_v6_exp" / "phishing_urls_pipeline_v6.pkl"
REPORT_DIR = ROOT_DIR / "training" / "reports"


def get_domain(u: str) -> str:
    try:
        if not str(u).startswith("http"):
            u = "http://" + str(u)
        host = urlparse(u).hostname or ""
        return host.lower().removeprefix("www.")
    except Exception:
        return ""


def categorize_url(u: str) -> str:
    u_lower = u.lower()
    parsed = urlparse(u_lower if u_lower.startswith("http") else "http://" + u_lower)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path or ""

    shared_hosts = ["tumblr.com", "blogspot.com", "narod.ru", "es.tl", "totalh.net", "wordpress.com", "vercel.app", "netlify.app", "web.app", "pages.dev"]
    if any(host.endswith(sh) for sh in shared_hosts):
        return "Shared Hosting / Blogging"
    if any(b in host for b in ["paypal", "netflix", "apple", "amazon", "chase", "microsoft", "skechers", "bt-"]):
        return "Brand Reference / Impersonation"
    if len(u_lower) < 32 or any(s in host for s in ["bit.ly", "tinyurl", "cort.as", "is.gd", "t.co"]):
        return "Short URL / Short Path"
    if any(d in path for d in ["project", "docx", "cs", "boc", "files", "data", "upload"]):
        return "Deep / Compromised Path"
    return "Standard Domain & Path"


def run_evaluation():
    print("=" * 70)
    print("FINAL INDEPENDENT EVALUATION OF EXPERIMENTAL V6 MODEL")
    print("=" * 70)

    if not TEST_NEW_PARQUET.exists():
        raise FileNotFoundError(f"Missing {TEST_NEW_PARQUET}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}")

    df_test = pd.read_parquet(TEST_NEW_PARQUET)
    print(f"Loaded untouched test split: {len(df_test)} samples")
    print(f"Class distribution: {df_test['status'].value_counts().to_dict()}")

    label_map = {"legitimate": 0, "phishing": 1}
    y_true = df_test["status"].astype(str).str.lower().map(label_map).astype(int)
    urls = df_test["url"]

    # Load frozen V6 model
    pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded pipeline: {MODEL_PATH}")

    # Check classes
    clf = pipeline.named_steps["classifier"]
    classes = [int(c) for c in clf.classes_]
    assert classes == [0, 1], f"Expected classes [0, 1], got {classes}"

    # Extract probabilities
    probs = pipeline.predict_proba(urls)[:, 1]

    # Evaluate at calibrated operating threshold (0.46)
    threshold = 0.46
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    print("\n" + "-" * 50)
    print(f"RESULTS ON UNTOUCHED V6 TEST BENCHMARK (Threshold {threshold}):")
    print("-" * 50)
    print(f"Total Test Samples:   {len(urls):,}")
    print(f"Actual Phishing:      {y_true.sum():,}")
    print(f"Actual Legitimate:    {(y_true == 0).sum():,}")
    print(f"Accuracy:             {acc * 100:.2f}%")
    print(f"Precision:            {prec * 100:.2f}%")
    print(f"Recall:               {rec * 100:.2f}%")
    print(f"F1 Score:             {f1 * 100:.2f}%")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"PR-AUC:               {pr_auc:.4f}")
    print(f"Phishing FNR:         {fnr * 100:.2f}% ({fn} missed out of {fn+tp})")
    print(f"Legitimate FPR:       {fpr * 100:.2f}% ({fp} false alarms out of {fp+tn})")
    print(f"Confusion Matrix:     TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Category breakdown
    df_test["y_true"] = y_true
    df_test["y_pred"] = preds
    df_test["prob"] = probs
    df_test["domain"] = df_test["url"].apply(get_domain)
    df_test["category"] = df_test["url"].apply(categorize_url)

    category_stats = []
    print("\n" + "-" * 50)
    print("PERFORMANCE BY TARGETED URL CATEGORY:")
    print("-" * 50)
    print(f"{'Category':<32} | {'Samples':<7} | {'Phishing':<8} | {'Recall':<8} | {'FNR':<7} | {'FPR':<7}")
    print("-" * 75)
    for cat, g in df_test.groupby("category"):
        cat_y_true = g["y_true"]
        cat_y_pred = g["y_pred"]
        phish_cnt = int(cat_y_true.sum())
        legit_cnt = int((cat_y_true == 0).sum())

        cat_tp = int(((cat_y_true == 1) & (cat_y_pred == 1)).sum())
        cat_fn = int(((cat_y_true == 1) & (cat_y_pred == 0)).sum())
        cat_tn = int(((cat_y_true == 0) & (cat_y_pred == 0)).sum())
        cat_fp = int(((cat_y_true == 0) & (cat_y_pred == 1)).sum())

        cat_rec = cat_tp / max(1, phish_cnt) if phish_cnt > 0 else 1.0
        cat_fnr = cat_fn / max(1, phish_cnt) if phish_cnt > 0 else 0.0
        cat_fpr = cat_fp / max(1, legit_cnt) if legit_cnt > 0 else 0.0

        print(f"{cat:<32} | {len(g):<7} | {phish_cnt:<8} | {cat_rec*100:<7.1f}% | {cat_fnr*100:<6.1f}% | {cat_fpr*100:<6.1f}%")
        category_stats.append({
            "category": cat,
            "total_samples": len(g),
            "phishing_samples": phish_cnt,
            "legitimate_samples": legit_cnt,
            "tp": cat_tp,
            "fn": cat_fn,
            "tn": cat_tn,
            "fp": cat_fp,
            "recall": round(cat_rec, 4),
            "phishing_fnr": round(cat_fnr, 4),
            "legitimate_fpr": round(cat_fpr, 4)
        })

    # Domain-grouped consistency
    top_domains = df_test["domain"].value_counts().head(10).index.tolist()
    domain_grouped_perf = []
    for d in top_domains:
        sub = df_test[df_test["domain"] == d]
        sub_y_true = sub["y_true"]
        sub_y_pred = sub["y_pred"]
        sub_acc = accuracy_score(sub_y_true, sub_y_pred)
        domain_grouped_perf.append({
            "domain": d,
            "samples": len(sub),
            "accuracy": round(sub_acc, 4)
        })

    # Save detailed evaluation JSON
    eval_record = {
        "evaluation_dataset": "v6_test_new.parquet",
        "model_artifact": str(MODEL_PATH),
        "threshold": threshold,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "phishing_fnr": round(fnr, 4),
            "legitimate_fpr": round(fpr, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        },
        "category_breakdown": category_stats,
        "top_domains_performance": domain_grouped_perf
    }

    out_json = REPORT_DIR / "website_detection_v6_test_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)
    print(f"\nSaved evaluation metrics to: {out_json}")

    return eval_record


if __name__ == "__main__":
    run_evaluation()
