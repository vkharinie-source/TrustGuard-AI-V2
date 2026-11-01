"""
TrustGuard AI - Complete Dataset Preparation & Training Pipeline
Runs all modules: Email, Message, URL, Job Scam, News
Call: NO SUITABLE KAGGLE DATASET FOUND (403 Forbidden on all available datasets)
Video: Too large for practical Render deployment
"""

import os
import sys
import json
import glob
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import joblib
import re

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE, "kaggle_datasets")
MODELS_DIR = os.path.join(BASE, "..", "models")
REPORTS_DIR = os.path.join(BASE, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved: {path}")

def save_report(module, report):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"{module}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"  Report: {report_path}")

def save_dataset_info(module, info):
    out_path = os.path.join(DATASETS_DIR, module, "dataset_info.json")
    with open(out_path, "w") as f:
        json.dump(info, f, indent=4)

def print_header(module):
    print()
    print("=" * 58)
    print(f"  {module} TRAINING PIPELINE")
    print("=" * 58)

def evaluate(y_test, y_pred, y_proba=None):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    auc  = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            pass
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Confusion Matrix: {cm}")
    return acc, prec, rec, f1, cm, auc

# ============================================================
# 1. EMAIL
# ============================================================

def train_email():
    print_header("EMAIL")
    orig_dir   = os.path.join(DATASETS_DIR, "email", "original")
    prep_dir   = os.path.join(DATASETS_DIR, "email")
    model_dir  = os.path.join(MODELS_DIR, "email")

    csv_path = os.path.join(orig_dir, "Phishing_Email.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] Phishing_Email.csv not found")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Email Text", "Email Type"])
    df.columns = ["idx", "text", "label"]

    total = len(df)
    real_count = (df["label"] == "Safe Email").sum()
    fake_count = (df["label"] == "Phishing Email").sum()

    print(f"  Kaggle Dataset : Phishing Email Detection (subhajournal/phishingemails)")
    print(f"  License        : GNU Lesser General Public License 3.0")
    print(f"  Total records  : {total}")
    print(f"  Safe Email     : {real_count}")
    print(f"  Phishing Email : {fake_count}")
    print()
    print("  ACTUAL DATASET SAMPLES:")
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        print(f"  Sample {i+1}  Label: {row['label']}")
        print(f"           Text : {str(row['text'])[:120]}...")
        print()

    X = df["text"].astype(str)
    y = (df["label"] == "Phishing Email").astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(os.path.join(prep_dir, "train.csv"), index=False)
    pd.DataFrame({"text": X_val,   "label": y_val  }).to_csv(os.path.join(prep_dir, "validation.csv"), index=False)
    pd.DataFrame({"text": X_test,  "label": y_test }).to_csv(os.path.join(prep_dir, "test.csv"), index=False)

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")
    print(f"  Test samples       : {len(X_test)}")

    print("  Training TF-IDF + Logistic Regression...")
    vec = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(Xtr, y_train)

    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]
    print("  Evaluation on Test Set:")
    acc, prec, rec, f1, cm, auc = evaluate(y_test, y_pred, y_proba)

    os.makedirs(model_dir, exist_ok=True)
    save_model(clf, os.path.join(model_dir, "email_model.pkl"))
    save_model(vec, os.path.join(model_dir, "email_vectorizer.pkl"))

    save_dataset_info("email", {
        "dataset_name": "Phishing Email Detection",
        "source": "Kaggle",
        "kaggle_url": "https://www.kaggle.com/datasets/subhajournal/phishingemails",
        "kaggle_owner": "subhajournal",
        "original_file": "Phishing_Email.csv",
        "download_status": "success",
        "total_samples": int(total),
        "classes": {"Safe Email": int(real_count), "Phishing Email": int(fake_count)},
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    })

    save_report("email", {
        "module": "EMAIL",
        "dataset": "Phishing Email Detection",
        "kaggle_url": "https://www.kaggle.com/datasets/subhajournal/phishingemails",
        "model": "TF-IDF + Logistic Regression",
        "total_samples": int(total),
        "train_count": len(X_train),
        "validation_count": len(X_val),
        "test_count": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": cm
    })
    print("  [OK] EMAIL TRAINING COMPLETE\n")


# ============================================================
# 2. MESSAGE / SMS
# ============================================================

def train_message():
    print_header("MESSAGE / SMS")
    orig_dir  = os.path.join(DATASETS_DIR, "message", "original")
    prep_dir  = os.path.join(DATASETS_DIR, "message")
    model_dir = os.path.join(MODELS_DIR, "message")

    csv_path = os.path.join(orig_dir, "spam.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] spam.csv not found")
        return

    df = pd.read_csv(csv_path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df = df.dropna()

    total = len(df)
    ham_count  = (df["label"] == "ham").sum()
    spam_count = (df["label"] == "spam").sum()

    print(f"  Kaggle Dataset : SMS Spam Collection (uciml/sms-spam-collection-dataset)")
    print(f"  Total records  : {total}")
    print(f"  Ham (REAL)     : {ham_count}")
    print(f"  Spam (FAKE)    : {spam_count}")
    print()
    print("  ACTUAL DATASET SAMPLES:")
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        print(f"  Sample {i+1}  Label: {row['label']}")
        print(f"           Text : {str(row['text'])[:120]}")
        print()

    X = df["text"].astype(str)
    y = (df["label"] == "spam").astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(os.path.join(prep_dir, "train.csv"), index=False)
    pd.DataFrame({"text": X_val,   "label": y_val  }).to_csv(os.path.join(prep_dir, "validation.csv"), index=False)
    pd.DataFrame({"text": X_test,  "label": y_test }).to_csv(os.path.join(prep_dir, "test.csv"), index=False)

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")
    print(f"  Test samples       : {len(X_test)}")

    print("  Training TF-IDF + Logistic Regression...")
    vec = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(Xtr, y_train)

    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]
    print("  Evaluation on Test Set:")
    acc, prec, rec, f1, cm, auc = evaluate(y_test, y_pred, y_proba)

    os.makedirs(model_dir, exist_ok=True)
    save_model(clf, os.path.join(model_dir, "message_model.pkl"))
    save_model(vec, os.path.join(model_dir, "message_vectorizer.pkl"))

    save_dataset_info("message", {
        "dataset_name": "SMS Spam Collection",
        "source": "Kaggle",
        "kaggle_url": "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
        "kaggle_owner": "uciml",
        "original_file": "spam.csv",
        "download_status": "success",
        "total_samples": int(total),
        "classes": {"ham": int(ham_count), "spam": int(spam_count)},
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    })

    save_report("message", {
        "module": "MESSAGE/SMS",
        "dataset": "SMS Spam Collection",
        "kaggle_url": "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
        "model": "TF-IDF + Logistic Regression",
        "total_samples": int(total),
        "train_count": len(X_train),
        "validation_count": len(X_val),
        "test_count": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": cm
    })
    print("  [OK] MESSAGE TRAINING COMPLETE\n")


# ============================================================
# 3. URL
# ============================================================

def extract_url_features(url):
    url = str(url)
    return {
        "url_length":        len(url),
        "has_https":         1 if url.startswith("https") else 0,
        "num_dots":          url.count("."),
        "num_subdomains":    max(0, url.count(".") - 1),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9.\-/:]", url)),
        "has_at_symbol":     1 if "@" in url else 0,
        "has_hyphen":        1 if "-" in url else 0,
        "has_ip":            1 if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) else 0,
        "num_query_params":  url.count("=") + url.count("&"),
        "url_depth":         url.count("/"),
        "has_suspicious_tld": 1 if re.search(r"\.(xyz|top|click|gq|tk|ml|cf|work|pw|cc)$", url, re.I) else 0,
    }

def train_url():
    print_header("WEBSITE / URL")
    orig_dir  = os.path.join(DATASETS_DIR, "url", "original")
    prep_dir  = os.path.join(DATASETS_DIR, "url")
    model_dir = os.path.join(MODELS_DIR, "url")

    csv_path = os.path.join(orig_dir, "malicious_phish.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] malicious_phish.csv not found")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["url", "type"])

    total = len(df)
    dist  = df["type"].value_counts()
    print(f"  Kaggle Dataset : Malicious URLs Dataset (sid321axn/malicious-urls-dataset)")
    print(f"  License        : CC0-1.0")
    print(f"  Total records  : {total}")
    print(f"  Class distribution:\n{dist.to_string()}")
    print()
    print("  ACTUAL DATASET SAMPLES:")
    for label in ["benign", "phishing", "malware", "defacement"]:
        sample = df[df["type"] == label].iloc[0] if (df["type"] == label).any() else None
        if sample is not None:
            print(f"  Label: {label}")
            print(f"  URL  : {sample['url']}")
            print()

    # Binary: benign=0, everything else=1
    y = (df["type"] != "benign").astype(int)
    X_raw = df["url"]

    print("  Extracting URL features...")
    features = pd.DataFrame(X_raw.apply(extract_url_features).tolist())

    X_temp, X_test, y_temp, y_test, url_temp, url_test = train_test_split(
        features, y, X_raw, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val, url_train, url_val = train_test_split(
        X_temp, y_temp, url_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    pd.concat([url_train.rename("url"), y_train.rename("label")], axis=1).to_csv(
        os.path.join(prep_dir, "train.csv"), index=False)
    pd.concat([url_val.rename("url"),   y_val.rename("label")],   axis=1).to_csv(
        os.path.join(prep_dir, "validation.csv"), index=False)
    pd.concat([url_test.rename("url"),  y_test.rename("label")],  axis=1).to_csv(
        os.path.join(prep_dir, "test.csv"), index=False)

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")
    print(f"  Test samples       : {len(X_test)}")

    print("  Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print("  Evaluation on Test Set:")
    acc, prec, rec, f1, cm, auc = evaluate(y_test, y_pred, y_proba)

    os.makedirs(model_dir, exist_ok=True)
    save_model(clf, os.path.join(model_dir, "url_model.pkl"))

    # Save feature names for inference
    feature_names = list(features.columns)
    with open(os.path.join(model_dir, "url_features.json"), "w") as f:
        json.dump(feature_names, f)

    save_dataset_info("url", {
        "dataset_name": "Malicious URLs Dataset",
        "source": "Kaggle",
        "kaggle_url": "https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset",
        "kaggle_owner": "sid321axn",
        "original_file": "malicious_phish.csv",
        "download_status": "success",
        "total_samples": int(total),
        "classes": dist.to_dict(),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    })

    save_report("url", {
        "module": "WEBSITE/URL",
        "dataset": "Malicious URLs Dataset",
        "kaggle_url": "https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset",
        "model": "Random Forest",
        "total_samples": int(total),
        "train_count": len(X_train),
        "validation_count": len(X_val),
        "test_count": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": cm
    })
    print("  [OK] URL TRAINING COMPLETE\n")


# ============================================================
# 4. JOB SCAM
# ============================================================

def train_job_scam():
    print_header("INTERNSHIP / JOB SCAM")
    orig_dir  = os.path.join(DATASETS_DIR, "job_scam", "original")
    prep_dir  = os.path.join(DATASETS_DIR, "job_scam")
    model_dir = os.path.join(MODELS_DIR, "job_scam")

    csv_path = os.path.join(orig_dir, "fake_job_postings.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] fake_job_postings.csv not found")
        return

    df = pd.read_csv(csv_path)
    # Combine text fields for richer NLP
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    df = df.dropna(subset=["fraudulent"])

    total      = len(df)
    legit_count = (df["fraudulent"] == 0).sum()
    scam_count  = (df["fraudulent"] == 1).sum()

    print(f"  Kaggle Dataset : Real or Fake Job Postings (shivamb/real-or-fake-fake-jobposting-prediction)")
    print(f"  License        : CC0-1.0")
    print(f"  Total records  : {total}")
    print(f"  Legitimate     : {legit_count}")
    print(f"  Scam/Fraudulent: {scam_count}")
    print()
    print("  ACTUAL DATASET SAMPLES:")
    real_sample = df[df["fraudulent"] == 0].iloc[0]
    fake_sample = df[df["fraudulent"] == 1].iloc[0]
    print(f"  Sample 1 | Label: LEGITIMATE")
    print(f"    Title  : {real_sample['title']}")
    print(f"    Desc   : {str(real_sample['description'])[:100]}...")
    print()
    print(f"  Sample 2 | Label: SCAM")
    print(f"    Title  : {fake_sample['title']}")
    print(f"    Desc   : {str(fake_sample['description'])[:100]}...")
    print()

    X = df["text"].astype(str)
    y = df["fraudulent"].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(os.path.join(prep_dir, "train.csv"), index=False)
    pd.DataFrame({"text": X_val,   "label": y_val  }).to_csv(os.path.join(prep_dir, "validation.csv"), index=False)
    pd.DataFrame({"text": X_test,  "label": y_test }).to_csv(os.path.join(prep_dir, "test.csv"), index=False)

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")
    print(f"  Test samples       : {len(X_test)}")

    print("  Training TF-IDF + Logistic Regression...")
    vec = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(Xtr, y_train)

    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]
    print("  Evaluation on Test Set:")
    acc, prec, rec, f1, cm, auc = evaluate(y_test, y_pred, y_proba)

    os.makedirs(model_dir, exist_ok=True)
    save_model(clf, os.path.join(model_dir, "job_scam_model.pkl"))
    save_model(vec, os.path.join(model_dir, "job_scam_vectorizer.pkl"))

    save_dataset_info("job_scam", {
        "dataset_name": "Real or Fake Job Posting Prediction",
        "source": "Kaggle",
        "kaggle_url": "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction",
        "kaggle_owner": "shivamb",
        "original_file": "fake_job_postings.csv",
        "download_status": "success",
        "total_samples": int(total),
        "classes": {"legitimate": int(legit_count), "scam": int(scam_count)},
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    })

    save_report("job_scam", {
        "module": "INTERNSHIP/JOB SCAM",
        "dataset": "Real or Fake Job Posting Prediction",
        "kaggle_url": "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction",
        "model": "TF-IDF + Logistic Regression (class_weight=balanced)",
        "total_samples": int(total),
        "train_count": len(X_train),
        "validation_count": len(X_val),
        "test_count": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": cm
    })
    print("  [OK] JOB SCAM TRAINING COMPLETE\n")


# ============================================================
# 5. NEWS
# ============================================================

def train_news():
    print_header("FAKE NEWS")
    orig_dir  = os.path.join(DATASETS_DIR, "news", "original")
    prep_dir  = os.path.join(DATASETS_DIR, "news")
    model_dir = os.path.join(MODELS_DIR, "news")

    fake_path = os.path.join(orig_dir, "Fake.csv")
    true_path = os.path.join(orig_dir, "True.csv")
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("[ERROR] Fake.csv or True.csv not found")
        return

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake["label"] = 1  # FAKE
    df_true["label"] = 0  # REAL
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    df = df.dropna(subset=["text", "label"])

    total      = len(df)
    fake_count = (df["label"] == 1).sum()
    real_count = (df["label"] == 0).sum()

    print(f"  Kaggle Dataset : Fake and Real News (clmentbisaillon/fake-and-real-news-dataset)")
    print(f"  License        : CC-BY-NC-SA-4.0")
    print(f"  Total records  : {total}")
    print(f"  REAL news      : {real_count}")
    print(f"  FAKE news      : {fake_count}")
    print()
    print("  ACTUAL DATASET SAMPLES:")
    real_s = df[df["label"] == 0].iloc[0]
    fake_s = df[df["label"] == 1].iloc[0]
    print(f"  Sample 1 | Label: REAL")
    print(f"    Title  : {real_s['title']}")
    print(f"    Text   : {str(real_s['text'])[:100]}...")
    print()
    print(f"  Sample 2 | Label: FAKE")
    print(f"    Title  : {fake_s['title']}")
    print(f"    Text   : {str(fake_s['text'])[:100]}...")
    print()

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(os.path.join(prep_dir, "train.csv"), index=False)
    pd.DataFrame({"text": X_val,   "label": y_val  }).to_csv(os.path.join(prep_dir, "validation.csv"), index=False)
    pd.DataFrame({"text": X_test,  "label": y_test }).to_csv(os.path.join(prep_dir, "test.csv"), index=False)

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")
    print(f"  Test samples       : {len(X_test)}")

    print("  Training TF-IDF + Logistic Regression...")
    vec = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(Xtr, y_train)

    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]
    print("  Evaluation on Test Set:")
    acc, prec, rec, f1, cm, auc = evaluate(y_test, y_pred, y_proba)

    os.makedirs(model_dir, exist_ok=True)
    save_model(clf, os.path.join(model_dir, "news_model.pkl"))
    save_model(vec, os.path.join(model_dir, "news_vectorizer.pkl"))

    save_dataset_info("news", {
        "dataset_name": "Fake and Real News Dataset",
        "source": "Kaggle",
        "kaggle_url": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
        "kaggle_owner": "clmentbisaillon",
        "original_files": ["Fake.csv", "True.csv"],
        "download_status": "success",
        "total_samples": int(total),
        "classes": {"REAL": int(real_count), "FAKE": int(fake_count)},
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test)
    })

    save_report("news", {
        "module": "FAKE NEWS",
        "dataset": "Fake and Real News Dataset",
        "kaggle_url": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
        "model": "TF-IDF + Logistic Regression",
        "total_samples": int(total),
        "train_count": len(X_train),
        "validation_count": len(X_val),
        "test_count": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": cm
    })
    print("  [OK] NEWS TRAINING COMPLETE\n")


# ============================================================
# 6. CALL
# ============================================================

def report_call():
    print_header("PHONE CALL / SCAM CALL")
    print()
    print("  STATUS: NO SUITABLE KAGGLE DATASET FOUND FOR CALL AI TRAINING.")
    print()
    print("  All known Kaggle call transcript datasets returned 403 Forbidden")
    print("  when attempted via Kaggle API with valid credentials.")
    print("  Datasets tried:")
    print("    - ibrahimbagwan/composite-scam-transcript-dataset  -> 403")
    print("    - harrimansheth/fraud-call-detection-dataset       -> 403")
    print("    - marklvl/call-transcripts-scam-determinations     -> 403")
    print()
    print("  Per requirements: No fabricated data created.")
    print("  No model was trained for this module.")
    print("  The existing rule-based call detector in the backend remains unchanged.")
    print()
    save_report("call", {
        "module": "PHONE CALL / SCAM CALL",
        "status": "NO SUITABLE KAGGLE DATASET FOUND",
        "reason": "All identified Kaggle call transcript datasets returned 403 Forbidden via API",
        "model": None,
        "accuracy": None
    })


# ============================================================
# 7. VIDEO
# ============================================================

def report_video():
    print_header("VIDEO DEEPFAKE")
    print()
    print("  STATUS: NO SUITABLE KAGGLE VIDEO DATASET USED.")
    print()
    print("  The Deepfake Detection Challenge (DFDC) dataset is several terabytes.")
    print("  Downloading and training on this dataset is impractical for Render CPU deployment.")
    print()
    print("  Per requirements:")
    print("    - model/deepfake_mobilenetv2_v3_best.keras is NOT modified.")
    print("    - Existing image DeepFake system remains EXACTLY as it was.")
    print("    - No video model was trained or substituted.")
    print()
    save_report("video", {
        "module": "VIDEO DEEPFAKE",
        "status": "NOT TRAINED - Dataset too large for practical Render deployment",
        "existing_model": "model/deepfake_mobilenetv2_v3_best.keras (UNTOUCHED)",
        "model": None,
        "accuracy": None
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 58)
    print("  TRUSTGUARD AI - COMPLETE TRAINING PIPELINE")
    print("=" * 58)

    train_email()
    train_message()
    train_url()
    train_job_scam()
    train_news()
    report_call()
    report_video()

    print()
    print("=" * 58)
    print("  ALL PIPELINES COMPLETE")
    print("=" * 58)
