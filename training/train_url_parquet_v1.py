from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)

ROOT = Path(".")
DATA = ROOT / "training" / "datasets" / "phishing_urls"
OUT = ROOT / "models" / "url_parquet_v1"
REPORTS = ROOT / "training" / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

print("Loading Parquet datasets...")
train_df = pd.read_parquet(DATA / "Training.parquet")
test_df = pd.read_parquet(DATA / "Testing.parquet")

label_col = "status"
mapping = {"legitimate": 0, "phishing": 1}

y_train = train_df[label_col].astype(str).str.lower().map(mapping)
y_test = test_df[label_col].astype(str).str.lower().map(mapping)

if y_train.isna().any() or y_test.isna().any():
    raise ValueError("Unexpected labels found in status column.")

# Use only numeric website features; exclude URL text and labels.
feature_cols = [
    c for c in train_df.columns
    if c not in ["url", label_col]
    and pd.api.types.is_numeric_dtype(train_df[c])
]

if not feature_cols:
    raise ValueError("No numeric feature columns were found.")

missing_test_features = set(feature_cols) - set(test_df.columns)
if missing_test_features:
    raise ValueError(f"Testing data is missing features: {missing_test_features}")

X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numeric features:", len(feature_cols))
print("Training labels:", y_train.value_counts().to_dict())

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
pipeline.fit(X_train, y_train)

print("Evaluating on separate Testing.parquet...")
pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:, 1]

report = {
    "module": "phishing_url",
    "dataset": "Training.parquet and Testing.parquet",
    "label_mapping": mapping,
    "model": "Median imputation + Random Forest",
    "feature_count": len(feature_cols),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_phishing": float(precision_score(y_test, pred, zero_division=0)),
    "recall_phishing": float(recall_score(y_test, pred, zero_division=0)),
    "f1_phishing": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "confusion_matrix_labels_0_1": confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).tolist(),
    "classification_report": classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["legitimate", "phishing"],
        output_dict=True,
        zero_division=0
    ),
    "model_file": str(OUT / "url_pipeline.pkl"),
    "feature_file": str(OUT / "url_features.json")
}

joblib.dump(pipeline, OUT / "url_pipeline.pkl")
(OUT / "url_features.json").write_text(
    json.dumps(feature_cols, indent=2),
    encoding="utf-8"
)
(REPORTS / "url_parquet_v1_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n========== URL MODEL TEST RESULTS ==========")
print("Accuracy:", round(report["accuracy"], 4))
print("Phishing precision:", round(report["precision_phishing"], 4))
print("Phishing recall:", round(report["recall_phishing"], 4))
print("Phishing F1:", round(report["f1_phishing"], 4))
print("ROC AUC:", round(report["roc_auc"], 4))
print("Confusion matrix:", report["confusion_matrix_labels_0_1"])
print("\nSaved model:", OUT / "url_pipeline.pkl")
print("Saved report:", REPORTS / "url_parquet_v1_report.json")
