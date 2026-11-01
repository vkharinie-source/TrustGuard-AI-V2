from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "datasets" / "phishing_urls"

# New output directory: preserve previous model versions.
OUT = ROOT / "models" / "url_parquet_v2"
REPORTS = ROOT / "training" / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LABEL_COL = "status"
LABEL_MAPPING = {"legitimate": 0, "phishing": 1}

print("Loading Parquet datasets...")
train_df = pd.read_parquet(DATA / "Training.parquet")
test_df = pd.read_parquet(DATA / "Testing.parquet")

for name, df in [("Training", train_df), ("Testing", test_df)]:
    if LABEL_COL not in df.columns:
        raise ValueError(f"{name}.parquet has no '{LABEL_COL}' column.")

    if df[LABEL_COL].isna().any():
        raise ValueError(f"{name}.parquet contains missing labels.")

    unexpected = (
        set(df[LABEL_COL].astype(str).str.lower().unique())
        - set(LABEL_MAPPING)
    )
    if unexpected:
        raise ValueError(f"Unexpected labels in {name}: {unexpected}")

y_all = train_df[LABEL_COL].astype(str).str.lower().map(LABEL_MAPPING)
y_test = test_df[LABEL_COL].astype(str).str.lower().map(LABEL_MAPPING)

feature_cols = [
    col for col in train_df.columns
    if col not in ["url", LABEL_COL]
    and pd.api.types.is_numeric_dtype(train_df[col])
]

if not feature_cols:
    raise ValueError("No numeric feature columns found.")

missing_features = set(feature_cols) - set(test_df.columns)
if missing_features:
    raise ValueError(
        f"Testing dataset is missing features: {sorted(missing_features)}"
    )

X_all = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Split only the provided training data. Keep Testing.parquet untouched.
X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y_all,
)

print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))
print("Numeric features:", len(feature_cols))
print("Training label counts:", y_train.value_counts().to_dict())
print("Validation label counts:", y_val.value_counts().to_dict())
print("Testing label counts:", y_test.value_counts().to_dict())

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

print("\nTraining Random Forest...")
pipeline.fit(X_train, y_train)


def evaluate_split(name, X, y):
    pred = pipeline.predict(X)
    prob = pipeline.predict_proba(X)[:, 1]

    result = {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_phishing": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall_phishing": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1_phishing": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y, pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y,
            pred,
            labels=[0, 1],
            target_names=["legitimate", "phishing"],
            output_dict=True,
            zero_division=0,
        ),
    }

    print(f"\n===== {name.upper()} RESULTS =====")
    print("Accuracy:", round(result["accuracy"], 4))
    print("Phishing precision:", round(result["precision_phishing"], 4))
    print("Phishing recall:", round(result["recall_phishing"], 4))
    print("Phishing F1:", round(result["f1_phishing"], 4))
    print("ROC AUC:", round(result["roc_auc"], 4))
    print("Confusion matrix:", result["confusion_matrix_labels_0_1"])

    return result


validation_results = evaluate_split("validation", X_val, y_val)

# Final test evaluation occurs only after training and validation evaluation.
test_results = evaluate_split("test", X_test, y_test)

model_path = OUT / "url_pipeline.pkl"
feature_path = OUT / "url_features.json"
report_path = REPORTS / "url_parquet_v2_report.json"

joblib.dump(pipeline, model_path)
feature_path.write_text(
    json.dumps(feature_cols, indent=2),
    encoding="utf-8",
)

report = {
    "module": "phishing_url",
    "dataset_format": "Parquet",
    "training_dataset": "Training.parquet",
    "test_dataset": "Testing.parquet",
    "label_mapping": LABEL_MAPPING,
    "model": "Median imputation + Random Forest",
    "feature_count": len(feature_cols),
    "split_method": "Stratified 85/15 split of Training.parquet",
    "random_state": RANDOM_STATE,
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "validation": validation_results,
    "test": test_results,
    "model_file": str(model_path.relative_to(ROOT)),
    "feature_file": str(feature_path.relative_to(ROOT)),
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nSaved new model:", model_path)
print("Saved feature list:", feature_path)
print("Saved report:", report_path)
print("\nExisting url and url_parquet_v1 models were not targeted.")
