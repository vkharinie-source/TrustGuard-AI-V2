import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets" / "job_scam" / "DataSet.csv"
OUT = BASE.parent / "models" / "job_scam_v2"
REPORTS = BASE / "reports"

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA, low_memory=False)
    required = ["fraudulent", "title", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # In this dataset, f = legitimate and t = fraudulent.
    labels = df["fraudulent"].astype(str).str.strip().str.lower()
    mapping = {"f": 0, "t": 1}
    valid = labels.isin(mapping)

    df = df.loc[valid].copy()
    y = labels.loc[valid].map(mapping).astype(int)

    text_columns = [
        "title", "location", "department", "salary_range",
        "company_profile", "description", "requirements",
        "benefits", "employment_type", "required_experience",
        "required_education", "industry", "function"
    ]
    columns = [c for c in text_columns if c in df.columns]
    X = df[columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()

    nonempty = X.ne("")
    X, y = X[nonempty], y[nonempty]

    if y.nunique() != 2 or y.value_counts().min() < 3:
        raise ValueError("Not enough examples in both classes to train safely.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("Dataset rows:", len(X))
    print("Label counts (0=legitimate, 1=fraudulent):")
    print(y.value_counts().sort_index().to_string())
    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Validation metrics are reported separately from final test metrics.
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("\nVALIDATION RESULTS")
    print("Accuracy:", round(accuracy_score(y_val, val_pred), 4))
    print(classification_report(
        y_val, val_pred, labels=[0, 1],
        target_names=["legitimate", "fraudulent"],
        zero_division=0
    ))

    print("FINAL TEST RESULTS")
    print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
    print("Precision:", round(precision_score(y_test, test_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, test_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_test, test_pred, zero_division=0), 4))
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist()
    print("Confusion matrix [legitimate, fraudulent]:", cm)

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    model_path = OUT / "job_scam_pipeline.pkl"
    joblib.dump(model, model_path)

    report = {
        "module": "job_scam",
        "dataset": str(DATA),
        "label_mapping": {"f": "legitimate (0)", "t": "fraudulent (1)"},
        "model": "TF-IDF + Logistic Regression",
        "total_samples": int(len(X)),
        "train_count": int(len(X_train)),
        "validation_count": int(len(X_val)),
        "test_count": int(len(X_test)),
        "validation_accuracy": round(accuracy_score(y_val, val_pred), 4),
        "accuracy": round(accuracy_score(y_test, test_pred), 4),
        "precision": round(precision_score(y_test, test_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, test_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, test_pred, zero_division=0), 4),
        "confusion_matrix": cm,
        "positive_class": "fraudulent",
        "model_file": str(model_path)
    }

    report_path = REPORTS / "job_scam_v2_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSaved model:", model_path)
    print("Saved report:", report_path)
    print("Existing job-scam model files were not modified.")

if __name__ == "__main__":
    main()
