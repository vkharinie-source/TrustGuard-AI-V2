import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets" / "email_spam" / "phishing_email.csv"
MODEL_DIR = BASE.parent / "models" / "email_v2"
REPORT_DIR = BASE / "reports"

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    print("Loading:", DATA.name)
    df = pd.read_csv(DATA, low_memory=False)

    # This dataset is expected to have text_combined and label.
    required = {"text_combined", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected columns: {df.columns.tolist()}. "
            "Expected text_combined and label; inspect before training."
        )

    df = df.dropna(subset=["text_combined", "label"]).copy()
    df["text_combined"] = df["text_combined"].astype(str).str.strip()
    df = df[df["text_combined"].ne("")]

    # Verify label values before mapping. Expected labels: 0 and 1.
    labels = set(df["label"].astype(str).str.strip().unique())
    print("Original label values:", sorted(labels))
    print("Original label counts:")
    print(df["label"].value_counts().to_string())

    if not labels.issubset({"0", "1"}):
        raise ValueError(
            "Unexpected label values. Do not train until label meaning is verified."
        )

    # Dataset's numeric labels: preserve the supplied mapping.
    # Verify whether 1 means phishing/spam and 0 means legitimate
    # from the dataset source before using this model in production.
    X = df["text_combined"]
    y = df["label"].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 3:
        raise ValueError("Both classes need at least 3 samples.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\nTotal samples: {len(X)}")
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

    print("\nTraining email model...")
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("\nVALIDATION")
    print("Accuracy:", round(accuracy_score(y_val, val_pred), 4))
    print(classification_report(y_val, val_pred, zero_division=0))

    print("\nFINAL TEST")
    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist()

    print("Accuracy:", round(acc, 4))
    print("Precision (label 1):", round(prec, 4))
    print("Recall (label 1):", round(rec, 4))
    print("F1 (label 1):", round(f1, 4))
    print("Confusion matrix [0, 1]:", cm)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "email_pipeline.pkl"
    joblib.dump(model, model_path)

    report = {
        "dataset": DATA.name,
        "text_column": "text_combined",
        "label_column": "label",
        "label_mapping": "Preserved source labels 0 and 1; confirm their meaning from dataset documentation.",
        "model": "TF-IDF + Logistic Regression",
        "total_samples": int(len(X)),
        "train_count": int(len(X_train)),
        "validation_count": int(len(X_val)),
        "test_count": int(len(X_test)),
        "accuracy": float(acc),
        "precision_label_1": float(prec),
        "recall_label_1": float(rec),
        "f1_label_1": float(f1),
        "confusion_matrix_labels_0_1": cm,
        "model_file": str(model_path)
    }

    report_path = REPORT_DIR / "email_v2_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nModel saved:", model_path)
    print("Report saved:", report_path)
    print("Existing models and reports were not overwritten.")

if __name__ == "__main__":
    main()
