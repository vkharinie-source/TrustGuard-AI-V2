from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)

data_dir = Path("training/datasets/scam_calls/prepared")
model_dir = Path("models/call_prepared_v1")
report_dir = Path("training/reports")

model_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

def load_split(filename):
    df = pd.read_csv(data_dir / filename)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {filename}")

    return df

train = load_split("train.csv")
val = load_split("validation.csv")
test = load_split("test.csv")

print("Train:", train.shape, train["label"].value_counts().to_dict())
print("Validation:", val.shape, val["label"].value_counts().to_dict())
print("Test:", test.shape, test["label"].value_counts().to_dict())

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining scam-call text classifier...")
model.fit(train["text"], train["label"])

def evaluate(df, split_name):
    y = df["label"]
    pred = model.predict(df["text"])
    prob = model.predict_proba(df["text"])[:, 1]

    return {
        "split": split_name,
        "samples": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "scam_precision": float(precision_score(y, pred, zero_division=0)),
        "scam_recall": float(recall_score(y, pred, zero_division=0)),
        "scam_f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y, pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y, pred,
            labels=[0, 1],
            target_names=["normal", "scam"],
            output_dict=True,
            zero_division=0
        )
    }

# Validation results are reported separately; test data is used only for final evaluation.
val_results = evaluate(val, "validation")
test_results = evaluate(test, "test")

joblib.dump(model, model_dir / "call_pipeline.pkl")

report = {
    "module": "scam_call_text_classification",
    "dataset": "prepared train.csv, validation.csv, test.csv",
    "label_mapping": {"0": "normal", "1": "scam"},
    "model": "TF-IDF + Logistic Regression",
    "train_samples": int(len(train)),
    "validation": val_results,
    "test": test_results,
    "model_file": str(model_dir / "call_pipeline.pkl")
}

(report_dir / "call_prepared_v1_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n========== SCAM CALL TEST RESULTS ==========")
print("Accuracy:", round(test_results["accuracy"], 4))
print("Scam precision:", round(test_results["scam_precision"], 4))
print("Scam recall:", round(test_results["scam_recall"], 4))
print("Scam F1:", round(test_results["scam_f1"], 4))
print("ROC AUC:", round(test_results["roc_auc"], 4))
print("Confusion matrix:", test_results["confusion_matrix_labels_0_1"])
print("\nSaved model:", model_dir / "call_pipeline.pkl")
print("Saved report:", report_dir / "call_prepared_v1_report.json")
