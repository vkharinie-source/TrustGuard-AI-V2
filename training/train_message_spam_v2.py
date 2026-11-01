
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "training" / "datasets" / "message_spam" / "dataset.csv"
OUTPUT_DIR = ROOT / "models" / "message_spam_v2"
REPORT_DIR = ROOT / "training" / "reports"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    text_candidates = [
        c for c in df.columns
        if c.strip().lower() in {"text", "message", "sms", "v2", "content"}
    ]
    label_candidates = [
        c for c in df.columns
        if c.strip().lower() in {"label", "labels", "type", "category", "v1"}
    ]

    if not text_candidates or not label_candidates:
        raise ValueError(f"Cannot identify text/label columns: {df.columns.tolist()}")

    df = df[[text_candidates[0], label_candidates[0]]].copy()
    df.columns = ["text", "label"]
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]

    label_map = {"ham": 0, "spam": 1}
    unexpected = sorted(set(df["label"]) - set(label_map))
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")

    df["label"] = df["label"].map(label_map).astype(int)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    # 70% train, 15% validation, 15% test.
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=train_val_df["label"],
    )

    print("Dataset samples:", len(df))
    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("\nTraining model...")
    pipeline.fit(train_df["text"], train_df["label"])

    def evaluate(split_df):
        predictions = pipeline.predict(split_df["text"])
        return {
            "samples": int(len(split_df)),
            "accuracy": float(accuracy_score(split_df["label"], predictions)),
            "spam_precision": float(precision_score(
                split_df["label"], predictions, pos_label=1, zero_division=0
            )),
            "spam_recall": float(recall_score(
                split_df["label"], predictions, pos_label=1, zero_division=0
            )),
            "spam_f1": float(f1_score(
                split_df["label"], predictions, pos_label=1, zero_division=0
            )),
            "confusion_matrix": confusion_matrix(
                split_df["label"], predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                split_df["label"],
                predictions,
                labels=[0, 1],
                target_names=["ham", "spam"],
                output_dict=True,
                zero_division=0,
            ),
        }

    print("\nEvaluating validation set...")
    validation_results = evaluate(val_df)

    print("Evaluating test set...")
    test_results = evaluate(test_df)

    results = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "label_mapping": {"ham": 0, "spam": 1},
        "split": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "total_samples": int(len(df)),
        "validation": validation_results,
        "test": test_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "message_spam_pipeline.pkl"
    report_path = REPORT_DIR / "message_spam_v2_report.json"

    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== VALIDATION RESULTS ===")
    print(json.dumps(validation_results, indent=2))

    print("\n=== TEST RESULTS ===")
    print(classification_report(
        test_df["label"],
        pipeline.predict(test_df["text"]),
        labels=[0, 1],
        target_names=["ham", "spam"],
        zero_division=0,
    ))

    print("Model saved:", model_path)
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
