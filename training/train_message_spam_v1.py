
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
OUTPUT_DIR = ROOT / "models" / "message_spam_v1"
REPORT_DIR = ROOT / "training" / "reports"

RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Dataset columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)

    # Find the message and label columns without assuming their names.
    text_candidates = [
        c for c in df.columns
        if c.strip().lower() in {"text", "message", "sms", "v2", "content"}
    ]
    label_candidates = [
        c for c in df.columns
        if c.strip().lower() in {"label", "labels", "type", "category", "v1"}
    ]

    if not text_candidates or not label_candidates:
        raise ValueError(
            "Could not identify text/label columns. "
            f"Available columns: {df.columns.tolist()}"
        )

    text_col = text_candidates[0]
    label_col = label_candidates[0]

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]

    # Explicitly map original dataset labels.
    label_map = {"ham": 0, "spam": 1}
    unexpected = sorted(set(df["label"]) - set(label_map))

    if unexpected:
        raise ValueError(
            f"Unexpected labels: {unexpected}. "
            "Review the dataset before training."
        )

    df["label"] = df["label"].map(label_map).astype(int)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    print("\nClass counts (0=ham, 1=spam):")
    print(df["label"].value_counts().sort_index())
    print("Total usable messages:", len(df))

    # Keep a held-out test set.
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

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

    predictions = pipeline.predict(test_df["text"])
    probabilities = pipeline.predict_proba(test_df["text"])[:, 1]

    report = classification_report(
        test_df["label"],
        predictions,
        labels=[0, 1],
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "label_mapping": {"ham": 0, "spam": 1},
        "total_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "accuracy": float(accuracy_score(test_df["label"], predictions)),
        "spam_precision": float(
            precision_score(test_df["label"], predictions, pos_label=1,
                            zero_division=0)
        ),
        "spam_recall": float(
            recall_score(test_df["label"], predictions, pos_label=1,
                         zero_division=0)
        ),
        "spam_f1": float(
            f1_score(test_df["label"], predictions, pos_label=1,
                     zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            test_df["label"], predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
    }

    # Save only into new versioned folders.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "message_spam_pipeline.pkl"
    report_path = REPORT_DIR / "message_spam_v1_report.json"

    joblib.dump(pipeline, model_path)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(classification_report(
        test_df["label"],
        predictions,
        labels=[0, 1],
        target_names=["ham", "spam"],
        zero_division=0,
    ))
    print("Confusion matrix [[ham, spam], [ham, spam]]:")
    print(results["confusion_matrix"])
    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")

    samples = [
        "Congratulations! You have won a prize. Call now to claim.",
        "Hi, are we meeting for lunch today?",
    ]

    print("\n=== QUICK SANITY CHECK ===")
    for message in samples:
        predicted = int(pipeline.predict([message])[0])
        probability = float(pipeline.predict_proba([message])[0][1])
        print({
            "message": message,
            "prediction": "spam" if predicted == 1 else "ham",
            "spam_probability": round(probability, 4),
        })


if __name__ == "__main__":
    main()
