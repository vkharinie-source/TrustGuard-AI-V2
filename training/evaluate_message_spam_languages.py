
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "training" / "datasets" / "message_spam" / "dataset.csv"
MODEL_PATH = ROOT / "models" / "message_spam_v1" / "message_spam_pipeline.pkl"

RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df[["text", "labels", "lang"]].dropna().copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["labels"] = df["labels"].astype(str).str.strip().str.lower()
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]
    df = df[df["labels"].isin(["ham", "spam"])]

    df["target"] = df["labels"].map({"ham": 0, "spam": 1})

    # Match the model training split, keeping the test set separate.
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(test_df["text"])

    test_df = test_df.copy()
    test_df["prediction"] = predictions

    print("=== OVERALL TEST RESULTS ===")
    print("Accuracy:", round(
        accuracy_score(test_df["target"], test_df["prediction"]), 4
    ))
    print(classification_report(
        test_df["target"],
        test_df["prediction"],
        labels=[0, 1],
        target_names=["ham", "spam"],
        zero_division=0,
    ))

    print("\n=== RESULTS BY LANGUAGE ===")
    for language, group in test_df.groupby("lang"):
        y_true = group["target"]
        y_pred = group["prediction"]

        print(f"\n{language.upper()} (n={len(group)})")
        print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
        print("Spam precision:", round(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4
        ))
        print("Spam recall:", round(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4
        ))
        print("Spam F1:", round(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4
        ))


if __name__ == "__main__":
    main()
