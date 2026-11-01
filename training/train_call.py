import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

MODULE = "call"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_datasets", MODULE)
ORIGINAL_DIR = os.path.join(DATA_DIR, "original")
PREPARED_DIR = os.path.join(DATA_DIR, "prepared")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", MODULE)
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

def main():
    print(f"==================================================")
    print(f"{MODULE.upper()} TRAINING PIPELINE")
    print(f"==================================================")

    if not os.path.exists(ORIGINAL_DIR):
        print("[ERROR] Original dataset directory not found.")
        return

    csv_files = glob.glob(os.path.join(ORIGINAL_DIR, "*.csv"))
    if not csv_files:
        print("[ERROR] Dataset not found. Dataset is empty.")
        print(f"Please download the Kaggle dataset into: {ORIGINAL_DIR}")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_files[0])
    
    if df.empty:
        print("[ERROR] Dataset is empty.")
        return

    print("Dataset verified.")
    print(f"Total samples: {len(df)}")
    
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[-1]

    print(f"Class distribution:")
    print(df[label_col].value_counts())

    print("\nSample records:")
    for i in range(min(3, len(df))):
        print(f"Sample {i+1}:\nLabel: {df.iloc[i][label_col]}\nText: {str(df.iloc[i][text_col])[:100]}...\n")

    os.makedirs(PREPARED_DIR, exist_ok=True)
    df = df.dropna(subset=[text_col, label_col])
    
    y = df[label_col].astype(str).apply(lambda x: 1 if x in ['1', 'spam', 'phishing', 'FAKE', 'fraudulent'] else 0)
    X = df[text_col].astype(str)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

    pd.DataFrame({'text': X_train, 'label': y_train}).to_csv(os.path.join(PREPARED_DIR, 'train.csv'), index=False)
    pd.DataFrame({'text': X_val, 'label': y_val}).to_csv(os.path.join(PREPARED_DIR, 'validation.csv'), index=False)
    pd.DataFrame({'text': X_test, 'label': y_test}).to_csv(os.path.join(PREPARED_DIR, 'test.csv'), index=False)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    print("\nTraining model...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    print("Training completed.\n")

    print("Evaluation:")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion Matrix: {cm}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{MODULE}_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, f"{MODULE}_vectorizer.pkl"))
    print(f"\nModel saved to {MODEL_DIR}")

    report = {
        "dataset": os.path.basename(csv_files[0]),
        "model": "TF-IDF + Logistic Regression",
        "train_count": len(X_train),
        "validation_count": len(X_val),
        "test_count": len(X_test),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, f"{MODULE}_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Report saved to {REPORT_DIR}/{MODULE}_report.json")

if __name__ == "__main__":
    main()
