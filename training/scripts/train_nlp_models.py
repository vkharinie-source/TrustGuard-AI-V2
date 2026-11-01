import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def train_model(module_name, data_path, text_col, label_col, positive_label):
    print(f"--- Training NLP Model for {module_name.upper()} ---")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print(f"Please download the Kaggle dataset and place it at {data_path}")
        return

    # Load Dataset
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    if text_col not in df.columns or label_col not in df.columns:
        print(f"Error: Columns '{text_col}' and/or '{label_col}' not found in the dataset.")
        print(f"Available columns: {df.columns.tolist()}")
        print("Please update the script with the correct column names.")
        return

    # Preprocessing
    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str)
    
    # Convert labels to binary (1 for fake/spam/scam, 0 for real/legitimate)
    y = (df[label_col] == positive_label).astype(int)

    # Train/Val/Test Split (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42) # 0.1765 of 0.85 is ~0.15

    print(f"Samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Vectorization
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluation
    print("Evaluating model on Test Set...")
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Saving Model and Vectorizer
    model_dir = f"../../models/{module_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{module_name}_model.pkl")
    vec_path = os.path.join(model_dir, f"{module_name}_vectorizer.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vec_path}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    # Change current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. EMAIL
    # Example config (Adjust column names based on the actual Kaggle dataset structure)
    train_model(
        module_name="email",
        data_path="../kaggle_datasets/email/phishing_email.csv",
        text_col="text_combined", # Replace with actual column containing email text
        label_col="label",       # Replace with actual label column
        positive_label=1         # Replace with actual positive class value (e.g., 1 for Phishing)
    )

    # 2. MESSAGE / SMS
    train_model(
        module_name="message",
        data_path="../kaggle_datasets/message/spam.csv",
        text_col="v2",
        label_col="v1",
        positive_label="spam"
    )

    # 4. INTERNSHIP / JOB SCAM
    train_model(
        module_name="job_scam",
        data_path="../kaggle_datasets/job_scam/fake_job_postings.csv",
        text_col="description",
        label_col="fraudulent",
        positive_label=1
    )

    # 5. NEWS
    train_model(
        module_name="news",
        data_path="../kaggle_datasets/news/fake_and_real_news.csv",
        text_col="text",
        label_col="label",
        positive_label="FAKE"
    )

    # 6. PHONE CALL / SCAM CALL
    train_model(
        module_name="call",
        data_path="../kaggle_datasets/call/scam_transcripts.csv",
        text_col="transcript",
        label_col="is_scam",
        positive_label=1
    )
