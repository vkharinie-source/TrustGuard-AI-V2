import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def train_url_model():
    print("--- Training ML Model for WEBSITE / URL ---")
    module_name = "url"
    data_path = "../kaggle_datasets/url/malicious_urls.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Expected features (Ensure these exist in your downloaded Kaggle dataset or perform feature engineering)
    # If the dataset provides raw URLs, you need to extract these features first.
    # We assume the dataset has pre-extracted features for simplicity in this template.
    feature_cols = [
        'url_length', 'has_https', 'num_dots', 'num_subdomains', 
        'num_special_chars', 'has_at_symbol', 'has_hyphen', 
        'has_ip', 'num_query_params'
    ]
    
    label_col = "label" # Example: 1 for malicious/phishing, 0 for benign
    
    if not all(col in df.columns for col in feature_cols + [label_col]):
        print("Missing required feature columns. Please extract URL features from the raw dataset first.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    df = df.dropna(subset=feature_cols + [label_col])
    X = df[feature_cols]
    y = df[label_col].astype(int)

    # Train/Val/Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model on Test Set...")
    y_pred = model.predict(X_test)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    model_dir = f"../../models/{module_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{module_name}_model.pkl")
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_url_model()
