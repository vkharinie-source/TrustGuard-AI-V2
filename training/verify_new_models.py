import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

base_dir = Path("D:/project/DeepFake-Detector-AI")
models_dir = base_dir / "models"
reports_dir = base_dir / "training" / "reports"

tests = [
    {
        "name": "Email Spam V3",
        "model_path": models_dir / "email_spam_training_v3" / "email_spam_pipeline.pkl",
        "report_path": reports_dir / "email_spam_training_v3_report.json",
        "image_path": reports_dir / "email_spam_training_v3_cm.png",
        "sample_input": ["Urgent: Please update your bank account information immediately by clicking this link."],
        "input_type": "text"
    },
    {
        "name": "Fake News V2",
        "model_path": models_dir / "fake_news_training_v2" / "fake_news_pipeline.pkl",
        "report_path": reports_dir / "fake_news_training_v2_report.json",
        "image_path": reports_dir / "fake_news_training_v2_cm.png",
        "sample_input": ["Breaking News: Shocking government conspiracy revealed in secret documents released today."],
        "input_type": "text"
    },
    {
        "name": "Job Scam V3",
        "model_path": models_dir / "job_scam_training_v3" / "job_scam_pipeline.pkl",
        "report_path": reports_dir / "job_scam_training_v3_report.json",
        "image_path": reports_dir / "job_scam_training_v3_cm.png",
        "sample_input": ["Work from home data entry clerk earn $5000 weekly wire money for startup equipment."],
        "input_type": "text"
    },
    {
        "name": "Phishing URLs V3",
        "model_path": models_dir / "phishing_urls_training_v3" / "phishing_urls_pipeline.pkl",
        "report_path": reports_dir / "phishing_urls_training_v3_report.json",
        "image_path": reports_dir / "phishing_urls_training_v3_cm.png",
        "features_path": models_dir / "phishing_urls_training_v3" / "url_features.json",
        "input_type": "numeric"
    },
    {
        "name": "Scam Calls V2",
        "model_path": models_dir / "scam_calls_training_v2" / "scam_calls_pipeline.pkl",
        "report_path": reports_dir / "scam_calls_training_v2_report.json",
        "image_path": reports_dir / "scam_calls_training_v2_cm.png",
        "sample_input": ["This is Officer Davis from IRS, your social security number has been flagged and warrant issued."],
        "input_type": "text"
    },
    {
        "name": "Message Spam V3",
        "model_path": models_dir / "message_spam_training_v3" / "message_spam_pipeline.pkl",
        "report_path": reports_dir / "message_spam_training_v3_report.json",
        "image_path": reports_dir / "message_spam_training_v3_cm.png",
        "sample_input": ["Congratulations! You have won a $1000 Walmart gift card. Text CLAIM to 8899 now."],
        "input_type": "text"
    }
]

print("=== VERIFICATION OF ALL 6 TRAINED MODELS & ARTIFACTS ===\n")

for t in tests:
    print(f"Checking {t['name']}...")
    assert t["model_path"].exists(), f"Model file missing: {t['model_path']}"
    assert t["report_path"].exists(), f"Report missing: {t['report_path']}"
    assert t["image_path"].exists(), f"Image missing: {t['image_path']}"
    
    with open(t["report_path"], "r", encoding="utf-8") as f:
        rep = json.load(f)
    print(f"  Report verified. Status: {rep.get('training_status')}, Test Acc: {rep['final_test_metrics']['accuracy']:.4f}, Test F1: {rep['final_test_metrics']['f1_score']:.4f}")

    pipeline = joblib.load(t["model_path"])
    print("  Loaded pipeline successfully.")
    
    if t["input_type"] == "text":
        pred = pipeline.predict(t["sample_input"])
        proba = pipeline.predict_proba(t["sample_input"])[0] if hasattr(pipeline, "predict_proba") else None
        print(f"  Sample prediction: {pred[0]} (probabilities: {proba})")
    elif t["input_type"] == "numeric":
        test_df = pd.read_parquet(base_dir / "training" / "datasets" / "phishing_urls" / "Testing.parquet")
        with open(t["features_path"], "r") as f:
            feat_cols = json.load(f)
        sample = test_df[feat_cols].iloc[:2].replace([np.inf, -np.inf], np.nan)
        pred = pipeline.predict(sample)
        proba = pipeline.predict_proba(sample)[0] if hasattr(pipeline, "predict_proba") else None
        print(f"  Numeric sample prediction on 2 rows: {pred.tolist()} (first row proba: {proba})")

    print("  [PASS]\n")

print("ALL 6 MODELS VERIFIED AND OPERATIONAL.")
