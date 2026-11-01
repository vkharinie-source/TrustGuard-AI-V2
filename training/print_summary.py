import json
from pathlib import Path

reports_dir = Path("D:/project/DeepFake-Detector-AI/training/reports")
files = [
    ("Email Spam Detection V3", "email_spam_training_v3_report.json"),
    ("Fake News Detection V2", "fake_news_training_v2_report.json"),
    ("Job Scam Detection V3", "job_scam_training_v3_report.json"),
    ("Phishing URL Detection V3", "phishing_urls_training_v3_report.json"),
    ("Scam Call Detection V2", "scam_calls_training_v2_report.json"),
    ("Message Spam Detection V3", "message_spam_training_v3_report.json")
]

for title, fname in files:
    fpath = reports_dir / fname
    with open(fpath, "r", encoding="utf-8") as fp:
        d = json.load(fp)
    val = d.get("validation_best_metrics", {})
    t = d.get("final_test_metrics", {})
    sp = d.get("split_counts", {})
    print(f"Name: {title}")
    print(f"Dataset: {d.get('dataset_name')}")
    print(f"Splits: Train={sp.get('training')}, Val={sp.get('validation')}, Test={sp.get('test')}")
    print(f"Validation: Acc={val.get('accuracy'):.4f}, F1={val.get('f1_score'):.4f}")
    print(f"Final Test: Acc={t.get('accuracy'):.4f}, Prec={t.get('precision'):.4f}, Rec={t.get('recall'):.4f}, F1={t.get('f1_score'):.4f}")
    print(f"Model Path: {d.get('model_output_path')}")
    print(f"Report Path: {str(fpath)}")
    print(f"Status: {d.get('training_status')}")
    print("-" * 50)
