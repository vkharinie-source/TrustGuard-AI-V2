import os
import glob
import pandas as pd
import json

MODULES = {
    "EMAIL": {
        "path": "kaggle_datasets/email/original",
        "expected_ext": "*.csv",
        "label_col": "label",
        "real_val": 0,
        "fake_val": 1
    },
    "MESSAGE": {
        "path": "kaggle_datasets/message/original",
        "expected_ext": "*.csv",
        "label_col": "v1",
        "real_val": "ham",
        "fake_val": "spam"
    },
    "URL": {
        "path": "kaggle_datasets/url/original",
        "expected_ext": "*.csv",
        "label_col": "label", # Note: User may need to adjust based on exact downloaded CSV
        "real_val": "benign",
        "fake_val": "phishing"
    },
    "JOB SCAM": {
        "path": "kaggle_datasets/job_scam/original",
        "expected_ext": "*.csv",
        "label_col": "fraudulent",
        "real_val": 0,
        "fake_val": 1
    },
    "NEWS": {
        "path": "kaggle_datasets/news/original",
        "expected_ext": "*.csv",
        "label_col": "label", # Varies by dataset
        "real_val": "REAL",
        "fake_val": "FAKE"
    },
    "CALL": {
        "path": "kaggle_datasets/call/original",
        "expected_ext": "*.csv",
        "label_col": "is_scam",
        "real_val": 0,
        "fake_val": 1
    },
    "VIDEO": {
        "path": "kaggle_datasets/video/original",
        "expected_ext": "*.mp4", # Videos
        "label_col": None,
        "real_val": None,
        "fake_val": None
    }
}

def verify_modules():
    print("==================================================")
    print("TRUSTGUARD KAGGLE DATASET VERIFICATION")
    print("==================================================\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for mod_name, config in MODULES.items():
        print(f"{mod_name}")
        mod_path = os.path.join(base_dir, config["path"])
        
        if not os.path.exists(mod_path):
            print("[ERROR] Dataset folder not found")
            print()
            continue
            
        files = glob.glob(os.path.join(mod_path, config["expected_ext"]))
        
        if not files:
            print("[ERROR] Dataset files not found")
            print(f"-> Please configure Kaggle API (save kaggle.json in ~/.kaggle/)")
            print(f"-> Download dataset into: {mod_path}")
            print()
            continue
            
        print("[OK] Dataset exists")
        print(f"[OK] Files found: {len(files)}")
        
        if config["expected_ext"] == "*.csv":
            try:
                # Assuming first csv is the main dataset for simplicity
                df = pd.read_csv(files[0])
                total_records = len(df)
                print(f"[OK] Records found: {total_records}")
                
                label_col = config["label_col"]
                if label_col in df.columns:
                    real_count = len(df[df[label_col] == config["real_val"]])
                    fake_count = len(df[df[label_col] == config["fake_val"]])
                    print(f"[OK] REAL samples: {real_count}")
                    print(f"[OK] FAKE samples: {fake_count}")
                    
                    print("\nSample records:")
                    for i in range(min(3, total_records)):
                        print(f"{i+1}. {df.iloc[i].to_dict()}")
                else:
                    print(f"[WARNING] Label column '{label_col}' not found. Check CSV headers.")
                    print("\nSample records:")
                    for i in range(min(3, total_records)):
                        print(f"{i+1}. {df.iloc[i].to_dict()}")
                        
            except Exception as e:
                print(f"[ERROR] Failed to read CSV: {e}")
        elif config["expected_ext"] == "*.mp4":
            print(f"[OK] Videos found: {len(files)}")
            print("\nSample files:")
            for i in range(min(3, len(files))):
                print(f"{i+1}. {os.path.basename(files[i])}")
                
        print("\n--------------------------------------------------")

    print("==================================================")
    print("ALL DATASET VERIFICATION COMPLETED")
    print("==================================================")

if __name__ == "__main__":
    verify_modules()
