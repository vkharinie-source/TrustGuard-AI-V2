import os
import argparse
import glob
import pandas as pd

def show_samples(module):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(base_dir, "kaggle_datasets", module.lower(), "original")
    
    if not os.path.exists(mod_path):
        print(f"[ERROR] Dataset folder for {module.upper()} not found.")
        return
        
    csv_files = glob.glob(os.path.join(mod_path, "*.csv"))
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in {mod_path}.")
        print("Please ensure the Kaggle dataset is downloaded and extracted here.")
        return
        
    df = pd.read_csv(csv_files[0])
    
    print(f"\n{module.upper()}")
    print("-" * 32)
    
    for i in range(min(10, len(df))):
        print(f"Sample {i+1}")
        row = df.iloc[i].to_dict()
        for k, v in row.items():
            print(f"{k}: {v}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show dataset samples")
    parser.add_argument("--module", type=str, required=True, help="Module name (email, message, url, job_scam, news, call)")
    args = parser.parse_args()
    
    show_samples(args.module)
