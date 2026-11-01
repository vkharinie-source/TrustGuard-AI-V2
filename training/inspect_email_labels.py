import pandas as pd
from pathlib import Path

path = Path("training/datasets/email_spam/phishing_email.csv")
df = pd.read_csv(path, low_memory=False)

for label in sorted(df["label"].dropna().unique()):
    group = df[df["label"] == label]
    print("\n" + "=" * 60)
    print(f"LABEL: {label} | Total rows: {len(group)}")
    print("=" * 60)

    for i, text in enumerate(group["text_combined"].dropna().head(5), 1):
        print(f"\nExample {i}:")
        print(str(text)[:700])
