import json
from pathlib import Path

for p in sorted(Path("training/reports").glob("*.json")):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        print("\n" + p.name)
        for k, v in d.items():
            if any(s in k.lower() for s in [
                "train", "valid", "val", "test", "accuracy",
                "precision", "recall", "f1", "auc",
                "confusion", "sample", "dataset", "class"
            ]):
                print(f"  {k}: {v}")
    except Exception as e:
        print(p.name, "ERROR:", e)
