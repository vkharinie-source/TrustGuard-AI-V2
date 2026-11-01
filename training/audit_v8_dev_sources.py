import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest_path = root / "training/reports/dataset_manifest_v8_dev.json"

data = json.loads(manifest_path.read_text(encoding="utf-8"))
records = data["records"]

print("\n=== V8 DEVELOPMENT DATASET AUDIT ===")
print("Total records:", len(records))

for split in ("train", "validation"):
    rows = [r for r in records if r.get("split") == split]
    print(f"\n{split.upper()} records:", len(rows))
    print("Sources:")
    for source, count in sorted(
        Counter(r.get("source_type", "UNKNOWN") for r in rows).items()
    ):
        print(f"  {source}: {count}")

    print("Labels:")
    for label, count in sorted(
        Counter(str(r.get("binary_label", "UNKNOWN")) for r in rows).items()
    ):
        print(f"  label {label}: {count}")

    print("Class/source combinations:")
    for key, count in sorted(
        Counter(
            (r.get("source_type", "UNKNOWN"),
             str(r.get("binary_label", "UNKNOWN")))
            for r in rows
        ).items()
    ):
        print(f"  {key[0]} | label={key[1]}: {count}")

print("\nAudit complete. No training or model files changed.")
