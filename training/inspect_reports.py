import json
from pathlib import Path

report_dir = Path("training/reports")

for path in sorted(report_dir.glob("*.json")):
    print("\n" + "=" * 65)
    print(path.name)
    print("=" * 65)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2, ensure_ascii=False)[:4500])
    except Exception as exc:
        print("Could not read report:", exc)
