import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[2]
HISTORY_FILE = BASE_DIR / "history.json"


def get_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(filename, prediction, confidence, category=None, **kwargs):
    history = get_history()

    item = {
        "id": len(history) + 1,
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }

    history.insert(0, item)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return item


def clear_history():
    """Clear all records from history.json."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump([], file, indent=2)
    return {"message": "History cleared successfully"}

