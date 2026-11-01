from pathlib import Path
import re

ROOT = Path(r"D:\project\DeepFake-Detector-AI\client\dist")

files = list(ROOT.glob("assets/index-*.js")) + list(ROOT.glob("assets/index-*.css"))

# Typical characters produced by broken UTF-8 -> Windows-1252 decoding.
MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "Ÿ", "œ", "ž", "™")

pattern = re.compile(
    r"(?:[ÃÂâð][^\x00-\x7F]{1,20})"
)

def score(text):
    return sum(text.count(ch) for ch in MOJIBAKE_MARKERS)

def repair_token(token):
    current = token

    for _ in range(3):
        try:
            candidate = current.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break

        if score(candidate) < score(current):
            current = candidate
        else:
            break

    return current

for path in files:
    text = path.read_text(encoding="utf-8")

    before = score(text)

    repaired = pattern.sub(
        lambda match: repair_token(match.group(0)),
        text
    )

    after = score(repaired)

    path.write_text(
        repaired,
        encoding="utf-8",
        newline=""
    )

    print(f"{path.name}")
    print(f"  mojibake score: {before} -> {after}")

print()
print("========================================")
print("FRONTEND ENCODING REPAIR COMPLETE")
print("========================================")
