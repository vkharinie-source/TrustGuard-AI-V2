from pathlib import Path
import re

ROOT = Path(r"D:\project\DeepFake-Detector-AI\client\dist")

MARKERS = (
    "Ã", "Â", "â", "ð", "Ÿ",
    "œ", "ž", "™", "š", "�"
)

def badness(text):
    return sum(text.count(x) for x in MARKERS)

def repair_run(run):
    current = run

    for _ in range(8):
        before = badness(current)

        try:
            candidate = current.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break

        after = badness(candidate)

        if after < before:
            current = candidate
        else:
            break

    return current

def repair_file(path):
    text = path.read_text(encoding="utf-8")
    before = badness(text)

    # Repair only non-ASCII runs.
    # Proper Unicode that cannot be represented as cp1252 is left untouched.
    repaired = re.sub(
        r"[^\x00-\x7F]+",
        lambda m: repair_run(m.group(0)),
        text
    )

    after = badness(repaired)

    path.write_text(
        repaired,
        encoding="utf-8",
        newline=""
    )

    print(f"{path.name}: {before} -> {after}")

for path in ROOT.glob("assets/*.js"):
    repair_file(path)

print()
print("==============================================")
print("GLOBAL FRONTEND ENCODING REPAIR COMPLETE")
print("==============================================")
