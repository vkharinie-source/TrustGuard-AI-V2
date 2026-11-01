"""
TrustGuard AI - Phase 2 V6 Pre-Cropped Dataset Validation Suite
Validates:
1. 1-to-1 correspondence between V5 source images and V6 crop outputs (exactly 6,000 files).
2. Exact matching split membership (train=4000, val=1000, test=1000) and class labels (50/50 balance).
3. Image readability, strict (224, 224) dimensions, and RGB 3-channel mode for 100% of files.
4. Zero test images moved into train or validation.
5. Manifest consistency and fallback documentation.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
V5_DIR = ROOT_DIR / "training" / "datasets" / "deepfake_v5_multisource"
V6_DIR = ROOT_DIR / "training" / "datasets" / "deepfake_v6_crops"
MANIFEST_PATH = ROOT_DIR / "training" / "reports" / "dataset_manifest_v6_crops.json"

EXPECTED_COUNTS = {
    "train": {"real": 2000, "fake": 2000, "total": 4000},
    "validation": {"real": 500, "fake": 500, "total": 1000},
    "test": {"real": 500, "fake": 500, "total": 1000},
}


def run_validation():
    print("=" * 80)
    print("PHASE 2: VALIDATING V6 PRE-CROPPED DATASET")
    print("=" * 80)

    assert V5_DIR.exists(), f"Source V5 dataset missing: {V5_DIR}"
    assert V6_DIR.exists(), f"Target V6 dataset missing: {V6_DIR}"
    assert MANIFEST_PATH.exists(), f"V6 manifest missing: {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    v5_files = {}
    for p in V5_DIR.rglob("*.jpg"):
        rel_split = p.relative_to(V5_DIR)
        v5_files[str(rel_split)] = p

    v6_files = {}
    for p in V6_DIR.rglob("*.jpg"):
        rel_split = p.relative_to(V6_DIR)
        v6_files[str(rel_split)] = p

    print(f"Total V5 source images: {len(v5_files)}")
    print(f"Total V6 cropped images: {len(v6_files)}")

    # 1. 1-to-1 Correspondence Check
    assert len(v5_files) == 6000, f"Expected 6,000 V5 files, found {len(v5_files)}"
    assert len(v6_files) == 6000, f"Expected 6,000 V6 files, found {len(v6_files)}"
    missing_in_v6 = set(v5_files.keys()) - set(v6_files.keys())
    assert len(missing_in_v6) == 0, f"Files missing in V6: {missing_in_v6}"

    # 2. Split and Class Count Assertions
    actual_counts = defaultdict(lambda: defaultdict(int))
    for rel_path in v6_files.keys():
        parts = Path(rel_path).parts
        split, cls = parts[0], parts[1]
        actual_counts[split][cls] += 1

    for split, expected in EXPECTED_COUNTS.items():
        assert actual_counts[split]["real"] == expected["real"], f"{split} real mismatch: {actual_counts[split]['real']} vs {expected['real']}"
        assert actual_counts[split]["fake"] == expected["fake"], f"{split} fake mismatch: {actual_counts[split]['fake']} vs {expected['fake']}"
        total = actual_counts[split]["real"] + actual_counts[split]["fake"]
        assert total == expected["total"], f"{split} total mismatch: {total} vs {expected['total']}"
    print("[PASS] 1-to-1 correspondence and exact split/class balance verified.")

    # 3. Image Readability, Dimensions & RGB Check
    print("Checking readability, dimensions (224x224), and RGB mode across all 6,000 crops...")
    checked = 0
    for p in v6_files.values():
        checked += 1
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            assert img.size == (224, 224), f"Image {p.name} has invalid size {img.size}, expected (224, 224)"
            assert img.mode == "RGB", f"Image {p.name} has invalid mode {img.mode}, expected RGB"
    print(f"[PASS] {checked} of {checked} crops verified readable, exactly (224, 224), and RGB.")

    # 4. Zero Test Contamination / Relocation Check
    v5_test_names = set(p.name for rel, p in v5_files.items() if rel.startswith("test"))
    v6_train_names = set(p.name for rel, p in v6_files.items() if rel.startswith("train"))
    v6_val_names = set(p.name for rel, p in v6_files.items() if rel.startswith("validation"))

    assert len(v5_test_names & v6_train_names) == 0, "Test image found in train!"
    assert len(v5_test_names & v6_val_names) == 0, "Test image found in validation!"
    print("[PASS] Zero test set relocation or contamination verified.")

    # 5. Manifest Consistency & Fallback Audit
    assert manifest["summary"]["total_images"] == 6000
    assert len(manifest["records"]) == 6000

    det_total = 0
    fb_total = 0
    for r in manifest["records"]:
        if r["crop_status"] == "FACE_DETECTED":
            det_total += 1
            assert r["face_bbox"] is not None
        elif r["crop_status"] == "FALLBACK_CENTRAL_80":
            fb_total += 1
            assert r["face_bbox"] is None
        else:
            raise ValueError(f"Unknown crop_status: {r['crop_status']}")

    print(f"[PASS] Manifest verified: {det_total} FACE_DETECTED ({det_total/6000*100:.1f}%), {fb_total} FALLBACK_CENTRAL_80 ({fb_total/6000*100:.1f}%).")
    print("=" * 80)
    print("ALL PHASE 2 VALIDATION CHECKS PASSED (0 FAILURES)")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
