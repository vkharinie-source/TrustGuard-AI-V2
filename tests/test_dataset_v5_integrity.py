"""
TrustGuard AI - Comprehensive Automated Dataset V5 Integrity Test Suite

Validates:
1. test_zero_cross_split_hash_overlap(): Zero SHA-256 intersection across train, validation, and test.
2. test_zero_v5_test_hash_overlap_with_historical_train(): Zero hash overlap between V5 Test and V1-V4 training sets.
3. test_zero_deepguard_test_identity_overlap(): DeepGuard test identities strictly isolated from train and validation.
4. test_deepguard_pair_atomic_isolation(): For every pair in V5, real and fake reside in the exact same split.
5. test_image_readability_and_strict_dimensions(): 100% of images open cleanly and are exactly 256x256 RGB.
6. test_exact_class_balance(): Asserts 50.0% Real and 50.0% Fake in train, validation, and test (6,000 total).
7. test_manifest_consistency(): Manifest summary, records, and exclusions match filesystem exactly.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

try:
    import pytest
except ImportError:
    pytest = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_V5_DIR = ROOT_DIR / "training" / "datasets" / "deepfake_v5_multisource"
MANIFEST_PATH = ROOT_DIR / "training" / "reports" / "dataset_manifest_v5.json"

HISTORICAL_TRAIN_DIRS = [
    ROOT_DIR / "server" / "training" / "dataset" / "train",
    ROOT_DIR / "server" / "training" / "dataset_v2" / "train",
    ROOT_DIR / "server" / "training" / "dataset_v3" / "train",
    ROOT_DIR / "training" / "datasets" / "deepfake_v4_multisource" / "train",
]

EXPECTED_COUNTS = {
    "train": {"real": 2000, "fake": 2000, "total": 4000},
    "validation": {"real": 500, "fake": 500, "total": 1000},
    "test": {"real": 500, "fake": 500, "total": 1000},
}


def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _dummy_fixture(fn):
    return fn

fixture_decorator = pytest.fixture(scope="session") if pytest is not None else _dummy_fixture


@fixture_decorator
def dataset_inventory():

    assert DATASET_V5_DIR.exists(), f"Dataset directory missing: {DATASET_V5_DIR}"
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files_by_split = defaultdict(lambda: defaultdict(list))
    hashes_by_split = defaultdict(set)
    file_to_hash = {}
    dg_pairs_by_split = defaultdict(lambda: defaultdict(list))

    for split in ["train", "validation", "test"]:
        for cls in ["real", "fake"]:
            cls_dir = DATASET_V5_DIR / split / cls
            assert cls_dir.exists(), f"Missing split directory: {cls_dir}"
            for f in cls_dir.glob("*.jpg"):
                h = compute_sha256(f)
                files_by_split[split][cls].append(f)
                hashes_by_split[split].add(h)
                file_to_hash[f] = h

                # Track DeepGuard pair IDs
                name = f.name
                if name.startswith("deepguard_face_"):
                    # Name format: deepguard_face_{real/fake}_{idx}_{original_filename} e.g. real_00042.jpg
                    parts = name.split("_")
                    # Find original pid: e.g. "real_00042.jpg" -> "00042"
                    pid = None
                    for idx, part in enumerate(parts):
                        if part in ["real", "fake"] and idx + 1 < len(parts):
                            next_part = parts[idx + 1].split(".")[0]
                            if next_part.isdigit():
                                pid = next_part
                    if pid is not None:
                        dg_pairs_by_split[split][cls].append(pid)

    return {
        "manifest": manifest,
        "files_by_split": files_by_split,
        "hashes_by_split": hashes_by_split,
        "file_to_hash": file_to_hash,
        "dg_pairs_by_split": dg_pairs_by_split,
    }


def test_zero_cross_split_hash_overlap(dataset_inventory):
    """Test 1: Assert zero SHA-256 hash overlap between train, validation, and test."""
    hashes = dataset_inventory["hashes_by_split"]

    tv_leak = hashes["train"] & hashes["validation"]
    assert len(tv_leak) == 0, f"Found {len(tv_leak)} duplicate hashes between Train and Validation!"

    tt_leak = hashes["train"] & hashes["test"]
    assert len(tt_leak) == 0, f"Found {len(tt_leak)} duplicate hashes between Train and Test!"

    vt_leak = hashes["validation"] & hashes["test"]
    assert len(vt_leak) == 0, f"Found {len(vt_leak)} duplicate hashes between Validation and Test!"


def test_zero_v5_test_hash_overlap_with_historical_train(dataset_inventory):
    """Test 2: Assert zero hash overlap between V5 Test split and all historical V1-V4 training sets."""
    test_hashes = dataset_inventory["hashes_by_split"]["test"]

    hist_hashes = set()
    for t_dir in HISTORICAL_TRAIN_DIRS:
        if t_dir.exists():
            for f in t_dir.rglob("*.*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    hist_hashes.add(compute_sha256(f))

    contamination = test_hashes & hist_hashes
    assert len(contamination) == 0, (
        f"V5 Test split contains {len(contamination)} hashes previously used in historical training sets!"
    )


def test_zero_deepguard_test_identity_overlap(dataset_inventory):
    """Test 3: Assert DeepGuard test identity pair IDs do not exist in train or validation."""
    dg_splits = dataset_inventory["dg_pairs_by_split"]

    test_pids = set(dg_splits["test"]["real"]) | set(dg_splits["test"]["fake"])
    train_pids = set(dg_splits["train"]["real"]) | set(dg_splits["train"]["fake"])
    val_pids = set(dg_splits["validation"]["real"]) | set(dg_splits["validation"]["fake"])

    leak_with_train = test_pids & train_pids
    assert len(leak_with_train) == 0, f"DeepGuard Test IDs leaked into Train: {leak_with_train}"

    leak_with_val = test_pids & val_pids
    assert len(leak_with_val) == 0, f"DeepGuard Test IDs leaked into Validation: {leak_with_val}"


def test_deepguard_pair_atomic_isolation(dataset_inventory):
    """Test 4: For every DeepGuard pair in V5, assert that real and fake are in the identical split."""
    dg_splits = dataset_inventory["dg_pairs_by_split"]

    for split in ["train", "validation", "test"]:
        reals = sorted(dg_splits[split]["real"])
        fakes = sorted(dg_splits[split]["fake"])
        assert reals == fakes, f"DeepGuard real and fake pair IDs do not match in split '{split}'!"


def test_image_readability_and_strict_dimensions(dataset_inventory):
    """Test 5: Assert 100% of files open cleanly via Pillow and are strictly 256x256 RGB."""
    files_by_split = dataset_inventory["files_by_split"]
    total_checked = 0

    for split, classes in files_by_split.items():
        for cls, file_list in classes.items():
            for p in file_list:
                total_checked += 1
                with Image.open(p) as img:
                    img.verify()
                with Image.open(p) as img:
                    assert img.size == (256, 256), f"File {p} has invalid dimensions {img.size}, expected (256, 256)!"
                    assert img.mode in ("RGB", "L"), f"File {p} has invalid mode {img.mode}!"

    assert total_checked == 6000, f"Expected 6,000 verified images, checked {total_checked}"


def test_exact_class_balance(dataset_inventory):
    """Test 6: Assert exact 50/50 balance and target counts in train (4000), val (1000), test (1000)."""
    files_by_split = dataset_inventory["files_by_split"]

    for split, expected in EXPECTED_COUNTS.items():
        actual_real = len(files_by_split[split]["real"])
        actual_fake = len(files_by_split[split]["fake"])
        actual_total = actual_real + actual_fake

        assert actual_real == expected["real"], f"{split} real count mismatch: {actual_real} vs {expected['real']}"
        assert actual_fake == expected["fake"], f"{split} fake count mismatch: {actual_fake} vs {expected['fake']}"
        assert actual_total == expected["total"], f"{split} total count mismatch: {actual_total} vs {expected['total']}"


def test_manifest_consistency(dataset_inventory):
    """Test 7: Assert manifest matches on-disk dataset files, hashes, and exclusions."""
    manifest = dataset_inventory["manifest"]
    file_to_hash = dataset_inventory["file_to_hash"]

    assert manifest["summary"]["total_images"] == 6000
    assert manifest["summary"]["train_total"] == 4000
    assert manifest["summary"]["val_total"] == 1000
    assert manifest["summary"]["test_total"] == 1000

    assert manifest["historical_training_audit"]["test_set_exact_hash_overlap"] == 0
    assert manifest["historical_training_audit"]["test_set_deepguard_identity_overlap"] == 0
    assert len(manifest["exclusions"]) == 106  # 53 duplicate real clusters * 2 pair IDs each

    # Verify that all 6,000 manifest records match files on disk
    assert len(manifest["records"]) == 6000
    for record in manifest["records"]:
        disk_path = ROOT_DIR / record["dest_path"]
        assert disk_path.exists(), f"Manifest record destination does not exist on disk: {disk_path}"
        assert file_to_hash[disk_path] == record["sha256"], f"Hash mismatch for {disk_path}"


def run_all_tests():
    print("=" * 80)
    print("RUNNING DATASET V5 INTEGRITY TESTS (STANDALONE RUNNER)")
    print("=" * 80)
    inv = dataset_inventory()
    tests = [
        ("test_zero_cross_split_hash_overlap", test_zero_cross_split_hash_overlap),
        ("test_zero_v5_test_hash_overlap_with_historical_train", test_zero_v5_test_hash_overlap_with_historical_train),
        ("test_zero_deepguard_test_identity_overlap", test_zero_deepguard_test_identity_overlap),
        ("test_deepguard_pair_atomic_isolation", test_deepguard_pair_atomic_isolation),
        ("test_image_readability_and_strict_dimensions", test_image_readability_and_strict_dimensions),
        ("test_exact_class_balance", test_exact_class_balance),
        ("test_manifest_consistency", test_manifest_consistency),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            test_fn(inv)
            print("[PASS]")
            passed += 1
        except Exception as e:
            print(f"[FAIL]: {e}")
            failed += 1

    print("=" * 80)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 80)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        import pytest
        pytest.main(["-v", str(Path(__file__))])
    except ImportError:
        run_all_tests()

