"""
Audit all datasets in the repository and generate training/reports/dataset_audit.json
"""
import os
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT_DIR / "training" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

audit_data = {
    "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "repository": "DeepFake-Detector-AI",
    "auditor": "TrustGuard AI Verification Engine",
    "datasets": []
}

# 1. Multi-source V4 Deepfake Benchmark
v4_dir = ROOT_DIR / "training" / "datasets" / "deepfake_v4_multisource"
if v4_dir.exists():
    v4_counts = {}
    for split in ["train", "validation", "test"]:
        v4_counts[split] = {}
        for cls in ["fake", "real"]:
            p = v4_dir / split / cls
            v4_counts[split][cls] = len(list(p.glob("*.*"))) if p.exists() else 0
    
    audit_data["datasets"].append({
        "name": "TrustGuard Multi-Source DeepFake Benchmark V4",
        "path": str(v4_dir.relative_to(ROOT_DIR)),
        "modality": "Image",
        "license": "Research / Educational Permitted",
        "provenance": "NVIDIA Flickr FFHQ (Real), StyleGAN 1-Million Faces by Bojan (Fake), DeepGuard Manipulation Pairs (Fake/Real)",
        "split_counts": v4_counts,
        "total_files": sum(sum(v.values()) for v in v4_counts.values()),
        "class_mapping": {"0": "fake", "1": "real"},
        "status": "active_benchmark_v4"
    })

# 2. Raw Kaggle Manjil Karki 140k
raw_kaggle = ROOT_DIR / "training" / "raw" / "manjilkarki_dataset" / "Dataset"
if raw_kaggle.exists():
    raw_counts = {}
    for cls in ["Fake", "Real"]:
        p = raw_kaggle / cls
        raw_counts[cls.lower()] = len(list(p.glob("*.*"))) if p.exists() else 0
        
    audit_data["datasets"].append({
        "name": "Kaggle manjilkarki/deepfake-and-real-images",
        "path": str(raw_kaggle.relative_to(ROOT_DIR)),
        "modality": "Image",
        "license": "Research / CC BY-SA 4.0",
        "provenance": "140k Real and Fake Faces (NVIDIA FFHQ & StyleGAN Bojan)",
        "split_counts": {"raw": raw_counts},
        "total_files": sum(raw_counts.values()),
        "status": "raw_archive"
    })

# 3. Video deepfake dataset
video_dir = ROOT_DIR / "training" / "dataset" / "video"
if video_dir.exists():
    vid_counts = {}
    for cls in ["fake", "real"]:
        p = video_dir / cls
        if p.exists():
            vid_counts[cls] = len(list(p.glob("*.*")))
    audit_data["datasets"].append({
        "name": "Video DeepFake Benchmark V1",
        "path": str(video_dir.relative_to(ROOT_DIR)),
        "modality": "Video",
        "license": "Research Permitted",
        "provenance": "FaceForensics++ / DeepGuard Temporal Video Clips",
        "split_counts": vid_counts,
        "total_files": sum(vid_counts.values()),
        "status": "active_video"
    })

# 4. Reports from all other modalities
reports = sorted(list(REPORTS_DIR.glob("*_report.json")))
for r in reports:
    try:
        with open(r, "r", encoding="utf-8") as f:
            data = json.load(f)
            audit_data["datasets"].append({
                "report_file": str(r.relative_to(ROOT_DIR)),
                "dataset_name": data.get("dataset_name", data.get("dataset", r.stem)),
                "samples_evaluated": data.get("test_samples_total", data.get("total_samples", "N/A")),
                "status": "verified_via_benchmark"
            })
    except Exception:
        pass

out_file = REPORTS_DIR / "dataset_audit.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2)

print(f"Successfully generated {out_file} with {len(audit_data['datasets'])} audited datasets.")
