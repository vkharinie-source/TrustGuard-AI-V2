"""
TrustGuard AI - Multimodal Media Authenticity & DeepFake Prediction Router
Endpoints:
- POST /api/predict/image & /predict: Multimodal image verification (V4 model, ELA, cross-model, calibration)
- POST /api/predict/video: Frame-by-frame video deepfake analysis with timeline
- POST /api/predict/audio: Synthetic speech / voice cloning acoustic analysis
- POST /api/forensics/ela: Standalone Error Level Analysis tampering detection
- GET /api/transparency/stats: Model & dataset audit transparency metrics
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.audio_detector import analyze_audio
from app.services.history_service import save_history
from app.services.image_forensics import perform_error_level_analysis
from app.services.predictor import predict_image
from app.services.video_detector import analyze_video

router = APIRouter(tags=["Prediction"])

ROOT_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT_DIR / "training" / "reports" / "dataset_manifest_v4.json"
REPORT_PATH = ROOT_DIR / "models" / "deepfake_image_v4" / "evaluation_report.json"
V4_REPORT_BACKUP = ROOT_DIR / "training" / "reports" / "deepfake_image_v4_report.json"


# ============================================================
# IMAGE VERIFICATION
# ============================================================

async def handle_image_prediction(file: UploadFile):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/jpg",
    }

    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are supported.",
        )

    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image size exceeds maximum allowed limit of 10 MB.",
        )

    try:
        result = predict_image(image_bytes, include_forensics=True, include_cross_model=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image inference error: {str(e)}",
        )

    # Persist to history with verified category & details
    try:
        save_history(
            filename=file.filename or "uploaded_image.jpg",
            prediction=result["prediction"],
            confidence=result["confidence"],
        )
    except Exception:
        pass

    return {
        "success": True,
        "filename": file.filename,
        **result,
    }


@router.post("/api/predict/image")
async def predict_api_image(file: UploadFile = File(...)):
    return await handle_image_prediction(file)


@router.post("/predict")
async def predict_root(file: UploadFile = File(...)):
    return await handle_image_prediction(file)


# Exported alias for app/main.py compatibility
handle_prediction = handle_image_prediction



# ============================================================
# VIDEO VERIFICATION
# ============================================================

@router.post("/api/predict/video")
async def predict_api_video(file: UploadFile = File(...)):
    allowed_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: MP4, WEBM, MOV, AVI, MKV.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Video exceeds 50 MB size limit.")

    # Save to safe temporary file
    temp_dir = tempfile.mkdtemp(prefix="trustguard_video_")
    temp_path = Path(temp_dir) / (file.filename or "upload.mp4")

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        # Execute video analysis using the calibrated image predictor
        result = analyze_video(temp_path, image_predictor_fn=predict_image, max_frames=12)

        try:
            save_history(
                filename=file.filename or "video_upload.mp4",
                prediction=result["prediction"],
                confidence=result["confidence"],
            )
        except Exception:
            pass

        return {
            "success": True,
            "filename": file.filename,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        # Cleanup temporary file
        try:
            if temp_path.exists():
                temp_path.unlink()
            os.rmdir(temp_dir)
        except Exception:
            pass


# ============================================================
# AUDIO VERIFICATION
# ============================================================

@router.post("/api/predict/audio")
async def predict_api_audio(file: UploadFile = File(...)):
    allowed_exts = {".wav", ".mp3", ".ogg", ".m4a"}
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: WAV, MP3, OGG, M4A.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio exceeds 25 MB size limit.")

    try:
        result = analyze_audio(content, filename=file.filename or "audio_upload.wav")

        try:
            save_history(
                filename=file.filename or "audio_upload.wav",
                prediction=result["prediction"],
                confidence=result["confidence"],
            )
        except Exception:
            pass

        return {
            "success": True,
            "filename": file.filename,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")


# ============================================================
# FORENSICS (Error Level Analysis)
# ============================================================

@router.post("/api/forensics/ela")
async def forensic_ela_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image is empty.")
    result = perform_error_level_analysis(content)
    return {
        "success": True,
        "filename": file.filename,
        **result,
    }


# ============================================================
# TRANSPARENCY & DATASET AUDIT DASHBOARD
# ============================================================

@router.get("/api/transparency/stats")
def get_transparency_stats():
    """
    Returns authentic training, dataset provenance, and validation metrics.
    """
    manifest_data = {}
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    eval_data = {}
    v8_metrics_path = ROOT_DIR / "training" / "reports" / "v8_exp4_validation_metrics.json"
    target_report = v8_metrics_path if v8_metrics_path.exists() else (REPORT_PATH if REPORT_PATH.exists() else V4_REPORT_BACKUP)
    if target_report.exists():
        try:
            with open(target_report, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            pass

    from app.services.predictor import (
        DEFAULT_THRESHOLD,
        IS_DUAL_STREAM,
        IS_MULTI_TASK,
        MODEL_NAME,
        MODEL_VERSION,
    )

    return {
        "success": True,
        "model_info": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "architecture": (
                "Dual-Stream MobileNetV2 (Full Frame + Haar Face Crop) Multi-Task Head"
                if IS_DUAL_STREAM
                else "MobileNetV2 CNN Feature Extractor + Dense ANN Classification Head"
            ),
            "input_resolution": "2x(224x224x3) RGB" if IS_DUAL_STREAM else "224x224x3 RGB",
            "decision_threshold": DEFAULT_THRESHOLD,
            "dual_stream": IS_DUAL_STREAM,
            "multi_task": IS_MULTI_TASK,
            "status": "Active Production Model",
        },
        "dataset_manifest": manifest_data,
        "evaluation_metrics": eval_data,
        "transparency_principles": [
            "No fabricated evaluation scores or synthetic benchmark claims.",
            "Independent untouched test split held out throughout training and threshold selection.",
            "Both class probabilities returned explicitly with confidence bounds.",
            "INCONCLUSIVE status enabled for borderline or ambiguous media uploads.",
        ],
    }
