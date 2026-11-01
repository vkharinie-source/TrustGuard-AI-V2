"""
TrustGuard AI - Cross-Model Verification Service
Executes comparative inference across multiple independent model checkpoints
(V4, V3, V2) to detect consensus or model divergence.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]

V4_MODEL_PATH = ROOT_DIR / "models" / "deepfake_image_v4" / "deepfake_mobilenetv4_best.keras"
if not V4_MODEL_PATH.exists():
    V4_MODEL_PATH = ROOT_DIR / "model" / "deepfake_mobilenetv2_v4_best.keras"

V3_MODEL_PATH = ROOT_DIR / "model" / "deepfake_mobilenetv2_v3_best.keras"
V2_MODEL_PATH = ROOT_DIR / "model" / "deepfake_mobilenetv2_v2_best.keras"

_loaded_models: Dict[str, Any] = {}


def _get_model(name: str, path: Path):
    if name not in _loaded_models:
        if path.exists():
            try:
                _loaded_models[name] = tf.keras.models.load_model(path, compile=False)
            except Exception:
                _loaded_models[name] = None
        else:
            _loaded_models[name] = None
    return _loaded_models[name]


def compare_models(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Run the same image across MobileNetV2 V4, V3, and V2.
    Evaluates consensus, highlights disagreements, and identifies uncertainty.
    """
    img_resized = pil_image.resize((224, 224))
    img_arr = np.expand_dims(np.asarray(img_resized, dtype=np.float32), axis=0)

    results: Dict[str, Any] = {}
    predictions_list = []

    # 1. Evaluate V4 (Primary Calibrated Model)
    m_v4 = _get_model("v4", V4_MODEL_PATH)
    if m_v4 is not None:
        try:
            score_v4 = float(m_v4.predict(img_arr, verbose=0)[0][0])
            label_v4 = "REAL" if score_v4 >= 0.50 else "FAKE"
            conf_v4 = round(score_v4 * 100.0 if label_v4 == "REAL" else (1.0 - score_v4) * 100.0, 2)
            results["v4_primary"] = {
                "version": "MobileNetV2 V4 (Calibrated)",
                "raw_score": round(score_v4, 4),
                "decision_threshold": 0.50,
                "prediction": label_v4,
                "confidence": conf_v4,
            }
            predictions_list.append(label_v4)
        except Exception as e:
            results["v4_primary"] = {"error": str(e)}

    # 2. Evaluate V3 (Legacy High-Threshold Model)
    m_v3 = _get_model("v3", V3_MODEL_PATH)
    if m_v3 is not None:
        try:
            score_v3 = float(m_v3.predict(img_arr, verbose=0)[0][0])
            label_v3 = "REAL" if score_v3 >= 0.9941 else "FAKE"
            # Calibrate confidence relative to high-precision decision threshold 0.9941
            if label_v3 == "FAKE":
                conf_v3 = round(min(98.5, max(75.0, 85.0 + (0.9941 - score_v3) * 500.0)), 2)
            else:
                conf_v3 = round(min(99.9, max(85.0, 90.0 + (score_v3 - 0.9941) / max(1.0 - 0.9941, 1e-6) * 9.9)), 2)
            results["v3_legacy"] = {
                "version": "MobileNetV2 V3 (Face-Swap Specialist)",
                "raw_score": round(score_v3, 6),
                "decision_threshold": 0.9941,
                "prediction": label_v3,
                "confidence": conf_v3,
            }
            predictions_list.append(label_v3)
        except Exception as e:
            results["v3_legacy"] = {"error": str(e)}

    # 3. Evaluate V2
    m_v2 = _get_model("v2", V2_MODEL_PATH)
    if m_v2 is not None:
        try:
            score_v2 = float(m_v2.predict(img_arr, verbose=0)[0][0])
            label_v2 = "REAL" if score_v2 >= 0.50 else "FAKE"
            conf_v2 = round(score_v2 * 100.0 if label_v2 == "REAL" else (1.0 - score_v2) * 100.0, 2)
            results["v2_baseline"] = {
                "version": "MobileNetV2 V2 (Baseline)",
                "raw_score": round(score_v2, 4),
                "decision_threshold": 0.50,
                "prediction": label_v2,
                "confidence": conf_v2,
            }
            predictions_list.append(label_v2)
        except Exception as e:
            results["v2_baseline"] = {"error": str(e)}

    # Determine consensus
    unique_preds = set(predictions_list)
    if len(unique_preds) == 1:
        consensus = "FULL_CONSENSUS"
        consensus_desc = f"All active models agreed on {list(unique_preds)[0]}."
        has_divergence = False
    elif len(unique_preds) > 1:
        consensus = "DIVERGENT"
        consensus_desc = "Models produced conflicting assessments. Recommend manual or forensic review."
        has_divergence = True
    else:
        consensus = "UNAVAILABLE"
        consensus_desc = "Multi-model evaluation unavailable."
        has_divergence = False

    return {
        "consensus_status": consensus,
        "consensus_description": consensus_desc,
        "has_divergence": has_divergence,
        "models": results,
    }
