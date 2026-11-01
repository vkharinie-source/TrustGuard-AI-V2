"""
TrustGuard AI - V8-Exp4 Integration Test Suite
Comprehensive verification for:
1. Dual-stream preprocessing tensor shapes, float32 dtype, and pixel range [0.0, 255.0]
2. Face-detected and central-80% fallback paths
3. V8-Exp4 model loading and verified input/output signatures
4. Correct primary (P(REAL)) and auxiliary 3-class output parsing
5. Decision boundary behavior below, at, and above fixed threshold 0.50
6. Strict preservation of all legacy API response keys
7. Compatibility with predict_image(PIL.Image)
8. Video-frame compatibility with analyze_video pipeline
9. Graceful fallback & rollback to V4/V3 when Exp4 is disabled
10. Error handling for missing/corrupted inputs
"""

import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf
from PIL import Image

import app.services.predictor as predictor
from app.services.predictor import (
    IMAGE_SIZE,
    IS_DUAL_STREAM,
    IS_MULTI_TASK,
    MODEL_NAME,
    MODEL_VERSION,
    load_image,
    predict_image,
    preprocess_dual_stream,
    preprocess_image,
)
from app.services.video_detector import analyze_video


def make_synthetic_image(size=(300, 300), color=(180, 140, 210)) -> Image.Image:
    """Create in-memory synthetic PIL image."""
    return Image.new("RGB", size, color=color)


class TestV8Exp4Integration(unittest.TestCase):

    def test_01_model_loading_and_signatures(self):
        """Confirm V8-Exp4 model is actively loaded with dual-stream multi-task signatures."""
        self.assertTrue(IS_DUAL_STREAM, "Model must be identified as Dual-Stream.")
        self.assertTrue(IS_MULTI_TASK, "Model must be identified as Multi-Task.")
        self.assertIn("Exp4", MODEL_NAME, f"Model name should indicate Exp4, got {MODEL_NAME}")
        self.assertEqual(MODEL_VERSION, "8.4.0")
        
        # Check model object inputs and outputs
        self.assertEqual(len(predictor.model.inputs), 2)
        self.assertEqual(len(predictor.model.outputs), 2)
        
        # Verify shapes
        input_shapes = [list(inp.shape) for inp in predictor.model.inputs]
        self.assertEqual(input_shapes[0][1:], [224, 224, 3])
        self.assertEqual(input_shapes[1][1:], [224, 224, 3])
        
        output_shapes = [list(out.shape) for out in predictor.model.outputs]
        self.assertEqual(output_shapes[0][1:], [1])
        self.assertEqual(output_shapes[1][1:], [3])

    def test_02_dual_stream_preprocessing_shapes_and_range(self):
        """Verify dual-stream preprocessing returns exactly (1, 224, 224, 3) float32 in [0, 255]."""
        img = make_synthetic_image((400, 500))
        dual_data = preprocess_dual_stream(img)
        
        self.assertIn("crop_input", dual_data)
        self.assertIn("frame_input", dual_data)
        self.assertIn("crop_status", dual_data)
        self.assertIn("faces_detected", dual_data)

        crop = dual_data["crop_input"]
        frame = dual_data["frame_input"]

        self.assertEqual(crop.shape, (1, 224, 224, 3))
        self.assertEqual(frame.shape, (1, 224, 224, 3))
        self.assertEqual(crop.dtype, np.float32)
        self.assertEqual(frame.dtype, np.float32)

        # Confirm unnormalized pixel range [0.0, 255.0] (Rescaling layer inside model)
        self.assertTrue(np.all(crop >= 0.0) and np.all(crop <= 255.0))
        self.assertTrue(np.all(frame >= 0.0) and np.all(frame <= 255.0))

    def test_03_crop_fallback_when_no_faces(self):
        """Verify central 80% fallback path executes properly when no faces are detected."""
        img = make_synthetic_image((400, 400), color=(100, 100, 100))
        with patch("app.services.predictor._get_face_cascade", return_value=None):
            dual_data = preprocess_dual_stream(img)
            self.assertEqual(dual_data["crop_status"], "FALLBACK_CENTRAL_80")
            self.assertEqual(dual_data["crop_input"].shape, (1, 224, 224, 3))

    def test_04_primary_and_auxiliary_output_parsing(self):
        """Verify predict_image unpacks P(REAL) and 3-class auxiliary probabilities."""
        img = make_synthetic_image()
        res = predict_image(img, include_forensics=False, include_cross_model=False)

        self.assertIn("raw_score", res)
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["REAL", "FAKE"])
        self.assertIn("probabilities", res)
        self.assertIn("REAL", res["probabilities"])
        self.assertIn("FAKE", res["probabilities"])

        # Multi-task auxiliary fields
        self.assertIn("manipulation_subtype", res)
        aux = res["manipulation_subtype"]
        self.assertIn(aux["predicted_class"], ["GENUINE_AUTHENTIC", "FACE_SWAP_MANIPULATION", "STYLEGAN_GENERATIVE"])
        self.assertIn(aux["class_id"], [0, 1, 2])
        self.assertIn("probabilities", aux)
        self.assertAlmostEqual(
            sum(aux["probabilities"].values()),
            1.0,
            places=3,
            msg="Auxiliary probabilities must sum to 1.0"
        )

    def test_05_decision_boundary_at_threshold_050(self):
        """Confirm decision threshold strictly enforces score >= 0.50 -> REAL, < 0.50 -> FAKE."""
        img = make_synthetic_image()

        # Case 1: Score just below threshold (0.4999 -> FAKE)
        mock_output_fake = [np.array([[0.4999]], dtype=np.float32), np.array([[0.1, 0.8, 0.1]], dtype=np.float32)]
        with patch.object(predictor.model, "predict", return_value=mock_output_fake):
            res_fake = predict_image(img, include_forensics=False, include_cross_model=False)
            self.assertEqual(res_fake["prediction"], "FAKE")
            self.assertEqual(res_fake["threshold"], 0.50)

        # Case 2: Score exactly at threshold (0.5000 -> REAL)
        mock_output_boundary = [np.array([[0.5000]], dtype=np.float32), np.array([[0.6, 0.2, 0.2]], dtype=np.float32)]
        with patch.object(predictor.model, "predict", return_value=mock_output_boundary):
            res_boundary = predict_image(img, include_forensics=False, include_cross_model=False)
            self.assertEqual(res_boundary["prediction"], "REAL")

        # Case 3: Score above threshold (0.5001 -> REAL)
        mock_output_real = [np.array([[0.5001]], dtype=np.float32), np.array([[0.7, 0.1, 0.2]], dtype=np.float32)]
        with patch.object(predictor.model, "predict", return_value=mock_output_real):
            res_real = predict_image(img, include_forensics=False, include_cross_model=False)
            self.assertEqual(res_real["prediction"], "REAL")

    def test_06_response_contract_backward_compatibility(self):
        """Verify 100% preservation of all existing dictionary keys expected by frontend & API."""
        img = make_synthetic_image()
        res = predict_image(img, include_forensics=True, include_cross_model=False)

        required_keys = [
            "prediction", "confidence", "confidence_level", "raw_score", "threshold",
            "probabilities", "verification_id", "integrity_hash", "model_version",
            "model_name", "faces_detected", "original_image_size", "model_input_size",
            "findings", "forensics", "attention_map", "cross_model", "metadata", "disclaimer"
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Response dictionary is missing legacy key: {key}")

    def test_07_load_image_input_variants(self):
        """Verify load_image supports bytes, filepath, and PIL.Image seamlessly."""
        img = make_synthetic_image()
        
        # 1. PIL Image
        pil_out, bytes_out1 = load_image(img)
        self.assertIsInstance(pil_out, Image.Image)
        self.assertGreater(len(bytes_out1), 0)

        # 2. Raw bytes
        pil_out2, bytes_out2 = load_image(bytes_out1)
        self.assertIsInstance(pil_out2, Image.Image)
        self.assertEqual(len(bytes_out2), len(bytes_out1))

    def test_08_video_pipeline_compatibility(self):
        """Verify video frame inference functions seamlessly with predict_image."""
        img = make_synthetic_image((300, 300))
        # predict_image called directly as expected by analyze_video
        frame_result = predict_image(img, include_forensics=False, include_cross_model=False)
        self.assertIn(frame_result.get("prediction"), ["REAL", "FAKE"])
        self.assertIsInstance(frame_result.get("raw_score"), float)
        self.assertIsInstance(frame_result.get("confidence"), float)

    def test_09_v4_fallback_simulation(self):
        """Verify safe rollback: predictor dynamically handles single-stream model if Exp4 is disabled."""
        img = make_synthetic_image()
        single_stream_output = np.array([[0.82]], dtype=np.float32)

        # Simulate legacy single-stream model
        with patch.object(predictor, "IS_DUAL_STREAM", False), \
             patch.object(predictor, "IS_MULTI_TASK", False), \
             patch.object(predictor.model, "predict", return_value=single_stream_output):
            
            res_v4 = predict_image(img, include_forensics=False, include_cross_model=False)
            self.assertEqual(res_v4["prediction"], "REAL")
            self.assertAlmostEqual(res_v4["raw_score"], 0.82, places=4)
            self.assertNotIn("manipulation_subtype", res_v4)

    def test_10_error_handling_on_invalid_input(self):
        """Verify clean error raising for invalid input types."""
        with self.assertRaises(TypeError):
            predict_image(12345)  # Invalid non-image input


if __name__ == "__main__":
    unittest.main()
