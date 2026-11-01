"""
TrustGuard AI - Verification Engine Test Suite
Tests for:
- Image verification endpoint (/api/predict/image, /predict)
- Forensic Error Level Analysis (/api/forensics/ela)
- Video frame analysis endpoint (/api/predict/video)
- Audio authenticity analysis endpoint (/api/predict/audio)
- Transparency dashboard endpoint (/api/transparency/stats)
- Input validation, security restrictions, and error handling
"""

import io
import json
import os
import struct
import unittest
import wave
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

def create_test_image(color=(120, 180, 240), size=(224, 224), format="JPEG"):
    """Helper to generate in-memory synthetic image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def create_test_wav(duration_sec=1.0, sample_rate=16000, freq=440.0):
    """Helper to generate in-memory synthetic PCM WAV audio bytes."""
    import math
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        num_samples = int(duration_sec * sample_rate)
        frames = []
        for i in range(num_samples):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            frames.append(struct.pack("<h", val))
        wav_file.writeframes(b"".join(frames))
    buf.seek(0)
    return buf.getvalue()


class TestVerificationPipeline(unittest.TestCase):

    def test_transparency_stats_endpoint(self):
        """Verify the transparency endpoint returns model and dataset provenance."""
        response = client.get("/api/transparency/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "model_info" in data
        assert "architecture" in data["model_info"]
        assert "transparency_principles" in data
        assert len(data["transparency_principles"]) > 0

    def test_image_prediction_valid_jpg(self):
        """Verify image prediction with valid JPEG."""
        img_bytes = create_test_image(format="JPEG")
        response = client.post(
            "/api/predict/image",
            files={"file": ("test_image.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["prediction"] in ["REAL", "FAKE", "INCONCLUSIVE"]
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 100.0
        assert "forensics" in data
        assert "file_hash_sha256" in data["forensics"]
        assert len(data["forensics"]["file_hash_sha256"]) == 64

    def test_image_prediction_root_alias(self):
        """Verify legacy /predict route returns identical contract."""
        img_bytes = create_test_image(format="PNG")
        response = client.post(
            "/predict",
            files={"file": ("test_image.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["prediction"] in ["REAL", "FAKE", "INCONCLUSIVE"]

    def test_image_prediction_invalid_format(self):
        """Verify rejection of unsupported media formats."""
        bad_bytes = b"This is a text file masquerading as image"
        response = client.post(
            "/api/predict/image",
            files={"file": ("malicious.exe", bad_bytes, "application/octet-stream")}
        )
        assert response.status_code == 400

    def test_image_prediction_empty_file(self):
        """Verify rejection of empty files."""
        response = client.post(
            "/api/predict/image",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400

    def test_forensics_ela_endpoint(self):
        """Verify Error Level Analysis endpoint returns base64 residual map."""
        img_bytes = create_test_image(format="JPEG")
        response = client.post(
            "/api/forensics/ela",
            files={"file": ("sample.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "ela_image_base64" in data
        assert "mean_error" in data
        assert "assessment" in data

    def test_audio_prediction_valid_wav(self):
        """Verify audio voice-cloning endpoint with PCM WAV audio."""
        wav_bytes = create_test_wav(duration_sec=0.5)
        response = client.post(
            "/api/predict/audio",
            files={"file": ("speech_sample.wav", wav_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["prediction"] in ["REAL", "FAKE", "INCONCLUSIVE"]
        assert "features" in data
        assert "spectral_centroid" in data["features"]

    def test_audio_prediction_unsupported_format(self):
        """Verify audio endpoint rejects unpermitted extensions."""
        response = client.post(
            "/api/predict/audio",
            files={"file": ("track.flac", b"dummy flac data", "audio/flac")}
        )
        assert response.status_code == 400

    def test_video_prediction_unsupported_format(self):
        """Verify video endpoint rejects invalid formats."""
        response = client.post(
            "/api/predict/video",
            files={"file": ("test.gif", b"GIF89a...", "image/gif")}
        )
        assert response.status_code == 400
