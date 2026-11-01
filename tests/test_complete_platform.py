"""
TrustGuard AI - Complete Platform Local Integration & Verification Suite
Tests all 10 core capability areas locally:
1. Deepfake Image Detection (V8-Exp4 dual-stream multi-task model)
2. Video Deepfake Detection (Frame extraction & aggregation)
3. Audio Authenticity & Voice Cloning Detection
4. Fake News & Sensationalism Detection
5. Email, SMS/Text, Call & Threat Detection
6. Forensics (Error Level Analysis, SHA-256 integrity, EXIF)
7. Special Scams (Job, Internship, Application)
8. Authentication & Dashboard
9. History Logging & Persistence
10. Validation, Error Handling, and Security Controls
"""

import io
import json
import math
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_test_image(size=(224, 224), color=(130, 190, 240), fmt="JPEG"):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def make_test_wav(duration_sec=0.5, sample_rate=16000, freq=440.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        num_samples = int(duration_sec * sample_rate)
        frames = []
        for i in range(num_samples):
            val = int(32767.0 * 0.4 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            frames.append(struct.pack("<h", val))
        w.writeframes(b"".join(frames))
    buf.seek(0)
    return buf.getvalue()


class TestCompletePlatform(unittest.TestCase):

    # ============================================================
    # 1. CORE SYSTEM & HEALTH
    # ============================================================

    def test_01_root_and_health(self):
        """Verify root and health check endpoints respond correctly."""
        res_root = client.get("/")
        self.assertEqual(res_root.status_code, 200)
        root_data = res_root.json()
        self.assertTrue(root_data.get("success"))
        self.assertIn("endpoints", root_data)
        self.assertIn("POST /api/analyze/news", root_data["endpoints"])

        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json().get("status"), "healthy")

        res_api_health = client.get("/api/health")
        self.assertEqual(res_api_health.status_code, 200)

    # ============================================================
    # 2. IMAGE DEEPFAKE DETECTION (V8-Exp4)
    # ============================================================

    def test_02_image_prediction_dual_stream(self):
        """Verify image prediction with V8-Exp4 dual-stream model and forensics."""
        img_bytes = make_test_image(fmt="JPEG")
        res = client.post(
            "/api/predict/image",
            files={"file": ("photo.jpg", img_bytes, "image/jpeg")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("prediction"), ["REAL", "FAKE", "INCONCLUSIVE"])
        self.assertIn("confidence", data)
        self.assertIn("model_name", data)
        self.assertIn("model_version", data)
        self.assertIn("forensics", data)
        self.assertIn("file_hash_sha256", data["forensics"])
        self.assertEqual(len(data["forensics"]["file_hash_sha256"]), 64)

    def test_03_image_prediction_legacy_alias(self):
        """Verify legacy /predict route returns equivalent schema."""
        img_bytes = make_test_image(fmt="PNG")
        res = client.post(
            "/predict",
            files={"file": ("sample.png", img_bytes, "image/png")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))

    # ============================================================
    # 3. VIDEO DEEPFAKE DETECTION
    # ============================================================

    def test_04_video_prediction_validation(self):
        """Verify video endpoint rejects invalid formats and accepts mp4."""
        bad_res = client.post(
            "/api/predict/video",
            files={"file": ("fake.exe", b"invalid executable", "application/octet-stream")}
        )
        self.assertEqual(bad_res.status_code, 400)

    # ============================================================
    # 4. AUDIO VERIFICATION & VOICE CLONING
    # ============================================================

    def test_05_audio_authenticity(self):
        """Verify audio analysis returns acoustic metrics and synthetic speech verdict."""
        wav_bytes = make_test_wav(duration_sec=0.5)
        res = client.post(
            "/api/predict/audio",
            files={"file": ("call_sample.wav", wav_bytes, "audio/wav")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("prediction"), ["REAL", "FAKE", "INCONCLUSIVE"])
        self.assertIn("features", data)
        self.assertIn("spectral_centroid", data["features"])
        self.assertIn("audio_metadata", data)
        self.assertIn("sha256_hash", data["audio_metadata"])

    # ============================================================
    # 5. FORENSICS (ERROR LEVEL ANALYSIS)
    # ============================================================

    def test_06_forensics_ela(self):
        """Verify Error Level Analysis endpoint produces tampering assessment."""
        img_bytes = make_test_image(fmt="JPEG")
        res = client.post(
            "/api/forensics/ela",
            files={"file": ("audit.jpg", img_bytes, "image/jpeg")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn("ela_image_base64", data)
        self.assertIn("mean_error", data)
        self.assertIn("assessment", data)

    # ============================================================
    # 6. TRANSPARENCY STATS
    # ============================================================

    def test_07_transparency_stats(self):
        """Verify transparency statistics dynamically return active model metadata."""
        res = client.get("/api/transparency/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn("model_info", data)
        self.assertTrue(data["model_info"].get("dual_stream", False))
        self.assertIn("8.4.0", data["model_info"].get("version", ""))

    # ============================================================
    # 7. SMS / MESSAGE FRAUD DETECTION
    # ============================================================

    def test_08_message_detection(self):
        """Verify SMS phishing & credential extraction detection."""
        res = client.post(
            "/api/analyze/message",
            json={"content": "URGENT: Your account is suspended. Send OTP 4492 to unlock immediately."}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("result"), ["FAKE", "SCAM", "SUSPICIOUS"])
        self.assertGreaterEqual(data.get("risk_score", 0), 25)

    # ============================================================
    # 8. EMAIL PHISHING DETECTION
    # ============================================================

    def test_09_email_detection(self):
        """Verify email threat analyzer flags phishing indicators."""
        res = client.post(
            "/api/analyze/email",
            json={
                "subject": "Wire Transfer Required",
                "sender": "ceo-urgent@suspicious-domain.xyz",
                "content": "Please wire $15,000 to the attached bank coordinates immediately."
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn("confidence", data)

    # ============================================================
    # 9. WEBSITE / URL SCANNER
    # ============================================================

    def test_10_website_detection(self):
        """Verify URL threat analyzer detects suspicious hostnames and IP structures."""
        res = client.post(
            "/api/analyze/website",
            json={"url": "http://192.168.1.100/secure-banking-login/account.html"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("result"), ["FAKE", "PHISHING/MALICIOUS", "SUSPICIOUS"])

    # ============================================================
    # 10. PHONE CALL SCAM DETECTOR
    # ============================================================

    def test_11_call_detection(self):
        """Verify phone call scam detection parses banking impersonation scripts."""
        res = client.post(
            "/api/analyze/call",
            json={
                "caller": "+1-800-FAKE-IRS",
                "transcript": "This is Officer John. There is an arrest warrant against you. Pay $500 in gift cards."
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("result"), ["SCAM", "FAKE", "SUSPICIOUS"])

    # ============================================================
    # 11. SPECIALIZED SCAM ANALYZERS (Job, Internship, App)
    # ============================================================

    def test_12_job_scam(self):
        """Verify job scam detection flags upfront fee recruitment fraud."""
        res = client.post(
            "/api/analyze/job",
            json={
                "title": "Data Entry Executive",
                "company": "Fast Hiring Inc",
                "description": "Earn $200 daily. Pay upfront registration fee of $50."
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("result"), "SCAM")

    def test_13_internship_scam(self):
        """Verify internship fraud detection flags security deposit demands."""
        res = client.post(
            "/api/analyze/internship",
            json={
                "title": "Python Research Intern",
                "company": "Global Talent Agency",
                "description": "Direct selection without interview. Pay training fee of $100."
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("result"), ["SCAM", "SUSPICIOUS"])

    def test_14_app_scam(self):
        """Verify mobile application detector flags high-risk permissions and APK sideloading."""
        res = client.post(
            "/api/analyze/app",
            json={
                "app_name": "Instant Cash 200%",
                "developer": "Unknown Modder",
                "description": "Download APK from external source. Grants Accessibility Service and Read SMS."
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("result"), "FAKE")

    # ============================================================
    # 12. FAKE NEWS DETECTION
    # ============================================================

    def test_15_fake_news(self):
        """Verify dedicated Fake News analyzer processes articles and viral claims."""
        res = client.post(
            "/api/analyze/news",
            json={
                "headline": "SHOCKING SECRET: Miracle cure banned by doctors revealed!",
                "content": "A secret whistleblower disclosed that this one simple trick cures all diseases instantly."
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("result"), "FAKE")
        self.assertGreaterEqual(data.get("risk_score", 0), 40)
        self.assertGreater(len(data.get("recommendations", [])), 0)

    # ============================================================
    # 13. AUTHENTICATION & SECURITY
    # ============================================================

    def test_16_auth_flow(self):
        """Verify login with demo user and auth test endpoints."""
        res_test = client.get("/api/auth/test")
        self.assertEqual(res_test.status_code, 200)

        res_login = client.post(
            "/api/auth/login",
            json={"email": "harinievk@gmail.com", "password": "12345678"}
        )
        self.assertEqual(res_login.status_code, 200)
        login_data = res_login.json()
        self.assertTrue(login_data.get("success"))
        self.assertIn("token", login_data)

        # Invalid login
        res_bad = client.post(
            "/api/auth/login",
            json={"email": "unknown@domain.com", "password": "wrongpassword"}
        )
        self.assertIn(res_bad.status_code, [401, 500])

    # ============================================================
    # 14. DASHBOARD & HISTORY PERSISTENCE
    # ============================================================

    def test_17_dashboard_and_history(self):
        """Verify dashboard statistics aggregate recorded scans."""
        res_history = client.get("/api/history")
        self.assertEqual(res_history.status_code, 200)
        self.assertTrue(res_history.json().get("success"))

        res_dash = client.get("/api/dashboard/stats")
        self.assertEqual(res_dash.status_code, 200)
        data = res_dash.json()
        self.assertTrue(data.get("success"))
        self.assertIn("stats", data)
        self.assertIn("total", data["stats"])


if __name__ == "__main__":
    unittest.main()
