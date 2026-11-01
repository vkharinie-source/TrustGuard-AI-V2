"""
TrustGuard AI - Website & Phishing URL Detection Test Suite
Comprehensive automated test suite covering all 12 verification criteria:
1. Verified phishing URL fixtures
2. Verified legitimate URL fixtures
3. Unseen-domain examples
4. Correct class-label mapping
5. Correct probability-column selection
6. URL preprocessing and feature ordering
7. Threshold boundaries
8. Invalid and empty URLs
9. Model-loading and inference failures
10. Backend API responses
11. Frontend result consistency
12. Regression tests for existing features
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.main import app
from app.services.website_detector import analyze_website, _get_ml_pipeline
from app.services.url_feature_extractor import URLFeatureExtractor

client = TestClient(app)


class TestWebsiteDetection(unittest.TestCase):

    # ============================================================
    # 1. VERIFIED PHISHING URL FIXTURES
    # ============================================================
    def test_01_verified_phishing_fixtures(self):
        """Verify that known phishing fixtures are classified as FAKE."""
        fixtures = [
            "http://paypal.login-verify-account-security.xyz/signin",
            "http://192.168.1.100:8080/secure-banking-login",
            "http://secure-login-chase-update-account.com",
            "http://portal-netflix-update.com/login.php",
            "http://apple-id-verify.support-device.top/login.html",
            "https://storage.cloud.google.com/prprhrhprc.appspot.com/index.htm#oncall-infra@eqiom.com",
            "https://onedrive.live.com/redir?resid=15D888F2C88E7D68%21104&authkey=%21ALaPI82Fus8uHw8",
            "https://myetherwallet-phishing-scam-site.ru/unlock",
            "https://steamcommunity-trade-offer-login.com/tradeoffer/new",
        ]
        for url in fixtures:
            res = analyze_website(url)
            self.assertEqual(
                res["result"], "FAKE",
                f"Phishing fixture misclassified as {res['result']}: {url}"
            )
            self.assertGreaterEqual(res["risk_score"], 30)
            self.assertGreaterEqual(res["confidence"], 60.0)

    # ============================================================
    # 2. VERIFIED LEGITIMATE URL FIXTURES
    # ============================================================
    def test_02_verified_legitimate_fixtures(self):
        """Verify that known legitimate fixtures are classified as SAFE."""
        fixtures = [
            "https://www.google.com",
            "https://chatgpt.com",
            "https://github.com/fastapi/fastapi",
            "https://en.wikipedia.org/wiki/Phishing",
            "https://stackoverflow.com/questions/509211/understanding-slice-notation",
            "https://docs.python.org/3/library/urllib.parse.html",
        ]
        for url in fixtures:
            res = analyze_website(url)
            self.assertEqual(
                res["result"], "SAFE",
                f"Legitimate fixture misclassified as {res['result']}: {url}"
            )
            self.assertLess(res["risk_score"], 20)
            self.assertGreaterEqual(res["confidence"], 80.0)

    # ============================================================
    # 3. UNSEEN DOMAIN EXAMPLES
    # ============================================================
    def test_03_unseen_domain_examples(self):
        """Verify detector generalizes to novel, previously unseen domains."""
        # Unseen legitimate
        legit_unseen = "https://unseen-academic-research-portal-2026.edu/publications"
        res_legit = analyze_website(legit_unseen)
        self.assertIn(res_legit["result"], ["SAFE", "SUSPICIOUS"])

        # Unseen phishing pattern (brand spoofing on high-risk TLD)
        phish_unseen = "http://secure-binance-kyc-verification.top/auth/login"
        res_phish = analyze_website(phish_unseen)
        self.assertEqual(res_phish["result"], "FAKE")

    # ============================================================
    # 4. CLASS LABEL MAPPING
    # ============================================================
    def test_04_class_label_mapping(self):
        """Verify model class labels map accurately to [0: legitimate, 1: phishing]."""
        pipeline, metadata, err = _get_ml_pipeline()
        self.assertIsNotNone(pipeline, f"Pipeline could not be loaded: {err}")
        self.assertTrue(hasattr(pipeline, "classes_") or hasattr(pipeline.named_steps["classifier"], "classes_"))

        clf = pipeline.named_steps["classifier"] if hasattr(pipeline, "named_steps") else pipeline
        classes = list(clf.classes_)
        self.assertEqual(classes, [0, 1])

    # ============================================================
    # 5. PROBABILITY COLUMN SELECTION
    # ============================================================
    def test_05_probability_column_selection(self):
        """Verify that index 1 corresponds to phishing probability."""
        pipeline, _, _ = _get_ml_pipeline()
        phishing_sample = "http://paypal.login-verify-account-security.xyz/signin"
        legit_sample = "https://www.google.com"

        prob_phish = pipeline.predict_proba([phishing_sample])[0]
        prob_legit = pipeline.predict_proba([legit_sample])[0]

        # In a 2-class classifier [0, 1], index 1 must be high for phishing and low for legitimate
        self.assertGreater(prob_phish[1], 0.90)
        self.assertLess(prob_legit[1], 0.10)
        self.assertAlmostEqual(prob_phish[0] + prob_phish[1], 1.0, places=4)

    # ============================================================
    # 6. URL PREPROCESSING AND FEATURE ORDERING
    # ============================================================
    def test_06_url_preprocessing_and_feature_ordering(self):
        """Verify URLFeatureExtractor produces exactly 32 deterministic features."""
        extractor = URLFeatureExtractor()
        urls = ["https://example.com/test?a=1", "http://192.168.1.1:8080/path"]
        feats = extractor.transform(urls)

        self.assertEqual(feats.shape, (2, 32))
        self.assertEqual(len(extractor.feature_names_), 32)
        # Check specific features
        # 1st URL is https: feature index 4 is is_https
        self.assertEqual(feats[0, 4], 1.0)
        # 2nd URL is http with IP: feature index 5 is is_ip
        self.assertEqual(feats[1, 4], 0.0)
        self.assertEqual(feats[1, 5], 1.0)

    # ============================================================
    # 7. THRESHOLD BOUNDARIES
    # ============================================================
    def test_07_threshold_boundaries(self):
        """Verify threshold boundary classifications."""
        pipeline, metadata, _ = _get_ml_pipeline()
        thresh = float(metadata.get("decision_threshold", 0.46)) if metadata else 0.46
        self.assertGreaterEqual(thresh, 0.40)
        self.assertLessEqual(thresh, 0.60)

        # Clear phishing with elevated risk score must trigger FAKE
        res = analyze_website("http://apple.id-device-verification.click")
        self.assertEqual(res["result"], "FAKE")

    # ============================================================
    # 8. INVALID AND EMPTY URLS
    # ============================================================
    def test_08_invalid_and_empty_urls(self):
        """Verify empty and malformed URLs raise controlled ValueError."""
        with self.assertRaises(ValueError):
            analyze_website("")

        with self.assertRaises(ValueError):
            analyze_website("   ")

        with self.assertRaises(ValueError):
            analyze_website("javascript:alert(1)")

        with self.assertRaises(ValueError):
            analyze_website("data:text/html,<script>alert(1)</script>")

    # ============================================================
    # 9. MODEL LOADING AND INFERENCE FAILURES
    # ============================================================
    def test_09_model_loading_and_inference_failures(self):
        """Verify that when ML pipeline fails, detector returns SUSPICIOUS, never false SAFE."""
        with patch("app.services.website_detector._get_ml_pipeline", return_value=(False, None, "Mocked failure")):
            res = analyze_website("https://some-unknown-unfamiliar-website.com")
            # If the model fails, the system must NOT return SAFE
            self.assertIn(res["result"], ["SUSPICIOUS", "FAKE"])
            self.assertIn("Automated classifier was offline", " ".join(res["details"]))

    # ============================================================
    # 10. BACKEND API RESPONSES
    # ============================================================
    def test_10_backend_api_responses(self):
        """Verify POST /api/analyze/website returns valid JSON schema and correct status."""
        response = client.post(
            "/api/analyze/website",
            json={"url": "https://www.google.com"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("result", data)
        self.assertIn("confidence", data)
        self.assertIn("risk_score", data)
        self.assertIn("details", data)
        self.assertIn("recommendations", data)
        self.assertEqual(data["result"], "SAFE")

        # Phishing URL via API
        phish_response = client.post(
            "/api/analyze/website",
            json={"url": "http://paypal.login-verify-account-security.xyz/signin"}
        )
        self.assertEqual(phish_response.status_code, 200)
        phish_data = phish_response.json()
        self.assertTrue(phish_data.get("success"))
        self.assertEqual(phish_data["result"], "FAKE")

        # Empty URL via API
        bad_response = client.post(
            "/api/analyze/website",
            json={"url": ""}
        )
        self.assertEqual(bad_response.status_code, 400)

    # ============================================================
    # 11. FRONTEND RESULT CONSISTENCY
    # ============================================================
    def test_11_frontend_result_consistency(self):
        """Verify that result and prediction strings always match frontend expected vocabulary."""
        urls = [
            "https://www.google.com",
            "http://192.168.1.100:8080/secure-banking-login",
            "http://unknown-domain-test.xyz"
        ]
        valid_labels = {"SAFE", "SUSPICIOUS", "FAKE"}
        for u in urls:
            res = analyze_website(u)
            self.assertIn(res["result"], valid_labels)
            self.assertEqual(res["result"], res["prediction"])
            self.assertIsInstance(res["confidence"], (int, float))
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 100.0)

    # ============================================================
    # 12. REGRESSION TESTS FOR EXISTING FEATURES
    # ============================================================
    # ============================================================
    # 13. EXACT FALSE NEGATIVE REGRESSION TEST
    # ============================================================
    def test_13_exact_false_negative_regressions(self):
        """Verify that previously failing phishing URLs are strictly classified as FAKE via API."""
        failing_urls = [
            "https://storage.cloud.google.com/prprhrhprc.appspot.com/index.htm#oncall-infra@eqiom.com",
            "https://onedrive.live.com/redir?resid=15D888F2C88E7D68%21104&authkey=%21ALaPI82Fus8uHw8",
            "http://paypal.login-verify-account-security.xyz/signin",
            "http://192.168.1.100:8080/secure-banking-login"
        ]
        for url in failing_urls:
            res = client.post("/api/analyze/website", json={"url": url})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("result"), "FAKE", f"URL {url} must be classified as FAKE, got {data.get('result')}")
            self.assertEqual(data.get("prediction"), "FAKE")
            self.assertGreaterEqual(data.get("confidence", 0.0), 75.0)

    # ============================================================
    # 14. RUNTIME MODEL PATH VERIFICATION
    # ============================================================
    def test_14_runtime_model_path_verification(self):
        """Verify that models/phishing_urls_v5/phishing_urls_pipeline.pkl is loaded at runtime."""
        from app.services.website_detector import get_loaded_model_path
        model_path = get_loaded_model_path()
        self.assertIsNotNone(model_path, "Loaded model path should not be None")
        self.assertTrue(
            "phishing_urls_v5" in model_path.replace("\\", "/"),
            f"Expected V5 model pipeline in path, got: {model_path}"
        )
        self.assertTrue(
            model_path.endswith("phishing_urls_pipeline.pkl"),
            f"Expected pipeline filename phishing_urls_pipeline.pkl, got: {model_path}"
        )

    # ============================================================
    # 15. METRICS AND CHECKS SCHEMA VERIFICATION
    # ============================================================
    def test_15_metrics_and_checks_schema(self):
        """Verify that API response returns all 14 evaluated checks, indicators count, and ML probability."""
        urls = [
            "https://www.google.com",
            "http://paypal.login-verify-account-security.xyz/signin",
            "https://onedrive.live.com/redir?resid=15D888F2C88E7D68%21104&authkey=%21ALaPI82Fus8uHw8"
        ]
        for u in urls:
            res = client.post("/api/analyze/website", json={"url": u})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("checks", data)
            self.assertIsInstance(data["checks"], dict)
            self.assertEqual(len(data["checks"]), 14, "Must evaluate exactly 14 security indicator checks")
            self.assertIn("indicators_evaluated", data)
            self.assertEqual(data["indicators_evaluated"], 14)
            self.assertIn("indicators_flagged", data)
            self.assertIsInstance(data["indicators_flagged"], int)
            self.assertIn("risk_score", data)
            self.assertGreaterEqual(data["risk_score"], 0)
            self.assertLessEqual(data["risk_score"], 100)
            self.assertIn("confidence", data)
            self.assertGreaterEqual(data["confidence"], 0.0)
            self.assertLessEqual(data["confidence"], 100.0)
            self.assertIn("ml_phishing_probability", data)

    # ============================================================
    # 16. INFERENCE FAILURE SAFETY (NEVER DEFAULT TO SAFE)
    # ============================================================
    def test_16_inference_failure_never_silently_safe(self):
        """Verify that if ML inference fails or pipeline is missing, verdict falls back to SUSPICIOUS, never SAFE."""
        from unittest.mock import patch
        import app.services.website_detector as wd

        # Mock _get_ml_pipeline returning (False, {}, "Simulated error")
        with patch.object(wd, "_get_ml_pipeline", return_value=(False, {}, "Simulated error")):
            res = wd.analyze_website("http://some-unverified-site-xyz-123.com/page")
            self.assertNotEqual(res["result"], "SAFE", "Inference failure must never silently return SAFE")
            self.assertEqual(res["result"], "SUSPICIOUS")
            self.assertEqual(res["prediction"], "SUSPICIOUS")
            self.assertTrue(any("Automated classifier was offline" in d or "unavailable" in d for d in res["details"]))

    # ============================================================
    # 17. ZERO RISK SCORE HANDLING FOR UNVERIFIED DOMAINS
    # ============================================================
    def test_17_zero_risk_score_handling(self):
        """Verify that zero risk score distinguishes verified official brands from general safe domains."""
        # Verified brand
        res_brand = client.post("/api/analyze/website", json={"url": "https://www.google.com"}).json()
        self.assertEqual(res_brand["result"], "SAFE")
        self.assertEqual(res_brand["risk_score"], 0)
        self.assertTrue(res_brand["checks"]["verified_official"])

        # Unverified legitimate site
        res_unverified = client.post("/api/analyze/website", json={"url": "https://python.org"}).json()
        self.assertEqual(res_unverified["result"], "SAFE")
        self.assertEqual(res_unverified["indicators_evaluated"], 14)


if __name__ == "__main__":
    unittest.main()


