"""
TrustGuard AI - Staging / Canary Integration Tests
Module: tests.test_website_staging_canary

Verifies:
1. Default flag state (ENABLE_WEBSITE_DETECTOR_V6 unset or 'false') routes to V5 production.
2. Staging flag state (ENABLE_WEBSITE_DETECTOR_V6='true') routes to V6 experimental candidate.
3. API response dictionary schema parity between V5 and V6 (14 security checks, matching fields).
4. Cloud storage hostname handling: storage.cloud.google.com is never whitelisted as authentic Google.
5. Deceptive subdomains and lookalike typosquats are properly blocked.
6. Malformed and unsafe protocol schemes (javascript:, data:) are rejected with HTTP 400.
7. Reversion/rollback to V5 is instant when flag is disabled.
"""

import os
import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.routes.website import is_v6_enabled


class TestWebsiteStagingCanary(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Ensure clean environment for each test
        if "ENABLE_WEBSITE_DETECTOR_V6" in os.environ:
            del os.environ["ENABLE_WEBSITE_DETECTOR_V6"]

    def tearDown(self):
        if "ENABLE_WEBSITE_DETECTOR_V6" in os.environ:
            del os.environ["ENABLE_WEBSITE_DETECTOR_V6"]

    def test_01_default_routing_uses_v5_production(self):
        """When ENABLE_WEBSITE_DETECTOR_V6 is unset, requests must use V5 production."""
        self.assertFalse(is_v6_enabled(), "Default flag must be False")
        resp = self.client.post("/api/analyze/website", json={"url": "https://www.google.com"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("phishing_urls_v5", data["model_loaded"], f"Expected V5 model, got: {data['model_loaded']}")
        self.assertEqual(data["result"], "SAFE")

    def test_02_canary_routing_uses_v6_when_enabled(self):
        """When ENABLE_WEBSITE_DETECTOR_V6 is 'true', requests use V6 experimental candidate."""
        os.environ["ENABLE_WEBSITE_DETECTOR_V6"] = "true"
        self.assertTrue(is_v6_enabled(), "Flag must evaluate to True")
        resp = self.client.post("/api/analyze/website", json={"url": "https://www.google.com"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("phishing_urls_v6_exp", data["model_loaded"], f"Expected V6 model, got: {data['model_loaded']}")
        self.assertEqual(data["result"], "SAFE")

    def test_03_schema_parity_between_v5_and_v6(self):
        """Verify identical dictionary keys and 14 checks between V5 and V6 API responses."""
        test_url = "https://www.wikipedia.org"

        # 1. Fetch V5 response
        os.environ["ENABLE_WEBSITE_DETECTOR_V6"] = "false"
        v5_resp = self.client.post("/api/analyze/website", json={"url": test_url}).json()

        # 2. Fetch V6 response
        os.environ["ENABLE_WEBSITE_DETECTOR_V6"] = "true"
        v6_resp = self.client.post("/api/analyze/website", json={"url": test_url}).json()

        # Check top-level keys
        required_keys = [
            "success", "result", "prediction", "confidence", "risk_score",
            "details", "recommendations", "url", "domain", "protocol",
            "checks", "indicators_evaluated", "indicators_flagged",
            "ml_phishing_probability", "model_loaded"
        ]
        for key in required_keys:
            self.assertIn(key, v5_resp, f"V5 response missing key: {key}")
            self.assertIn(key, v6_resp, f"V6 response missing key: {key}")

        # Check 14 security checks
        required_checks = [
            "https_encryption", "domain_format", "trusted_tld", "url_shortener",
            "security_keywords", "subdomain_depth", "url_length", "at_symbol",
            "path_redirection", "brand_integrity", "query_parameters",
            "hosting_platform", "verified_official", "ml_classifier"
        ]
        for chk in required_checks:
            self.assertIn(chk, v5_resp["checks"], f"V5 checks missing: {chk}")
            self.assertIn(chk, v6_resp["checks"], f"V6 checks missing: {chk}")
        self.assertEqual(len(v6_resp["checks"]), 14, "V6 must have exactly 14 security checks")

    def test_04_storage_cloud_google_com_handling(self):
        """Verify storage.cloud.google.com is treated as shared hosting and not whitelisted in both models."""
        target_url = "https://storage.cloud.google.com/malicious-phish-bucket/login.html"

        for flag_val in ["false", "true"]:
            os.environ["ENABLE_WEBSITE_DETECTOR_V6"] = flag_val
            resp = self.client.post("/api/analyze/website", json={"url": target_url}).json()
            self.assertEqual(resp["result"], "FAKE", f"storage.cloud.google.com must be FAKE with flag={flag_val}")
            self.assertFalse(resp["checks"]["hosting_platform"], "hosting_platform check must be False (flagged)")
            self.assertFalse(resp["checks"]["verified_official"], "verified_official check must be False (not whitelisted)")

    def test_05_lookalike_and_deceptive_subdomains(self):
        """Verify lookalikes and deceptive subdomains are caught under both V5 and V6."""
        fixtures = [
            "http://paypa1.com/login",
            "http://paypal.com.attacker-domain.xyz/auth",
            "http://skechersbuty.com/auth/login.php"
        ]
        for flag_val in ["false", "true"]:
            os.environ["ENABLE_WEBSITE_DETECTOR_V6"] = flag_val
            for url in fixtures:
                resp = self.client.post("/api/analyze/website", json={"url": url}).json()
                self.assertIn(resp["result"], ["FAKE", "SUSPICIOUS"], f"Failed to detect {url} with flag={flag_val}")

    def test_06_malformed_urls_and_unsafe_schemes(self):
        """Verify malformed and unsafe URL protocol schemes are rejected with HTTP 400."""
        invalid_inputs = [
            "",
            "   ",
            "javascript:alert('xss')",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:MsgBox(1)"
        ]
        for bad in invalid_inputs:
            resp = self.client.post("/api/analyze/website", json={"url": bad})
            self.assertEqual(resp.status_code, 400, f"Expected HTTP 400 for input: '{bad}'")

    def test_07_rollback_safety_instant_reversion(self):
        """Verify that disabling the canary flag instantly reverts API traffic to V5."""
        # 1. Enable V6
        os.environ["ENABLE_WEBSITE_DETECTOR_V6"] = "true"
        r1 = self.client.post("/api/analyze/website", json={"url": "https://www.netflix.com"}).json()
        self.assertIn("phishing_urls_v6_exp", r1["model_loaded"])

        # 2. Disable flag
        os.environ["ENABLE_WEBSITE_DETECTOR_V6"] = "false"
        r2 = self.client.post("/api/analyze/website", json={"url": "https://www.netflix.com"}).json()
        self.assertIn("phishing_urls_v5", r2["model_loaded"], "Must immediately revert to V5")


if __name__ == "__main__":
    unittest.main()
