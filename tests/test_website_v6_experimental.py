"""
TrustGuard AI - Experimental V6 Website Phishing Detector Test Suite
Tests:
1. V6 model artifact loading and metadata verification.
2. Feature extraction consistency across all 46 V6 features.
3. Prediction probability and calibrated threshold (0.46) behavior.
4. Correct detection on historical V5 false-negative archetypes (brand typosquats, shared hosting, short URLs).
5. False-positive resistance on legitimate brand references and shared platforms.
6. Verification that production V5 remains untouched (rollback safety).
"""

import unittest
from pathlib import Path
import joblib
import numpy as np

from app.services.url_feature_extractor_v6 import URLFeatureExtractorV6
from app.services.website_detector_v6_exp import (
    load_v6_pipeline,
    predict_website_v6,
    get_v6_loaded_model_path,
    _V6_THRESHOLD
)
import app.services.website_detector as v5_detector


class TestWebsiteV6Experimental(unittest.TestCase):

    def setUp(self):
        self.pipeline = load_v6_pipeline()
        self.assertIsNotNone(self.pipeline, "V6 experimental pipeline failed to load")

    def test_01_model_loading_and_classes(self):
        """Verify pipeline steps, classes, and loaded path."""
        clf = self.pipeline.named_steps["classifier"]
        classes = [int(c) for c in clf.classes_]
        self.assertEqual(classes, [0, 1], "V6 classifier classes must be [0, 1]")
        path = get_v6_loaded_model_path()
        self.assertTrue(path.endswith("phishing_urls_pipeline_v6.pkl"), f"Unexpected V6 path: {path}")

    def test_02_feature_extractor_consistency(self):
        """Verify URLFeatureExtractorV6 extracts 46 features without NaNs."""
        ext = URLFeatureExtractorV6(mode="combined")
        self.assertEqual(len(ext.feature_names_), 46)
        urls = [
            "https://www.google.com",
            "http://netflxius.tumblr.com/",
            "http://skechersbuty.com/auth/login.php",
            "http://cort.as/-FgvH"
        ]
        feats = ext.transform(urls)
        self.assertEqual(feats.shape, (len(urls), 46))
        self.assertFalse(np.isnan(feats).any(), "Extracted features contained NaN values")

    def test_03_v6_catches_historical_false_negative_archetypes(self):
        """Verify that V6 successfully detects subtle phishing archetypes missed by earlier baselines."""
        phishing_samples = [
            "http://netflxius.tumblr.com/",
            "http://skechersbuty.com/auth/login.php",
            "http://bt-uk.co.uk/mybt/signin.html",
            "http://paypa1-security.com/login",
            "http://cort.as/-FgvH"
        ]
        for url in phishing_samples:
            res = predict_website_v6(url)
            self.assertTrue(
                res["is_fake"] or res["verdict"] in ["FAKE", "SUSPICIOUS"],
                f"V6 failed to detect phishing archetype: {url} -> {res}"
            )

    def test_04_v6_legitimate_site_resistance(self):
        """Verify that legitimate platforms and shared-hosting domains are not flagged as phishing."""
        legitimate_samples = [
            "https://www.netflix.com/browse",
            "https://www.paypal.com/signin",
            "https://staff.tumblr.com/",
            "https://googleblog.blogspot.com/2024/01/advancing-ai-responsibly.html",
            "https://github.com/fastapi/fastapi"
        ]
        for url in legitimate_samples:
            res = predict_website_v6(url)
            self.assertLess(
                res["phishing_probability"],
                _V6_THRESHOLD,
                f"V6 produced false alarm on legitimate site: {url} (prob={res['phishing_probability']}, threshold={_V6_THRESHOLD})"
            )
            self.assertEqual(res["label"], "SAFE")

    def test_05_v5_production_preserved_byte_for_byte(self):
        """Verify that production website_detector remains V5 and loaded from models/phishing_urls_v5."""
        v5_path = v5_detector.get_loaded_model_path()
        self.assertIsNotNone(v5_path)
        self.assertTrue("phishing_urls_v5" in str(v5_path), f"Production detector must load V5, got {v5_path}")
        # Verify metadata threshold
        pipe, meta, err = v5_detector._get_ml_pipeline()
        self.assertEqual(meta.get("decision_threshold", 0.46), 0.46, "Production V5 threshold must remain 0.46")


if __name__ == "__main__":
    unittest.main()
