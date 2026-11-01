import os
import joblib
import re

class MLUrlService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(base_dir, "models", "url", "url_model.pkl")
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            print("URL ML Model not found. Please train the model using Kaggle dataset first.")

    def extract_features(self, url: str) -> list:
        # Example feature extraction matching the training template
        return [
            len(url),                             # url_length
            1 if url.startswith("https") else 0,  # has_https
            url.count("."),                       # num_dots
            url.count(".") - 1 if url.count(".") > 0 else 0, # num_subdomains (approx)
            len(re.findall(r'[^a-zA-Z0-9]', url)), # num_special_chars
            1 if "@" in url else 0,               # has_at_symbol
            1 if "-" in url else 0,               # has_hyphen
            1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0, # has_ip
            url.count("=") + url.count("&")       # num_query_params (approx)
        ]

    def analyze(self, url: str) -> dict:
        if not self.model:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "details": ["ML Model not trained. Please download Kaggle dataset and run training script."]
            }
        
        features = [self.extract_features(url)]
        prediction = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0]
        
        confidence = max(prob) * 100
        result = "PHISHING/MALICIOUS" if prediction == 1 else "LEGITIMATE"

        return {
            "result": result,
            "confidence": round(confidence, 2),
            "details": ["Prediction provided by ML URL Classification Model (Kaggle Dataset)."]
        }
