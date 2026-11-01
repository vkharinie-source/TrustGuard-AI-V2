import os
from pathlib import Path
import joblib

class MLEmailService:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[2]
        self.pipeline_path = base_dir / "models" / "email_spam_training_v3" / "email_spam_pipeline.pkl"
        self.model_path = base_dir / "models" / "email" / "email_model.pkl"
        self.vec_path = base_dir / "models" / "email" / "email_vectorizer.pkl"
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if self.pipeline_path.exists():
            try:
                self.pipeline = joblib.load(self.pipeline_path)
            except Exception as e:
                print(f"Error loading email pipeline: {e}")
        elif self.model_path.exists() and self.vec_path.exists():
            try:
                model = joblib.load(self.model_path)
                vectorizer = joblib.load(self.vec_path)
                from sklearn.pipeline import Pipeline
                self.pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
            except Exception as e:
                print(f"Error loading email model: {e}")

    def analyze(self, text: str) -> dict:
        if not self.pipeline:
            return {
                "result": "ERROR",
                "is_phishing": False,
                "confidence": 0.0,
                "phish_probability": 0.0,
                "details": ["ML Email model pipeline not loaded."]
            }
        
        try:
            prediction = int(self.pipeline.predict([text])[0])
            prob = self.pipeline.predict_proba([text])[0]
            phish_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            confidence = round(max(prob) * 100.0, 2)
            result = "FAKE" if prediction == 1 else "REAL"

            return {
                "result": result,
                "is_phishing": prediction == 1,
                "confidence": confidence,
                "phish_probability": round(phish_prob, 4),
                "details": [f"Machine Learning Email Classifier predicted {result} ({confidence}% confidence)."]
            }
        except Exception as e:
            return {
                "result": "ERROR",
                "is_phishing": False,
                "confidence": 0.0,
                "phish_probability": 0.0,
                "details": [str(e)]
            }
