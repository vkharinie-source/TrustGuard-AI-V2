import os
from pathlib import Path
import joblib

class MLCallService:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[2]
        self.pipeline_path = base_dir / "models" / "scam_calls_training_v2" / "scam_calls_pipeline.pkl"
        self.model_path = base_dir / "models" / "call" / "call_model.pkl"
        self.vec_path = base_dir / "models" / "call" / "call_vectorizer.pkl"
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if self.pipeline_path.exists():
            try:
                self.pipeline = joblib.load(self.pipeline_path)
            except Exception as e:
                print(f"Error loading scam calls pipeline: {e}")
        elif self.model_path.exists() and self.vec_path.exists():
            try:
                model = joblib.load(self.model_path)
                vectorizer = joblib.load(self.vec_path)
                from sklearn.pipeline import Pipeline
                self.pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
            except Exception as e:
                print(f"Error loading call model: {e}")

    def analyze(self, transcript: str) -> dict:
        if not self.pipeline:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "is_scam": False,
                "scam_probability": 0.0,
                "details": ["ML Model not trained."]
            }
        
        try:
            prediction = int(self.pipeline.predict([transcript])[0])
            prob = self.pipeline.predict_proba([transcript])[0]
            scam_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            confidence = round(max(prob) * 100.0, 2)
            result = "SCAM" if prediction == 1 else "LEGITIMATE"

            return {
                "result": result,
                "is_scam": prediction == 1,
                "confidence": confidence,
                "scam_probability": round(scam_prob, 4),
                "details": [f"ML Scam Call Model (Kaggle Benchmark) predicted {result} ({confidence}% confidence)."]
            }
        except Exception as e:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "is_scam": False,
                "scam_probability": 0.0,
                "details": [str(e)]
            }

