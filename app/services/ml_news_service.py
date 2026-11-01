import os
from pathlib import Path
import joblib

class MLNewsService:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[2]
        self.pipeline_path = base_dir / "models" / "fake_news_training_v2" / "fake_news_pipeline.pkl"
        self.model_path = base_dir / "models" / "news" / "news_model.pkl"
        self.vec_path = base_dir / "models" / "news" / "news_vectorizer.pkl"
        self.pipeline = None
        self.model = None
        self.vectorizer = None
        self.threshold = 0.40
        self._load_model()

    def _load_model(self):
        if self.pipeline_path.exists():
            try:
                self.pipeline = joblib.load(self.pipeline_path)
            except Exception as e:
                print(f"Error loading fake news pipeline: {e}")
        elif os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vec_path)
                from sklearn.pipeline import Pipeline
                self.pipeline = Pipeline([("tfidf", self.vectorizer), ("clf", self.model)])
            except Exception as e:
                print(f"Error loading legacy news model: {e}")

    def analyze(self, text: str) -> dict:
        if not self.pipeline:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "details": ["News ML Model not found."]
            }
        
        try:
            prob = self.pipeline.predict_proba([text])[0]
            fake_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            is_fake = fake_prob >= self.threshold
            prediction = 1 if is_fake else 0
            
            confidence = round(float(fake_prob * 100.0 if is_fake else (1.0 - fake_prob) * 100.0), 2)
            result = "FAKE" if prediction == 1 else "REAL"

            return {
                "result": result,
                "confidence": confidence,
                "fake_probability": round(fake_prob, 4),
                "details": [f"Machine Learning Fake News Classifier predicted {result} ({confidence}% confidence)."]
            }
        except Exception as e:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "details": [str(e)]
            }
