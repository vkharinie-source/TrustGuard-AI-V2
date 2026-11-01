import os
from pathlib import Path
import joblib

class MLJobScamService:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[2]
        self.pipeline_path = base_dir / "models" / "job_scam_training_v3" / "job_scam_pipeline.pkl"
        self.model_path = base_dir / "models" / "job_scam" / "job_scam_model.pkl"
        self.vec_path = base_dir / "models" / "job_scam" / "job_scam_vectorizer.pkl"
        self.pipeline = None
        self.model = None
        self.vectorizer = None
        self.threshold = 0.65
        self._load_model()

    def _load_model(self):
        if self.pipeline_path.exists():
            try:
                self.pipeline = joblib.load(self.pipeline_path)
            except Exception as e:
                print(f"Error loading job scam v3 pipeline: {e}")
        elif os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vec_path)
                from sklearn.pipeline import Pipeline
                self.pipeline = Pipeline([("tfidf", self.vectorizer), ("clf", self.model)])
            except Exception as e:
                print(f"Error loading legacy job scam model: {e}")

    def analyze(self, text: str) -> dict:
        if not self.pipeline:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "details": ["Job Scam ML Model not loaded."]
            }
        
        try:
            prob = self.pipeline.predict_proba([text])[0]
            scam_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            is_scam = scam_prob >= self.threshold
            prediction = 1 if is_scam else 0
            
            confidence = round(float(scam_prob * 100.0 if is_scam else (1.0 - scam_prob) * 100.0), 2)
            result = "SCAM" if prediction == 1 else "LEGITIMATE"

            return {
                "result": result,
                "confidence": confidence,
                "scam_probability": round(scam_prob, 4),
                "details": [f"Machine Learning Job Scam Classifier predicted {result} ({confidence}% confidence)."]
            }
        except Exception as e:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "details": [str(e)]
            }
