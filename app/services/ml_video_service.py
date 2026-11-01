class MLVideoService:
    def __init__(self):
        # A suitable Kaggle dataset for video (DFDC) is terabytes in size, which is not practical for Render deployments.
        # As per the requirements, we DO NOT replace or modify the existing image model.
        pass

    def analyze(self, video_path: str) -> dict:
        return {
            "result": "ERROR",
            "confidence": 0.0,
            "details": ["A suitable Kaggle video dataset was not used due to size constraints for Render deployment. Existing image DeepFake model is retained."]
        }
