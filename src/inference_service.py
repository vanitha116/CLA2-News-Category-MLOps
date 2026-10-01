import os
import time
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.config import (
    MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME,
    MODELS_DIR, ALL_CATEGORIES
)

class NewsInferenceService:
    """
    Production Inference Service:
    Pulls the active 'Production' stage model and its preprocessing pipeline
    from the MLflow Model Registry to perform real-time news category classification.
    """
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.model = None
        self.preprocessor = None
        self.model_version = None
        self._load_production_artifacts()

    def _load_production_artifacts(self):
        print(f"[INFERENCE] Connecting to MLflow Registry: '{MODEL_REGISTRY_NAME}'...")
        versions = self.client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
        prod_version = next((v for v in versions if v.current_stage == "Production"), None)

        if not prod_version:
            raise RuntimeError(f"No model in 'Production' stage found in registry '{MODEL_REGISTRY_NAME}'.")

        self.model_version = prod_version.version
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/Production"
        print(f"[INFERENCE] Loading Production model (Version {self.model_version}) from: {model_uri}...")
        self.model = mlflow.sklearn.load_model(model_uri)

        # Load Preprocessor Pipeline Dependency
        prep_path = MODELS_DIR / "preprocessor_pipeline.pkl"
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")

        print(f"[INFERENCE] Loading preprocessor dependency from {prep_path}...")
        self.preprocessor = joblib.load(prep_path)
        print(f"[INFERENCE] [READY] Inference service initialized successfully!")

    def predict(self, headline: str, short_description: str = ""):
        """
        Classifies an input news article and returns the predicted category with latency.
        """
        raw_text = [f"{headline} {short_description}".strip()]

        t0 = time.time()
        # Transform using preprocessor pipeline
        features = self.preprocessor.transform(raw_text)

        # Predict using registered production model
        pred_label = self.model.predict(features)[0]
        latency_ms = (time.time() - t0) * 1000

        return {
            "headline": headline,
            "predicted_category": str(pred_label),
            "production_model_version": int(self.model_version),
            "inference_latency_ms": round(latency_ms, 2)
        }

if __name__ == "__main__":
    service = NewsInferenceService()

    sample_articles = [
        ("NASA's James Webb Telescope Discovers Distant Galaxy From Dawn Of Universe", "Astronomers observe cosmic structures formed just after the Big Bang."),
        ("Stock Markets Rally As Federal Reserve Lowers Interest Rates", "Wall Street indexes reach new records following policy announcements."),
        ("Hollywood Stars Stun On The Red Carpet At The Academy Awards", "Fashion critics praise avant-garde designs at the annual gala."),
        ("Election Turnout Surges In Crucial Swing States As Voters Head To Polls", "Voters cast ballots in historic midterm elections across the nation.")
    ]

    print("\n" + "="*80)
    print("LIVE PRODUCTION INFERENCE VERIFICATION")
    print("="*80)
    for head, desc in sample_articles:
        result = service.predict(head, desc)
        print(f"\nHeadline:  \"{result['headline']}\"")
        print(f"Predicted: -> [ {result['predicted_category']} ] (Model V{result['production_model_version']} in {result['inference_latency_ms']} ms)")
    print("="*80)
