import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import yaml


class ModelLoader:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.model = None
        self.preprocessor = None
        self.model_version = "1"
        self.metadata = {}
        self._load_artifacts()

    def _load_artifacts(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._load_from_registry(config)

        deployment_model = Path("deployment/model.pkl")
        deployment_preprocessor = Path("deployment/preprocessor.pkl")
        if self.model is None:
            for model_path in (deployment_model, Path(config["paths"]["model"])):
                if model_path.is_file():
                    self.model = joblib.load(model_path)
                    print(f"[MODEL LOADER] Loaded model from local path: {model_path}")
                    break

        if self.preprocessor is None:
            for prep_path in (
                deployment_preprocessor,
                Path(config["paths"]["preprocessor"]),
            ):
                if prep_path.is_file():
                    self.preprocessor = joblib.load(prep_path)
                    print(f"[MODEL LOADER] Loaded preprocessor from: {prep_path}")
                    break

        if not self.metadata:
            self.metadata = self._load_metadata(
                ["deployment/model_metadata.json", "artifacts/production_model_report.json"]
            )
        if self.model_version == "1":
            self.model_version = str(self.metadata.get("version", self.model_version))

    def _load_from_registry(self, config):
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            print(f"[MODEL LOADER] MLflow is unavailable ({exc}). Trying local artifacts.")
            return

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        try:
            model_name = config["registry"]["model_name"]
            versions = MlflowClient().search_model_versions(f"name='{model_name}'")
            model_version = next(
                (version for version in versions if version.current_stage == "Production"),
                None,
            )
            if model_version is None and versions:
                model_version = max(versions, key=lambda version: int(version.version))

            if model_version:
                self.model_version = str(model_version.version)
                model_uri = f"models:/{model_name}/{self.model_version}"
                self.model = mlflow.sklearn.load_model(model_uri)
                print(f"[MODEL LOADER] Loaded model V{self.model_version} from MLflow Registry.")
        except Exception as exc:
            print(f"[MODEL LOADER] Could not pull from MLflow Registry ({exc}). Trying local fallback.")

    @staticmethod
    def _load_metadata(metadata_paths):
        for metadata_path in metadata_paths:
            path = Path(metadata_path)
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[MODEL LOADER] Could not read metadata from {path}: {exc}")
        return {}

    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def predict_one(self, headline: str, short_description: str = ""):
        if not self.is_ready():
            raise RuntimeError("Model or preprocessor is not loaded and ready.")

        raw_text = [f"{headline} {short_description}".strip()]
        t0 = time.time()
        X_mat = self.preprocessor.transform(raw_text)
        pred = self.model.predict(X_mat)[0]
        latency_ms = (time.time() - t0) * 1000

        # Try to obtain confidence if available
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_mat)[0]
            confidence = float(np.max(probs))

        return {
            "headline": headline,
            "predicted_category": str(pred),
            "confidence": confidence,
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2)
        }

    def predict_batch(self, articles):
        if not self.is_ready():
            raise RuntimeError("Model or preprocessor is not loaded and ready.")

        texts = [f"{a.headline} {a.short_description}".strip() for a in articles]
        t0 = time.time()
        X_mat = self.preprocessor.transform(texts)
        preds = self.model.predict(X_mat)
        total_latency = (time.time() - t0) * 1000
        avg_latency = total_latency / max(len(articles), 1)

        results = []
        for i, a in enumerate(articles):
            results.append({
                "headline": a.headline,
                "predicted_category": str(preds[i]),
                "confidence": None,
                "model_version": self.model_version,
                "latency_ms": round(avg_latency, 2)
            })
        return results
