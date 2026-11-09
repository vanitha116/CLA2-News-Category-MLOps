import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def train_and_register():
    print("[INFO] Starting Stage 05: Training + MLflow + Model Registry...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = config["data"]["processed_dir"]
    X_train = sp.load_npz(os.path.join(processed_dir, "X_train_final.npz"))
    X_test = sp.load_npz(os.path.join(processed_dir, "X_test_final.npz"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"), allow_pickle=True)

    model_name = config["registry"]["model_name"]
    params = {
        "C": config["model"].get("C", 1.0),
        "class_weight": config["model"].get("class_weight", "balanced"),
        "random_state": config["model"].get("random_state", 42)
    }

    with mlflow.start_run(run_name="LinearSVC_PipelineCandidate") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "LinearSVC")
        mlflow.log_param("lab", "Lab 7 Pipeline")

        print("[INFO] Training LinearSVC Candidate...")
        model = LinearSVC(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        }

        mlflow.log_metrics(metrics)

        preprocessor_path = config["paths"]["preprocessor"]
        if os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessing_dependency")

        print(f"[INFO] Pushing Candidate to Model Registry: '{model_name}'...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

        joblib.dump(model, config["paths"]["model"])
        print(f"[SUCCESS] Candidate trained: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        return True


if __name__ == "__main__":
    passed = train_and_register()
    sys.exit(0 if passed else 1)
