import json
import os
import time
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def train_and_track(run_name="LogisticRegression_Baseline"):
    print(f"\n--- Starting MLflow Run: {run_name} ---")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = config["data"]["processed_dir"]
    X_train = sp.load_npz(os.path.join(processed_dir, "X_train_final.npz"))
    X_test = sp.load_npz(os.path.join(processed_dir, "X_test_final.npz"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"), allow_pickle=True)

    params = {
        "C": config["model"]["C"],
        "max_iter": 500,
        "random_state": config["model"]["random_state"],
        "class_weight": config["model"]["class_weight"]
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", "LogisticRegression")
        mlflow.log_param("train_samples", int(X_train.shape[0]))
        mlflow.log_param("test_samples", int(X_test.shape[0]))

        model = LogisticRegression(**params)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "training_latency_sec": float(train_time)
        }

        mlflow.log_metrics(metrics)
        print(f"[METRICS] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Confusion Matrix artifact
        os.makedirs("artifacts", exist_ok=True)
        top_cats = pd.Series(y_test).value_counts().head(8).index.tolist()
        mask = np.isin(y_test, top_cats) & np.isin(y_pred, top_cats)

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test[mask], y_pred[mask], labels=top_cats, ax=ax, cmap="Blues", xticks_rotation=45
        )
        ax.set_title(f"Confusion Matrix - {run_name}")
        plt.tight_layout()
        cm_path = "artifacts/confusion_matrix.png"
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Save lineage metadata artifact
        meta_path = os.path.join(processed_dir, "dataset_metadata.json")
        if os.path.exists(meta_path):
            mlflow.log_artifact(meta_path, artifact_path="metadata")

        # Log Model artifact
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Save local backup
        joblib.dump(model, config["paths"]["model"])
        print(f"[SUCCESS] Run '{run_name}' successfully tracked to MLflow!")
        return metrics


def run_reproducibility_check():
    print("\n--- Running Deterministic Reproducibility Check ---")
    m1 = train_and_track(run_name="Run_1_Verification")
    m2 = train_and_track(run_name="Run_2_Verification")

    delta_f1 = abs(m1["f1_score"] - m2["f1_score"])
    reproducible = delta_f1 < 1e-6

    report = {
        "stage": "Lab 4 - Reproducibility Verification",
        "delta_f1": delta_f1,
        "is_reproducible": reproducible,
        "status": "PASSED" if reproducible else "FAILED"
    }

    report_path = "artifacts/reproducibility_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[REPRODUCIBILITY] Delta F1: {delta_f1:.8f} | Verified: {reproducible}")
    return reproducible


if __name__ == "__main__":
    train_and_track()
    run_reproducibility_check()
