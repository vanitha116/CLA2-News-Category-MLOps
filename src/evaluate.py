import json
import os
import joblib
import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model():
    print("[INFO] Starting Baseline Model Evaluation...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    model_path = config["paths"]["model"]
    report_path = config["paths"]["evaluation_report"]

    X_test_path = os.path.join(processed_dir, "X_test_final.npz")
    y_test_path = os.path.join(processed_dir, "y_test.npy")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model artifact missing at: {model_path}")
        return False

    model = joblib.load(model_path)
    X_test = sp.load_npz(X_test_path)
    y_test = np.load(y_test_path, allow_pickle=True)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    }

    print("\n--- Model Evaluation Results ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "stage": "Lab 3 - Baseline Evaluation",
        "model_path": model_path,
        "test_samples": int(X_test.shape[0]),
        "metrics": metrics
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\n[SUCCESS] Evaluation report saved to {report_path}.")
    return True


if __name__ == "__main__":
    evaluate_model()
