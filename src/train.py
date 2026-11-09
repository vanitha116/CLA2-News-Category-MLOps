import os
import joblib
import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression


def train_model():
    print("[INFO] Starting Baseline Model Training...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    X_train_path = os.path.join(processed_dir, "X_train_final.npz")
    y_train_path = os.path.join(processed_dir, "y_train.npy")

    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print("[ERROR] Training data not found. Run src/preprocess.py first.")
        return False

    X_train = sp.load_npz(X_train_path)
    y_train = np.load(y_train_path, allow_pickle=True)

    model_cfg = config["model"]
    model = LogisticRegression(
        C=model_cfg.get("C", 1.0),
        max_iter=500,
        random_state=model_cfg.get("random_state", 42),
        class_weight=model_cfg.get("class_weight", "balanced")
    )

    print(f"[INFO] Fitting Logistic Regression on {X_train.shape[0]} training samples...")
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(config["paths"]["model"]), exist_ok=True)
    joblib.dump(model, config["paths"]["model"])
    print(f"[SUCCESS] Model saved successfully to {config['paths']['model']}.")
    return True


if __name__ == "__main__":
    train_model()
