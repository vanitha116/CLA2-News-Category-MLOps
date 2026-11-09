import json
import os
import sys
import numpy as np
import scipy.sparse as sp
import yaml


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]

    X_train_path = os.path.join(processed_dir, "X_train_final.npz")
    X_test_path = os.path.join(processed_dir, "X_test_final.npz")
    y_train_path = os.path.join(processed_dir, "y_train.npy")
    y_test_path = os.path.join(processed_dir, "y_test.npy")

    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("[ERROR] Output matrices not found. Run preprocessing first.")
        return False

    X_train = sp.load_npz(X_train_path)
    X_test = sp.load_npz(X_test_path)
    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    errors = []

    # 1. Assert No Missing Values / NaNs
    nan_train = int(np.isnan(X_train.data).sum())
    nan_test = int(np.isnan(X_test.data).sum())
    if nan_train > 0:
        errors.append(f"NaNs detected in X_train: {nan_train}")
    if nan_test > 0:
        errors.append(f"NaNs detected in X_test: {nan_test}")

    # 2. Check Dimensionality Consistency
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature dimension mismatch: Train ({X_train.shape[1]}) != Test ({X_test.shape[1]})")

    # 3. Check Row Counts Match Target Vectors
    if X_train.shape[0] != len(y_train):
        errors.append(f"Row count mismatch: X_train ({X_train.shape[0]}) != y_train ({len(y_train)})")
    if X_test.shape[0] != len(y_test):
        errors.append(f"Row count mismatch: X_test ({X_test.shape[0]}) != y_test ({len(y_test)})")

    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/preprocessing_summary_report.json"

    report = {
        "stage": "Lab 5 - Processed Output Validation",
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": [len(y_train)],
            "y_test_shape": [len(y_test)]
        },
        "data_quality": {
            "missing_values_X_train": nan_train,
            "missing_values_X_test": nan_test
        },
        "errors": errors
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("[SUCCESS] Output Validation PASSED.")
        print(f"[INFO] Features are clean, non-null, and dimensionally consistent ({X_train.shape[1]} features).")
        print(f"[INFO] Summary report saved to {report_path}.")
        return True


if __name__ == "__main__":
    passed = validate_preprocessing_outputs()
    sys.exit(0 if passed else 1)
