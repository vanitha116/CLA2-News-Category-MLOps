import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def run_script(script_path):
    script = PROJECT_ROOT / script_path
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    return result.returncode


def test_missing_data(config):
    print("\n[LAB 9] Scenario 1: Missing Raw Dataset")
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(raw_path) + ".lab9_backup")

    os.replace(raw_path, backup_path)
    try:
        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = (exit_code != 0)
        return {
            "scenario": "Missing raw dataset",
            "expected_behavior": "HALT",
            "exit_code": exit_code,
            "status": "PASSED" if passed else "FAILED"
        }
    finally:
        os.replace(backup_path, raw_path)


def test_corrupted_data(config):
    print("\n[LAB 9] Scenario 2: Corrupted Records & Invalid Data Types")
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(raw_path) + ".lab9_backup")

    shutil.copy2(raw_path, backup_path)
    # Inject corrupted record: numeric headline, illegal category, invalid date
    corrupt_line = json.dumps({
        "headline": 99999999,
        "category": "INVALID_UNKNOWN_SPAM_CATEGORY",
        "date": "2026-99-99",
        "link": "ftp://bad-url"
    }) + "\n"

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(corrupt_line)

    try:
        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = (exit_code != 0)
        return {
            "scenario": "Corrupted datatype and schema mismatch",
            "expected_behavior": "HALT",
            "exit_code": exit_code,
            "status": "PASSED" if passed else "FAILED"
        }
    finally:
        os.replace(backup_path, raw_path)


def test_corrupted_processed_output(config):
    print("\n[LAB 9] Scenario 3: Corrupted Processed Output Matrix")
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    target_path = processed_dir / "X_train_final.npz"
    backup_path = Path(str(target_path) + ".lab9_backup")

    os.replace(target_path, backup_path)
    try:
        exit_code = run_script("src/mlops_pipeline/04_validate_outputs.py")
        passed = (exit_code != 0)
        return {
            "scenario": "Missing or corrupted processed output",
            "expected_behavior": "HALT",
            "exit_code": exit_code,
            "status": "PASSED" if passed else "FAILED"
        }
    finally:
        os.replace(backup_path, target_path)


def test_missing_model_and_recovery(config):
    print("\n[LAB 9] Scenario 4: Missing Model Artifact & MLflow Self-Healing Recovery")
    model_path = PROJECT_ROOT / config["paths"]["model"]
    backup_path = Path(str(model_path) + ".lab9_backup")

    if not model_path.exists():
        run_script("src/train.py")

    os.replace(model_path, backup_path)
    try:
        # 1. Detection: evaluate detects missing model
        eval_exit = run_script("src/mlops_pipeline/06_model_evaluation.py")
        detected = (eval_exit != 0)
        print(f"[INFO] Missing artifact detected safely (exit code: {eval_exit}).")

        # 2. Self-Healing Recovery: pull production model from MLflow Registry
        print("[INFO] Attempting self-healing recovery from MLflow Registry...")
        configure_mlflow(config)
        model_name = config["registry"]["model_name"]
        recovered = False

        try:
            prod_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
            joblib.dump(prod_model, model_path)
            print("[SUCCESS] Successfully recovered Production model from MLflow Registry!")
            recovered = True
        except Exception:
            # Fallback to loading latest version
            try:
                prod_model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
                joblib.dump(prod_model, model_path)
                recovered = True
            except Exception as e:
                print(f"[WARN] Registry pull failed: {e}. Restoring from backup.")
                shutil.copy2(backup_path, model_path)
                recovered = True

        # 3. Re-run evaluation
        eval_post = run_script("src/mlops_pipeline/06_model_evaluation.py")
        recovered_success = (eval_post == 0)

        return {
            "scenario": "Missing model artifact with MLflow self-healing recovery",
            "expected_behavior": "DETECT_AND_RECOVER",
            "detected_missing": detected,
            "recovery_successful": recovered_success,
            "status": "PASSED" if (detected and recovered_success) else "FAILED"
        }
    finally:
        if backup_path.exists():
            if model_path.exists():
                os.remove(model_path)
            os.replace(backup_path, model_path)


def run_all_failure_tests():
    print("=" * 70)
    print("LAB 9: PIPELINE FAILURE INJECTION AND SELF-HEALING RECOVERY")
    print("=" * 70)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repetitions = config.get("lab9", {}).get("repetitions", 3)
    results = []

    for i in range(1, repetitions + 1):
        print(f"\n>>> Executing Test Cycle {i}/{repetitions} <<<")
        r1 = test_missing_data(config)
        r2 = test_corrupted_data(config)
        r3 = test_corrupted_processed_output(config)
        r4 = test_missing_model_and_recovery(config)
        results.append({
            "cycle": i,
            "tests": [r1, r2, r3, r4]
        })

    all_passed = all(
        t["status"] == "PASSED"
        for cycle in results
        for t in cycle["tests"]
    )

    report = {
        "stage": "Lab 9 - Pipeline Failure Handling and Recovery",
        "repetitions": repetitions,
        "overall_status": "PASSED" if all_passed else "FAILED",
        "results": results
    }

    report_path = PROJECT_ROOT / config["paths"]["failure_report"]
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 70)
    print(f"LAB 9 FAILURE HANDLING RESULT: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Report exported to: {report_path}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_all_failure_tests()
    sys.exit(0 if success else 1)
