import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    ("Data Loading & Ingestion", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing Pipeline", "src/mlops_pipeline/03_preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Training + MLflow + Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Deployment Update", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Production Inference", "src/mlops_pipeline/09_inference.py")
]


def load_data_load_report():
    report_path = PROJECT_ROOT / "artifacts" / "data_load_report.json"
    if not report_path.exists():
        return {"status": "ingested"}
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(force=False):
    print("=" * 70)
    print("STARTING NEWS CATEGORY CLASSIFICATION MLOPS PIPELINE")
    print("=" * 70)

    run_log = {"pipeline": "News Category Classification MLOps", "stages": [], "force": force}

    for stage_index, (stage_name, script_rel_path) in enumerate(STAGES):
        print("\n" + "=" * 70)
        print(f"RUNNING STAGE {stage_index + 1}: {stage_name}")
        print("=" * 70)

        script_path = PROJECT_ROOT / script_rel_path
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
        status = "PASSED" if result.returncode == 0 else "FAILED"

        run_log["stages"].append({
            "stage": stage_name,
            "status": status,
            "exit_code": result.returncode
        })

        print(f"--> {stage_name}: {status}")

        if result.returncode != 0:
            print(f"[HALT] Pipeline stopped because '{stage_name}' failed.")
            run_log["overall_status"] = "FAILED"
            break

        # Handle Lab 1 duplicate batch check
        if stage_index == 0 and not force:
            report = load_data_load_report()
            data_status = report.get("status")
            if data_status == "already_ingested":
                print("[INFO] Batch was already ingested. Stopping safely without retraining.")
                run_log["overall_status"] = "NO_NEW_DATA"
                break

    all_passed = len(run_log["stages"]) == len(STAGES) and all(s["status"] == "PASSED" for s in run_log["stages"])
    if all_passed:
        run_log["overall_status"] = "PASSED"

    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)
    log_path = PROJECT_ROOT / "logs" / "news_classification_pipeline_run_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)
    if run_log.get("overall_status") == "PASSED":
        print("[SUCCESS] News Category Classification MLOps Pipeline completed successfully!")
    elif run_log.get("overall_status") == "NO_NEW_DATA":
        print("[INFO] No new data batch to retrain on. Pipeline stopped gracefully.")
    else:
        print("[ERROR] News Category Classification MLOps Pipeline failed.")
    print(f"[INFO] Pipeline log saved to: {log_path}")
    print("=" * 70)
    return run_log.get("overall_status") in {"PASSED", "NO_NEW_DATA"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run News Classification MLOps Pipeline")
    parser.add_argument("--force", action="store_true", help="Force pipeline execution across all stages")
    args = parser.parse_args()

    passed = run_pipeline(force=args.force)
    sys.exit(0 if passed else 1)
