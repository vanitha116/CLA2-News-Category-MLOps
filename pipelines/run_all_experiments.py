import sys
import os
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_step(script_name: str, exp_title: str):
    print("\n" + "="*85)
    print(f"  EXECUTION PHASE: {exp_title}")
    print(f"  Script: pipelines/{script_name}")
    print("="*85)

    script_path = BASE_DIR / "pipelines" / script_name
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))

    if result.returncode != 0:
        print(f"\n[FAILURE] {exp_title} exited with error code {result.returncode}.")
        sys.exit(result.returncode)

    elapsed = time.time() - t0
    print(f"[PHASE COMPLETED] {exp_title} finished in {elapsed:.2f}s.")

if __name__ == "__main__":
    print("\n" + "*"*85)
    print("*" + " "*83 + "*")
    print("*" + "   MASTER MLOPS PIPELINE: EXPERIMENTS 4, 5, AND 6 SEQUENTIAL RUNNER".center(83) + "*")
    print("*" + " "*83 + "*")
    print("*"*85)

    overall_start = time.time()

    # Step 1: Experiment 4
    run_step("run_experiment_4.py", "Experiment 4: MLflow Tracking & Reproducibility")

    # Step 2: Experiment 5
    run_step("run_experiment_5.py", "Experiment 5: Pandera Data Validation & Idempotent Preprocessing")

    # Step 3: Experiment 6
    run_step("run_experiment_6.py", "Experiment 6: MLflow Model Registry & Champion-Challenger Lifecycle")

    total_time = time.time() - overall_start
    print("\n" + "*"*85)
    print("*" + " "*83 + "*")
    print("*" + "   ALL MLOPS EXPERIMENTS SUCCESSFULLY COMPLETED!".center(83) + "*")
    print("*" + f"   Total Pipeline Duration: {total_time:.2f} seconds".center(83) + "*")
    print("*" + " "*83 + "*")
    print("*"*85)
    print("\nNext Steps:")
    print("1. Launch the interactive MLflow UI in your VS Code terminal:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("2. Open and run the interactive notebooks under 'notebooks/'")
    print("   - notebooks/experiment_4_tracking.ipynb")
    print("   - notebooks/experiment_5_data_validation.ipynb")
    print("   - notebooks/experiment_6_model_registry.ipynb")
