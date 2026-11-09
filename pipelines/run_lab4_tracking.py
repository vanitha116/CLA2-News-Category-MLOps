import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab4():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 4: MLflow Experiment Tracking & Reproducibility")
    print("[INFO] =========================================")
    
    scripts = [
        ("src/preprocess.py", "Data Preprocessing"),
        ("src/train_mlflow.py", "MLflow Model Tracking & Reproducibility Check")
    ]
    
    for script_rel, name in scripts:
        print(f"\n[INFO] ---> Executing {name} ({script_rel})...")
        script = PROJECT_ROOT / script_rel
        result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Lab 4 Pipeline fully executed!")

if __name__ == "__main__":
    run_lab4()
