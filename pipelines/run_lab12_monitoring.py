import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab12():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 12: Continuous Monitoring & Drift Retraining")
    print("[INFO] =========================================")
    
    steps = [
        ("src/mlops_pipeline/12_drift_analysis.py", "Consolidated Drift Analysis"),
        ("src/mlops_pipeline/12_retrain_model.py", "Candidate Retraining on Evolved Data"),
        ("src/mlops_pipeline/12_model_comparison.py", "Model Comparison (Production vs Retrained)")
    ]
    
    for script_rel, name in steps:
        print(f"\n[INFO] ---> Executing {name} ({script_rel})...")
        script = PROJECT_ROOT / script_rel
        result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Lab 12 Continuous Monitoring & Drift Retraining fully executed!")

if __name__ == "__main__":
    run_lab12()
