import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab2():
    print("=" * 70)
    print("LAB 2: DATA VALIDATION & PREPROCESSING PROFILING")
    print("=" * 70)
    
    steps = [
        ("src/mlops_pipeline/02_validate_data.py", "Raw Data Validation"),
        ("src/preprocess.py", "Baseline Data Preprocessing")
    ]
    
    for script_rel, name in steps:
        print(f"\n[INFO] ---> Executing {name} ({script_rel})...")
        script = PROJECT_ROOT / script_rel
        result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with code {result.returncode}.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Lab 2 Pipeline fully executed!")

if __name__ == "__main__":
    run_lab2()
