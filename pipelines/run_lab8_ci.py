import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab8():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 8: Pipeline Automation & Quality Gate")
    print("[INFO] =========================================")
    
    steps = [
        ("src/mlops_pipeline/news_classification_pipeline.py", "Pipeline Execution", ["--force"]),
        ("src/mlops_pipeline/07_quality_gate.py", "Quality Gate Enforcement", [])
    ]
    
    for script_rel, name, args in steps:
        print(f"\n[INFO] ---> Executing {name} ({script_rel})...")
        script = PROJECT_ROOT / script_rel
        result = subprocess.run([sys.executable, str(script)] + args, cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Lab 8 CI & Quality Gate fully executed!")

if __name__ == "__main__":
    run_lab8()
