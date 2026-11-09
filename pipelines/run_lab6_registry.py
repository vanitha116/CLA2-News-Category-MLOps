import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab6():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 6: Model Registry & Lifecycle Management")
    print("[INFO] =========================================")
    
    steps = [
        ("src/train_registry.py", "Model Registration Candidate"),
        ("src/automate_lifecycle.py", "Automated Champion vs Challenger Lifecycle"),
        ("src/generate_registry_report.py", "Production Model Registry Report")
    ]
    
    for script_rel, name in steps:
        print(f"\n[INFO] ---> Executing {name} ({script_rel})...")
        script = PROJECT_ROOT / script_rel
        result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with code {result.returncode}.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Lab 6 Model Registry Pipeline fully executed!")

if __name__ == "__main__":
    run_lab6()
