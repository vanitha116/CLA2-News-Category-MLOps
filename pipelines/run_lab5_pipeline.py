import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab5():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 5: Production Data Pipeline & Validation")
    print("[INFO] =========================================")
    
    steps = [
        ("src/validate_data.py", "Data Validation"),
        ("src/preprocess_pipeline.py", "Production Preprocessing Pipeline"),
        ("src/validate_outputs.py", "Output Validation")
    ]
    
    for script_rel, name in steps:
        print(f"\n[INFO] ---> Executing {name} ({script_rel})...")
        script = PROJECT_ROOT / script_rel
        result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with code {result.returncode}.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Lab 5 Production Pipeline fully executed!")

if __name__ == "__main__":
    run_lab5()
