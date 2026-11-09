import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab9():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 9: Failure Injection & Recovery")
    print("[INFO] =========================================")
    
    script = PROJECT_ROOT / "src" / "mlops_pipeline" / "10_failure_injection.py"
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print("[ERROR] Lab 9 failure injection testing failed.")
        sys.exit(result.returncode)
        
    print("\n[SUCCESS] Lab 9 Failure Handling & Self-Healing fully executed!")

if __name__ == "__main__":
    run_lab9()
