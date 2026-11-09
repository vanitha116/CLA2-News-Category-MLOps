import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab13():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 13: End-to-End MLOps Automation & Retraining Trigger")
    print("[INFO] =========================================")
    
    script = PROJECT_ROOT / "src" / "mlops_pipeline" / "12_monitor_trigger.py"
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print("[ERROR] Lab 13 execution failed.")
        sys.exit(result.returncode)
        
    print("\n[SUCCESS] Lab 13 End-to-End MLOps Automation fully executed!")

if __name__ == "__main__":
    run_lab13()
