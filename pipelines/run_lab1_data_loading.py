import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab1():
    print("=" * 70)
    print("LAB 1: DATA INGESTION, LINEAGE TRACKING & VERSIONING")
    print("=" * 70)
    
    script = PROJECT_ROOT / "src" / "mlops_pipeline" / "01_load_data.py"
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print("[ERROR] Lab 1 execution failed.")
        sys.exit(result.returncode)
        
    print("\n[SUCCESS] Lab 1 Pipeline fully executed!")

if __name__ == "__main__":
    run_lab1()
