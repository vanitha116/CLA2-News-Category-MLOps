import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_lab7():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 7: ML Pipeline Engineering")
    print("[INFO] =========================================")
    
    script = PROJECT_ROOT / "src" / "mlops_pipeline" / "news_classification_pipeline.py"
    result = subprocess.run([sys.executable, str(script), "--force"], cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print("[ERROR] Lab 7 Pipeline execution failed.")
        sys.exit(result.returncode)
        
    print("\n[SUCCESS] Lab 7 ML Pipeline fully executed!")

if __name__ == "__main__":
    run_lab7()
