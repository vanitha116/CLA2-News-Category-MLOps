"""
Lab 11: Containerization & Kubernetes Deployment
Packages deployment bundle and validates consistency
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_lab11():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 11: Containerization & Kubernetes Deployment")
    print("[INFO] =========================================")

    # Step 1: Prepare deployment bundle (model.pkl, preprocessor.pkl, metadata.json into deployment/)
    print("\n[INFO] ---> Step 1: Production Deployment Artifact Preparation...")
    prep_script = PROJECT_ROOT / "src" / "mlops_pipeline" / "11_prepare_deployment.py"
    result = subprocess.run([sys.executable, str(prep_script)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("[ERROR] Deployment bundle preparation failed.")
        sys.exit(result.returncode)

    # Step 2: Run deployment automation
    print("\n[INFO] ---> Step 2: Production Deployment Automation...")
    deploy_script = PROJECT_ROOT / "deployment" / "deploy_production.py"
    result = subprocess.run([sys.executable, str(deploy_script)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("[ERROR] Deployment automation failed.")
        sys.exit(result.returncode)

    # Step 3: Container parity & consistency test via pytest
    print("\n[INFO] ---> Step 3: Container Parity & Model Consistency Test (pytest)...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_local_vs_container.py", "-v", "--tb=short", "-s"],
        cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        print("[ERROR] Container consistency test failed.")
        sys.exit(result.returncode)

    print("\n[SUCCESS] Lab 11 Containerization & Deployment fully executed!")


if __name__ == "__main__":
    run_lab11()
