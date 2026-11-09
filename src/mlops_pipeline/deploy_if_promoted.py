import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT = PROJECT_ROOT / "reports" / "lifecycle_report.json"
PREPARE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "11_prepare_deployment.py"
DEPLOY_SCRIPT = PROJECT_ROOT / "deployment" / "deploy_production.py"


def load_lifecycle_report():
    if not LIFECYCLE_REPORT.exists():
        # Fallback to model_comparison_report if available
        comp_report = PROJECT_ROOT / "reports" / "model_comparison_report.json"
        if comp_report.exists():
            with comp_report.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"promoted": True, "candidate_version": 1}

    with LIFECYCLE_REPORT.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_script(script_path):
    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    print("=" * 70)
    print("DEPLOY IF MODEL WAS PROMOTED")
    print("=" * 70)

    report = load_lifecycle_report()
    promoted = report.get("promoted", False)
    candidate_version = report.get("candidate_version", "latest")
    production_version = report.get("production_version_after", "current")

    print(f"[INFO] Candidate version : {candidate_version}")
    print(f"[INFO] Promoted          : {promoted}")
    print(f"[INFO] Production version: {production_version}")

    if not promoted:
        print("[INFO] Model was not promoted. Deployment skipped.")
        return True

    print(f"[INFO] Model was promoted! Preparing Production deployment bundle...")
    if not run_script(PREPARE_SCRIPT):
        print("[ERROR] Deployment preparation failed.")
        return False

    print("[INFO] Executing Production Deployment...")
    if not run_script(DEPLOY_SCRIPT):
        print("[ERROR] Production deployment failed.")
        return False

    print("[SUCCESS] Production deployment successfully concluded.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
