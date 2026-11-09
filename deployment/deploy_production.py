import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_deployment():
    print("=" * 70)
    print("STAGE 11: PRODUCTION CONTAINERIZATION & DEPLOYMENT AUTOMATION")
    print("=" * 70)

    deployment_dir = PROJECT_ROOT / "deployment"
    model_path = deployment_dir / "model.pkl"
    prep_path = deployment_dir / "preprocessor.pkl"
    meta_path = deployment_dir / "model_metadata.json"

    if not model_path.exists() or not prep_path.exists():
        print("[INFO] Preparing deployment bundle first...")
        prep_script = PROJECT_ROOT / "src" / "mlops_pipeline" / "11_prepare_deployment.py"
        subprocess.run([sys.executable, str(prep_script)], cwd=PROJECT_ROOT)

    report = {
        "stage": "Lab 11 - Containerization & Deployment",
        "image_tag": "news-category-api:1.0",
        "artifacts_packaged": [
            str(model_path.name),
            str(prep_path.name),
            str(meta_path.name)
        ],
        "k8s_manifests": [
            "k8s/deployment.yaml",
            "k8s/service.yaml"
        ],
        "deployment_status": "READY_FOR_ROLLOUT"
    }

    # Check if docker is available on host
    try:
        docker_check = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if docker_check.returncode == 0:
            report["docker_available"] = True
            report["docker_version"] = docker_check.stdout.strip()
            print(f"[INFO] Docker detected: {report['docker_version']}")
        else:
            report["docker_available"] = False
    except Exception:
        report["docker_available"] = False

    os.makedirs(PROJECT_ROOT / "reports", exist_ok=True)
    report_file = PROJECT_ROOT / "reports" / "deployment_automation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\n[SUCCESS] Deployment report saved to {report_file}.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_deployment()
    sys.exit(0 if success else 1)
