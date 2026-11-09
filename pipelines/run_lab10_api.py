"""
Lab 10: Model Serving & API Engineering
Runs API tests using pytest and validates inference pipeline
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_lab10():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 10: Model Serving & API Engineering")
    print("[INFO] =========================================")

    # Step 1: Verify inference pipeline works (uses local model/preprocessor)
    print("\n[INFO] ---> Step 1: Production Inference Verification...")
    inference_script = PROJECT_ROOT / "src" / "mlops_pipeline" / "09_inference.py"
    result = subprocess.run([sys.executable, str(inference_script)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("[ERROR] Production Inference Verification failed.")
        sys.exit(result.returncode)

    # Step 2: Run API test suite via pytest
    print("\n[INFO] ---> Step 2: FastAPI Endpoints Test Suite (pytest)...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_api.py", "-v", "--tb=short", "-x"],
        cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        print("[ERROR] FastAPI test suite failed.")
        sys.exit(result.returncode)

    # Step 3: Run performance benchmarks via pytest
    print("\n[INFO] ---> Step 3: API Performance Benchmarks (pytest)...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_performance.py", "-v", "--tb=short", "-s"],
        cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        print("[ERROR] Performance benchmarks failed.")
        sys.exit(result.returncode)

    # Write API testing report
    report = {
        "stage": "Lab 10 - ML Model Serving & API Engineering",
        "api_endpoints_verified": ["/", "/health", "/metadata", "/predict", "/predict/batch"],
        "validation_handling": "Standardized 422 with field-level details",
        "service_readiness_check": "Standardized 503 if artifacts missing",
        "status": "PASSED"
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/api_testing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n[SUCCESS] Lab 10 Model Serving & API Engineering fully executed!")


if __name__ == "__main__":
    run_lab10()
