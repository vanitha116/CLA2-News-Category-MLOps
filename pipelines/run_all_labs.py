import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAB_RUNNERS = [
    ("Lab 1: Data Ingestion & Lineage", "pipelines/run_lab1_data_loading.py"),
    ("Lab 2: Preprocessing Profiling", "pipelines/run_lab2_preprocessing.py"),
    ("Lab 3: Baseline ML Pipeline", "pipelines/run_lab3_baseline.py"),
    ("Lab 4: Experiment Tracking & Reproducibility", "pipelines/run_lab4_tracking.py"),
    ("Lab 5: Production Pipeline & Validation", "pipelines/run_lab5_pipeline.py"),
    ("Lab 6: Model Registry & Lifecycle", "pipelines/run_lab6_registry.py"),
    ("Lab 7: Pipeline Engineering", "pipelines/run_lab7_pipeline.py"),
    ("Lab 8: CI Automation & Quality Gate", "pipelines/run_lab8_ci.py"),
    ("Lab 9: Failure Injection & Recovery", "pipelines/run_lab9_failure_handling.py"),
    ("Lab 10: Model Serving & API Engineering", "pipelines/run_lab10_api.py"),
    ("Lab 11: Containerization & Deployment", "pipelines/run_lab11_deployment.py"),
    ("Lab 12: Continuous Monitoring & Drift", "pipelines/run_lab12_monitoring.py"),
    ("Lab 13: End-to-End Retraining Trigger", "pipelines/run_lab13_end_to_end.py")
]


def execute_all_labs():
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "    NEWS CATEGORY CLASSIFICATION MLOPS PLATFORM: LABS 1 TO 13 RUNNER".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    overall_start = time.time()
    summary = []

    for name, runner_rel in LAB_RUNNERS:
        print("\n" + "=" * 75)
        print(f"  EXECUTING: {name}")
        print(f"  Runner   : {runner_rel}")
        print("=" * 75)

        script = PROJECT_ROOT / runner_rel
        t0 = time.time()
        result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
        elapsed = time.time() - t0
        status = "PASSED" if result.returncode == 0 else "FAILED"

        summary.append({
            "lab": name,
            "status": status,
            "duration": round(elapsed, 2)
        })

        if result.returncode != 0:
            print(f"\n[HALT] {name} failed with code {result.returncode}. Stopping master execution.")
            break

    total_time = time.time() - overall_start

    print("\n" + "#" * 80)
    print("MASTER EXECUTION SUMMARY REPORT:")
    print("#" * 80)
    for s in summary:
        print(f"  [{s['status']:6s}] {s['lab']:45s} ({s['duration']:6.2f} s)")
    print("-" * 80)
    print(f"Total Duration: {total_time:.2f} seconds")
    print("#" * 80)

    all_passed = all(s["status"] == "PASSED" for s in summary) and len(summary) == len(LAB_RUNNERS)
    return all_passed


if __name__ == "__main__":
    success = execute_all_labs()
    sys.exit(0 if success else 1)
