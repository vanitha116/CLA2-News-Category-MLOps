import json
import os
import sys
import yaml


def run_quality_gate():
    print("[INFO] =========================================")
    print("[INFO] Starting Stage 07: Model Quality Gate")
    print("[INFO] =========================================")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report_path = config["paths"]["evaluation_report"]
    gate_report_path = config["paths"]["quality_gate_report"]
    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    if not os.path.exists(report_path):
        print(f"[ERROR] Evaluation report not found: {report_path}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    actual_value = float(report["metrics"].get(metric_name, 0.0))
    passed = actual_value >= minimum_value

    print(f"[INFO] Quality Gate Check: {metric_name} = {actual_value:.4f} | Minimum Required >= {minimum_value:.4f}")

    os.makedirs(os.path.dirname(gate_report_path), exist_ok=True)
    gate_report = {
        "stage": "Stage 07 - Model Quality Gate (Lab 8)",
        "metric": metric_name,
        "minimum_value": minimum_value,
        "actual_value": actual_value,
        "status": "PASSED" if passed else "FAILED"
    }

    with open(gate_report_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=4)

    if passed:
        print("[SUCCESS] Model Quality Gate PASSED. Candidate qualifies for lifecycle evaluation.")
    else:
        print("[ERROR] Model Quality Gate FAILED. Candidate will not be promoted to Production.")

    return passed


if __name__ == "__main__":
    passed = run_quality_gate()
    sys.exit(0 if passed else 1)
