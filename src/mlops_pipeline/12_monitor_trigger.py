import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIFT_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "12_drift_analysis.py"
DRIFT_REPORT = PROJECT_ROOT / "outputs" / "drift" / "drift_analysis_report.csv"
NEW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "new_news_category.json"
DATASET_LINEAGE_PATH = PROJECT_ROOT / "data" / "raw" / "archive" / "dataset_lineage.json"
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "news_classification_pipeline.py"
TRIGGER_REPORT = PROJECT_ROOT / "outputs" / "monitoring" / "monitor_trigger_report.json"
DATA_LOAD_REPORT = PROJECT_ROOT / "artifacts" / "data_load_report.json"


def run_drift_analysis():
    print("[INFO] Running Drift Analysis Suite...")
    result = subprocess.run([sys.executable, str(DRIFT_SCRIPT)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def detect_drift():
    if not DRIFT_REPORT.exists():
        return False
    df = pd.read_csv(DRIFT_REPORT)
    for col in df.columns:
        if df[col].astype(str).str.strip().str.upper().eq("DRIFT").any():
            return True
    return False


def get_data_load_status():
    if not DATA_LOAD_REPORT.exists():
        return None
    try:
        with DATA_LOAD_REPORT.open("r", encoding="utf-8") as f:
            return json.load(f).get("status")
    except (OSError, json.JSONDecodeError):
        return None


def has_pending_batch():
    if not NEW_DATA_PATH.exists():
        return False
    if not DATASET_LINEAGE_PATH.exists():
        return True

    with DATASET_LINEAGE_PATH.open("r", encoding="utf-8") as f:
        lineage = json.load(f)
    if not isinstance(lineage, list):
        raise ValueError(f"Dataset lineage must be a list: {DATASET_LINEAGE_PATH}")

    digest = hashlib.sha256()
    with NEW_DATA_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    batch_hash = digest.hexdigest()
    return not any(entry.get("sha256") == batch_hash for entry in lineage)


def run_monitor_trigger():
    print("=" * 70)
    print("LAB 13: END-TO-END MONITORING TRIGGER + RETRAINING WORKFLOW")
    print("=" * 70)

    drift_ok = run_drift_analysis()
    drift_detected = detect_drift() if drift_ok else False
    new_data_present = NEW_DATA_PATH.exists()
    new_data_pending = has_pending_batch()

    print(f"\n[INFO] Monitoring Evaluation:")
    print(f"  -> Drift Detected    : {drift_detected}")
    print(f"  -> New Batch Present : {new_data_present}")
    print(f"  -> New Batch Pending : {new_data_pending}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_analysis_success": drift_ok,
        "drift_detected": drift_detected,
        "new_data_present": new_data_present,
        "new_data_pending": new_data_pending,
        "pipeline_called": False,
        "new_data_ingested": False,
        "retraining_executed": False,
        "status": "idle"
    }

    if not drift_ok:
        print("[ERROR] Drift analysis failed; refusing to trigger retraining.")
        report["status"] = "drift_analysis_failed"
    elif not new_data_pending and drift_detected:
        print("[WARN] Drift detected, but no un-ingested dataset batch is available.")
        report["status"] = "waiting_for_new_data"
    elif not new_data_pending and new_data_present:
        print("[INFO] Incoming batch was already ingested. No retraining is needed.")
        report["status"] = "no_new_batch"
    elif not drift_detected and not new_data_pending:
        print("[INFO] No drift detected and no incoming batch present. System operating normally.")
        report["status"] = "normal_no_drift"
    else:
        print("[INFO] Conditions satisfied! Triggering News Classification MLOps Pipeline...")
        report["pipeline_called"] = True
        pipe_result = subprocess.run([sys.executable, str(PIPELINE_SCRIPT)], cwd=PROJECT_ROOT)

        load_status = get_data_load_status()
        report["pipeline_exit_code"] = pipe_result.returncode
        report["data_load_status"] = load_status
        report["new_data_ingested"] = (load_status == "ingested")
        report["retraining_executed"] = (pipe_result.returncode == 0 and load_status == "ingested")
        if pipe_result.returncode != 0:
            report["status"] = "pipeline_failed"
        elif load_status == "ingested":
            report["status"] = "pipeline_success"
        elif load_status in {"already_ingested", "no_new_data"}:
            report["status"] = "pipeline_no_new_data"
        else:
            report["status"] = "pipeline_incomplete"

    os.makedirs(TRIGGER_REPORT.parent, exist_ok=True)
    with open(TRIGGER_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 70)
    print(f"MONITOR TRIGGER STATUS: {report['status']}")
    print(f"Summary saved to: {TRIGGER_REPORT}")
    print("=" * 70)
    return report["status"] not in {"drift_analysis_failed", "pipeline_failed", "pipeline_incomplete"}


if __name__ == "__main__":
    run_monitor_trigger()
