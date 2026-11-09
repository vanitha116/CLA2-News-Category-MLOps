import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from src.validation import validate_schema


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_lineage(lineage_path):
    if not lineage_path.exists():
        return []
    try:
        with lineage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_lineage(lineage_path, lineage):
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=4)


def get_next_archive_version(archive_dir):
    versions = []
    for p in archive_dir.glob("news_category_v*.json"):
        try:
            num = int(p.stem.replace("news_category_v", ""))
            versions.append(num)
        except ValueError:
            continue
    return max(versions, default=0) + 1


def load_data():
    print("[INFO] ==================================================")
    print("[INFO] Starting Lab 1: Data Ingestion & Lineage Tracking")
    print("[INFO] ==================================================")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = Path(config["data"]["raw_path"])
    new_data_path = Path(config["data"]["new_data_path"])
    archive_dir = Path(config["data"]["archive_dir"])
    lineage_path = archive_dir / "dataset_lineage.json"

    os.makedirs("artifacts", exist_ok=True)
    report_path = Path("artifacts/data_load_report.json")

    lineage = load_lineage(lineage_path)

    # If raw baseline does not have an initial lineage record, create it
    if raw_path.exists() and not lineage:
        base_hash = calculate_file_hash(raw_path)
        with open(raw_path, "r", encoding="utf-8") as f:
            base_count = sum(1 for line in f if line.strip())
        lineage.append({
            "version": 1,
            "filename": raw_path.name,
            "sha256": base_hash,
            "records": base_count,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "batch_type": "baseline"
        })
        save_lineage(lineage_path, lineage)
        print(f"[INFO] Initialized baseline dataset lineage (v1, {base_count} records).")

    if not new_data_path.exists():
        print("[INFO] No new batch (new_news_category.json) present.")
        report = {
            "status": "no_new_data",
            "message": "No new dataset batch found to ingest.",
            "current_records": sum(1 for line in open(raw_path, "r", encoding="utf-8") if line.strip()) if raw_path.exists() else 0
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        return True

    batch_hash = calculate_file_hash(new_data_path)
    print(f"[INFO] Detected new incoming batch: {new_data_path}")
    print(f"[INFO] Batch SHA-256: {batch_hash}")

    # Check for duplicate batch ingestion
    existing_hashes = {entry.get("sha256") for entry in lineage}
    if batch_hash in existing_hashes:
        print("[WARN] [IDEMPOTENCE] Batch has already been ingested in a previous run. Skipping re-ingestion.")
        report = {
            "status": "already_ingested",
            "batch_hash": batch_hash,
            "message": "Duplicate batch detected. Retraining skipped."
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        return True

    # Validate schema of incoming batch
    new_articles = []
    with open(new_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                new_articles.append(json.loads(line))

    new_df = pd.DataFrame(new_articles)
    print(f"[INFO] Validating schema of incoming {len(new_df)} records...")
    is_valid, failures = validate_schema(new_df, strict=False, output_report_name="new_data_validation.csv")

    if not is_valid:
        print(f"[ERROR] Incoming batch failed schema validation! {len(failures)} failures.")
        report = {
            "status": "validation_failed",
            "batch_hash": batch_hash,
            "failures": len(failures)
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        return False

    # Archive previous dataset
    next_ver = get_next_archive_version(archive_dir)
    archive_file = archive_dir / f"news_category_v{next_ver:02d}.json"
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, archive_file)
    print(f"[INFO] Archived previous dataset to {archive_file}.")

    # Append new batch to current raw dataset
    with open(raw_path, "a", encoding="utf-8") as f:
        for line in open(new_data_path, "r", encoding="utf-8"):
            if line.strip():
                f.write(line.strip() + "\n")

    current_count = sum(1 for line in open(raw_path, "r", encoding="utf-8") if line.strip())

    # Record lineage
    lineage.append({
        "version": next_ver + 1,
        "filename": new_data_path.name,
        "sha256": batch_hash,
        "new_records": len(new_df),
        "total_records": current_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "archived_as": archive_file.name
    })
    save_lineage(lineage_path, lineage)

    report = {
        "status": "ingested",
        "batch_hash": batch_hash,
        "new_records": len(new_df),
        "total_records": current_count,
        "version": next_ver + 1
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Successfully ingested new batch! Total dataset now: {current_count} articles.")
    return True


if __name__ == "__main__":
    success = load_data()
    sys.exit(0 if success else 1)
