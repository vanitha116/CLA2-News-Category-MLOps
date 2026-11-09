import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from src.validation import validate_schema


def validate_raw_data():
    print("[INFO] Starting Stage 02: Raw Data Validation...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        print(f"[ERROR] Target raw file not found at: {raw_path}")
        return False

    articles = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    df = pd.DataFrame(articles)
    print(f"[INFO] Validating schema across {len(df)} records in {raw_path}...")

    is_valid, failures = validate_schema(df, strict=False, output_report_name="baseline_validation.csv")

    if is_valid:
        print("[SUCCESS] Raw Data Validation PASSED. Dataset is clean.")
        return True
    else:
        print(f"[ERROR] Raw Data Validation FAILED with {len(failures)} failures.")
        return False


if __name__ == "__main__":
    passed = validate_raw_data()
    sys.exit(0 if passed else 1)
