import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from src.validation import validate_schema


def run_data_validation(output_report_name="baseline_validation.csv"):
    print("[INFO] Starting Raw Data Validation...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    articles = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    df = pd.DataFrame(articles)
    print(f"[INFO] Validating schema across {len(df)} records...")

    is_valid, failures = validate_schema(df, strict=False, output_report_name=output_report_name)

    if is_valid:
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
    else:
        print(f"[ERROR] Schema Validation FAILED with {len(failures)} failures.")

    return is_valid


if __name__ == "__main__":
    passed = run_data_validation()
    sys.exit(0 if passed else 1)
