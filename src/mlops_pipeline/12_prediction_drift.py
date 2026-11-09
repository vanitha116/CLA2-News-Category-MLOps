import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ks_2samp
import yaml


def calculate_prediction_drift():
    print("[INFO] Running Prediction Drift Analysis...")
    os.makedirs("outputs/drift", exist_ok=True)
    report_path = Path("outputs/drift/prediction_drift_report.csv")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model"]
    prep_path = config["paths"]["preprocessor"]
    base_raw_path = Path("data/processed/X_test_raw.csv")
    new_data_path = Path(config["data"]["new_data_path"])

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        print(f"[ERROR] Model or preprocessor missing.")
        return False

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    base_df = pd.read_csv(base_raw_path)
    base_preds = model.predict(preprocessor.transform(base_df["combined_text"]))

    if new_data_path.exists():
        new_articles = [json.loads(line) for line in open(new_data_path, "r", encoding="utf-8") if line.strip()]
        new_df = pd.DataFrame(new_articles)
        new_df["combined_text"] = new_df["headline"].fillna("") + " " + new_df["short_description"].fillna("")
    else:
        new_df = base_df.sample(frac=0.5, random_state=42).copy()
        new_df["combined_text"] = new_df["combined_text"] + " technology artificial intelligence breakthrough"

    new_preds = model.predict(preprocessor.transform(new_df["combined_text"]))

    # Map categories to integer indices for distribution comparison
    all_cats = sorted(list(set(base_preds) | set(new_preds)))
    cat_to_id = {c: i for i, c in enumerate(all_cats)}

    base_ids = np.array([cat_to_id[c] for c in base_preds])
    new_ids = np.array([cat_to_id[c] for c in new_preds])

    ks_stat, p_val = ks_2samp(base_ids, new_ids)
    drift_status = "DRIFT" if (ks_stat > 0.08 and p_val < 0.05) else "NO_DRIFT"

    report_df = pd.DataFrame([
        {
            "target": "predicted_category_distribution",
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_val), 6),
            "status": drift_status
        }
    ])

    report_df.to_csv(report_path, index=False)
    print("\n--- Prediction Drift Report ---")
    print(report_df.to_string(index=False))
    print(f"\n[SUCCESS] Prediction drift report saved to {report_path}.")
    return True


if __name__ == "__main__":
    calculate_prediction_drift()
