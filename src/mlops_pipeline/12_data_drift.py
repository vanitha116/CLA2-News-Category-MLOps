import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_data_drift():
    print("[INFO] Running Data Drift Analysis...")
    os.makedirs("outputs/drift", exist_ok=True)
    report_path = Path("outputs/drift/data_drift_report.csv")

    baseline_csv = Path("data/processed/X_test_raw.csv")
    new_data_path = Path("data/raw/new_news_category.json")

    if not baseline_csv.exists():
        print(f"[ERROR] Baseline data not found at {baseline_csv}")
        return False

    base_df = pd.read_csv(baseline_csv)

    # If new_news_category.json exists, read it; otherwise simulate evolved data
    if new_data_path.exists():
        articles = [json.loads(line) for line in open(new_data_path, "r", encoding="utf-8") if line.strip()]
        new_df = pd.DataFrame(articles)
        new_df["combined_text"] = new_df["headline"].fillna("") + " " + new_df["short_description"].fillna("")
    else:
        # Simulate slight text distribution shift for testing
        new_df = base_df.sample(frac=0.5, random_state=123).copy()
        new_df["combined_text"] = new_df["combined_text"] + " breaking updates live alerts"

    # Compute text length and word count distributions
    base_lens = base_df["combined_text"].str.len().values
    new_lens = new_df["combined_text"].str.len().values

    base_words = base_df["combined_text"].str.split().str.len().values
    new_words = new_df["combined_text"].str.split().str.len().values

    # KS Test
    ks_len_stat, ks_len_p = ks_2samp(base_lens, new_lens)
    ks_word_stat, ks_word_p = ks_2samp(base_words, new_words)

    threshold = 0.05
    len_drift = "DRIFT" if (ks_len_stat > threshold and ks_len_p < 0.05) else "NO_DRIFT"
    word_drift = "DRIFT" if (ks_word_stat > threshold and ks_word_p < 0.05) else "NO_DRIFT"

    report_df = pd.DataFrame([
        {"feature": "text_character_length", "ks_statistic": round(float(ks_len_stat), 6), "p_value": round(float(ks_len_p), 6), "status": len_drift},
        {"feature": "text_word_count", "ks_statistic": round(float(ks_word_stat), 6), "p_value": round(float(ks_word_p), 6), "status": word_drift}
    ])

    report_df.to_csv(report_path, index=False)
    print("\n--- Data Drift Report ---")
    print(report_df.to_string(index=False))
    print(f"\n[SUCCESS] Data drift report saved to {report_path}.")
    return True


if __name__ == "__main__":
    calculate_data_drift()
