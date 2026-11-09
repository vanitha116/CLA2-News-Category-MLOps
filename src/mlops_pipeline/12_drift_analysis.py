import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_sub_drift(script_name):
    script_path = PROJECT_ROOT / "src" / "mlops_pipeline" / script_name
    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Drift analysis step failed: {script_name} (exit code {result.returncode})")


def main():
    print("=" * 70)
    print("STAGE 12: CONSOLIDATED DRIFT ANALYSIS")
    print("=" * 70)

    # Run sub-components
    run_sub_drift("12_data_drift.py")
    run_sub_drift("12_prediction_drift.py")
    run_sub_drift("12_concept_drift.py")

    reports_dir = PROJECT_ROOT / "outputs" / "drift"
    data_drift_p = reports_dir / "data_drift_report.csv"
    pred_drift_p = reports_dir / "prediction_drift_report.csv"
    concept_drift_p = reports_dir / "concept_drift_report.csv"

    rows = []

    if data_drift_p.exists():
        df_d = pd.read_csv(data_drift_p)
        for _, r in df_d.iterrows():
            rows.append({"drift_type": "Data Drift", "component": r["feature"], "status": r["status"]})

    if pred_drift_p.exists():
        df_p = pd.read_csv(pred_drift_p)
        for _, r in df_p.iterrows():
            rows.append({"drift_type": "Prediction Drift", "component": r["target"], "status": r["status"]})

    if concept_drift_p.exists():
        df_c = pd.read_csv(concept_drift_p)
        for _, r in df_c.iterrows():
            rows.append({"drift_type": "Concept Drift", "component": r["metric"], "status": r["status"]})

    summary_df = pd.DataFrame(rows)
    summary_path = reports_dir / "drift_analysis_report.csv"
    summary_df.to_csv(summary_path, index=False)

    has_drift = (summary_df["status"].str.upper() == "DRIFT").any() if not summary_df.empty else False

    print("\n" + "=" * 70)
    print("CONSOLIDATED DRIFT SUMMARY:")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"Overall Drift Detected: {has_drift}")
    print(f"Report exported to: {summary_path}")
    return has_drift


if __name__ == "__main__":
    main()
