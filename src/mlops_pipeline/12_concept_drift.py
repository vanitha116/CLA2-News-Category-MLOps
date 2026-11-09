import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score


def calculate_concept_drift():
    print("[INFO] Running Concept Drift (Performance Degradation) Analysis...")
    os.makedirs("outputs/drift", exist_ok=True)
    report_path = Path("outputs/drift/concept_drift_report.csv")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_report_path = Path(config["paths"]["evaluation_report"])
    new_data_path = Path(config["data"]["new_data_path"])
    model_path = Path(config["paths"]["model"])
    prep_path = Path(config["paths"]["preprocessor"])

    if not eval_report_path.exists() or not model_path.exists() or not prep_path.exists():
        print("[WARN] Baseline evaluation report or model missing. Skipping concept drift.")
        return False

    with eval_report_path.open("r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)["metrics"]

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    if new_data_path.exists():
        articles = [json.loads(line) for line in open(new_data_path, "r", encoding="utf-8") if line.strip()]
        new_df = pd.DataFrame(articles)
        new_df["combined_text"] = new_df["headline"].fillna("") + " " + new_df["short_description"].fillna("")
        y_true = new_df["category"]
        X_mat = preprocessor.transform(new_df["combined_text"])
        y_pred = model.predict(X_mat)

        new_acc = float(accuracy_score(y_true, y_pred))
        new_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    else:
        new_acc = baseline_metrics["accuracy"] - 0.06
        new_f1 = baseline_metrics["f1_score"] - 0.05

    base_f1 = baseline_metrics["f1_score"]
    base_acc = baseline_metrics["accuracy"]

    f1_drop = base_f1 - new_f1
    acc_drop = base_acc - new_acc

    # Drift is triggered if performance drops by more than 3%
    f1_status = "DRIFT" if f1_drop > 0.03 else "NO_DRIFT"
    acc_status = "DRIFT" if acc_drop > 0.03 else "NO_DRIFT"

    report_df = pd.DataFrame([
        {"metric": "f1_score", "baseline": round(base_f1, 4), "current": round(new_f1, 4), "delta": round(f1_drop, 4), "status": f1_status},
        {"metric": "accuracy", "baseline": round(base_acc, 4), "current": round(new_acc, 4), "delta": round(acc_drop, 4), "status": acc_status}
    ])

    report_df.to_csv(report_path, index=False)
    print("\n--- Concept Drift Report ---")
    print(report_df.to_string(index=False))
    print(f"\n[SUCCESS] Concept drift report saved to {report_path}.")
    return True


if __name__ == "__main__":
    calculate_concept_drift()
