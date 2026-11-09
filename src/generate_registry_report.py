import json
import os
import time
import mlflow
from mlflow.tracking import MlflowClient
import yaml


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def generate_registry_report():
    print("[INFO] Generating Model Registry Production Report...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in versions if mv.current_stage == "Production"), None)

        if not champion:
            print("[WARN] No model found in 'Production' stage. Using latest version for report.")
            champion = sorted(versions, key=lambda v: int(v.version))[-1] if versions else None

        if not champion:
            print("[ERROR] No models found in registry.")
            return False

        run = client.get_run(champion.run_id)

        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": run.data.metrics,
            "hyperparameters": run.data.params,
            "preprocessing_dependency": config["paths"]["preprocessor"]
        }

        os.makedirs("artifacts", exist_ok=True)
        report_path = config["paths"]["production_report"]

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Production report generated for Version {champion.version}.")
        print(f"[INFO] Report saved to: {report_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to generate registry report: {e}")
        return False


if __name__ == "__main__":
    generate_registry_report()
