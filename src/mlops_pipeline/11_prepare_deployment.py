import json
import os
import shutil
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def prepare_deployment_bundle():
    print("[INFO] ==================================================")
    print("[INFO] Starting Stage 11: Deployment Artifact Preparation")
    print("[INFO] ==================================================")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()

    model_name = config["registry"]["model_name"]
    deployment_dir = "deployment"
    os.makedirs(deployment_dir, exist_ok=True)

    versions = client.search_model_versions(f"name='{model_name}'")
    prod_version = next((v for v in versions if v.current_stage == "Production"), None)

    if not prod_version:
        print("[WARN] No model currently in 'Production'. Using latest version for deployment bundle.")
        prod_version = sorted(versions, key=lambda v: int(v.version))[-1] if versions else None

    if not prod_version:
        print("[ERROR] No models found in registry!")
        return False

    v_num = prod_version.version
    run_id = prod_version.run_id
    run = client.get_run(run_id)

    print(f"[INFO] Packaging Production Model Version {v_num} (Run ID: {run_id})...")

    # 1. Save model into deployment/
    model_uri = f"models:/{model_name}/{v_num}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        model = joblib.load(config["paths"]["model"])

    target_model_path = os.path.join(deployment_dir, "model.pkl")
    joblib.dump(model, target_model_path)

    # 2. Save preprocessor into deployment/
    src_prep = config["paths"]["preprocessor"]
    target_prep_path = os.path.join(deployment_dir, "preprocessor.pkl")
    if os.path.exists(src_prep):
        shutil.copy2(src_prep, target_prep_path)

    # 3. Create deployment metadata JSON
    metadata = {
        "model_name": model_name,
        "version": int(v_num),
        "stage": prod_version.current_stage,
        "run_id": run_id,
        "metrics": run.data.metrics,
        "parameters": run.data.params,
        "deployment_artifacts": ["model.pkl", "preprocessor.pkl"]
    }

    meta_path = os.path.join(deployment_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Deployment bundle prepared in '{deployment_dir}/'!")
    print(f"  -> Model: {target_model_path}")
    print(f"  -> Preprocessor: {target_prep_path}")
    print(f"  -> Metadata: {meta_path}")
    return True


if __name__ == "__main__":
    success = prepare_deployment_bundle()
    import sys
    sys.exit(0 if success else 1)
