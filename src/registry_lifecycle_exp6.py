import os
import json
import time
import joblib
import numpy as np
import scipy.sparse as sp
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    MODEL_REGISTRY_NAME, PROCESSED_DATA_DIR,
    MODELS_DIR, ARTIFACTS_DIR, RANDOM_STATE
)

def setup_registry_client():
    """Initializes MLflow tracking URI and returns client."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return MlflowClient()

def train_and_register_candidate(version_name: str, model_type: str, model_kwargs: dict):
    """
    Trains a model candidate on the processed dataset, logs the run,
    attaches the preprocessor dependency, and registers it to MLflow Model Registry.
    """
    client = setup_registry_client()
    print(f"\n[REGISTRY] Training and registering candidate: {version_name} ({model_type})...")

    X_train = sp.load_npz(PROCESSED_DATA_DIR / "X_train_final.npz")
    X_test = sp.load_npz(PROCESSED_DATA_DIR / "X_test_final.npz")
    y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy", allow_pickle=True)
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy", allow_pickle=True)

    if model_type == "LogisticRegression":
        model = LogisticRegression(**model_kwargs)
    elif model_type == "LinearSVC":
        model = LinearSVC(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    }

    with mlflow.start_run(run_name=version_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(model_kwargs)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)

        # Attach preprocessing dependency to model run artifacts
        prep_path = MODELS_DIR / "preprocessor_pipeline.pkl"
        if os.path.exists(prep_path):
            mlflow.log_artifact(str(prep_path), artifact_path="preprocessing_dependency")

        # Register model to MLflow Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )

        print(f"[REGISTRY] Logged candidate Run ID: {run_id}")
        print(f"  -> Metrics: Accuracy={metrics['test_accuracy']:.4f} | F1={metrics['f1_weighted']:.4f}")

    return run_id, metrics

def manage_lifecycle_champion_challenger(primary_metric: str = "f1_weighted"):
    """
    Automated Lifecycle Management:
    1. Moves any newly registered versions to Staging.
    2. Identifies the Best Challenger in Staging based on primary_metric.
    3. Conducts automated Face-Off between Challenger and current Production Champion.
    4. Automatically promotes superior Challenger to Production and archives older Champion.
    """
    print("\n" + "="*80)
    print("MODEL REGISTRY: AUTOMATED CHAMPION-CHALLENGER LIFECYCLE MANAGEMENT")
    print("="*80)

    client = setup_registry_client()

    # Retrieve all versions of the registered model
    all_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    if not all_versions:
        print(f"[ERROR] No model versions found in registry under '{MODEL_REGISTRY_NAME}'.")
        return

    # 1. Discover newly registered models (Stage: None) and transition to Staging
    new_versions = [v for v in all_versions if v.current_stage == "None"]
    if new_versions:
        print(f"[LIFECYCLE] Found {len(new_versions)} new version(s) in 'None' stage. Moving to 'Staging'...")
        for v in new_versions:
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=v.version,
                stage="Staging",
                archive_existing_versions=False
            )
            print(f"  -> Version {v.version} transitioned to 'Staging'.")

    # Refresh versions
    all_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    staging_versions = [v for v in all_versions if v.current_stage == "Staging"]
    production_versions = [v for v in all_versions if v.current_stage == "Production"]

    current_champion = production_versions[0] if production_versions else None

    if not staging_versions:
        print("[LIFECYCLE] No candidates in 'Staging' to evaluate.")
        if current_champion:
            print(f"[LIFECYCLE] Current Production Champion is Version {current_champion.version}.")
        return

    # 2. Evaluate Staging candidates to find Best Challenger
    print("\n[LIFECYCLE] Evaluating Staging candidates based on:", primary_metric)
    best_challenger = None
    best_score = -1.0

    for v in staging_versions:
        run = client.get_run(v.run_id)
        score = run.data.metrics.get(primary_metric, 0.0)
        print(f"  Candidate V{v.version} (Run: {run.info.run_name}) -> {primary_metric}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_challenger = v

    print(f"\n[LIFECYCLE] Best Challenger is Version {best_challenger.version} with {primary_metric}={best_score:.4f}")

    # 3. Champion vs Challenger Face-Off
    promote = False
    if current_champion is None:
        print("[LIFECYCLE] No active Champion in Production. Challenger promoted by default!")
        promote = True
    else:
        champ_run = client.get_run(current_champion.run_id)
        champ_score = champ_run.data.metrics.get(primary_metric, 0.0)
        print(f"[LIFECYCLE] Current Champion V{current_champion.version} has {primary_metric}={champ_score:.4f}")

        if best_score > champ_score:
            diff = best_score - champ_score
            print(f"[LIFECYCLE] [PROMOTION] Challenger V{best_challenger.version} OUTPERFORMS Champion by +{diff:.4f}!")
            promote = True
        else:
            print(f"[LIFECYCLE] [RETAINED] Champion V{current_champion.version} retains title. Challenger rejected.")

    # 4. Execute Promotion and Archival
    if promote:
        print(f"[LIFECYCLE] Promoting Version {best_challenger.version} to 'Production'...")
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=best_challenger.version,
            stage="Production",
            archive_existing_versions=False
        )
        # Tag version
        client.set_model_version_tag(
            name=MODEL_REGISTRY_NAME,
            version=best_challenger.version,
            key="deployment_status",
            value="CHAMPION_ACTIVE"
        )

        if current_champion:
            print(f"[LIFECYCLE] Archiving displaced Champion (Version {current_champion.version})...")
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=current_champion.version,
                stage="Archived",
                archive_existing_versions=False
            )

    # 5. Clean up remaining defeated Staging candidates
    updated_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    remaining_staging = [v for v in updated_versions if v.current_stage == "Staging"]
    for v in remaining_staging:
        print(f"[LIFECYCLE] Archiving unpromoted candidate Version {v.version}...")
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=v.version,
            stage="Archived",
            archive_existing_versions=False
        )

    print("[LIFECYCLE] [SUCCESS] Lifecycle transitions completed!")

def rollback_model_version(target_version: int):
    """
    Rollback Management: Restores a specified previous version to Production
    and archives the currently failing production version.
    """
    print(f"\n[ROLLBACK] Initiating rollback to Version {target_version}...")
    client = setup_registry_client()
    versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")

    current_prod = next((v for v in versions if v.current_stage == "Production"), None)
    target = next((v for v in versions if int(v.version) == int(target_version)), None)

    if not target:
        print(f"[ROLLBACK] [ERROR] Target version {target_version} does not exist!")
        return False

    if current_prod:
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=current_prod.version,
            stage="Archived"
        )
        print(f"[ROLLBACK] Current Production V{current_prod.version} demoted to 'Archived'.")

    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=str(target_version),
        stage="Production"
    )
    client.set_model_version_tag(
        name=MODEL_REGISTRY_NAME,
        version=str(target_version),
        key="rollback_event",
        value=f"Reinstated on {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"[ROLLBACK] [SUCCESS] Version {target_version} successfully reinstated to 'Production'!")
    return True

def generate_production_registry_report():
    """
    Exports a comprehensive production deployment report ensuring complete traceability.
    """
    print("\n--- Generating Production Model Registry Report ---")
    client = setup_registry_client()
    versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    champion = next((v for v in versions if v.current_stage == "Production"), None)

    if not champion:
        print("[ERROR] No active model found in 'Production' stage.")
        return

    run = client.get_run(champion.run_id)

    report = {
        "registry_status": "READY_FOR_DEPLOYMENT",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_lineage": {
            "registered_name": MODEL_REGISTRY_NAME,
            "version": int(champion.version),
            "current_stage": champion.current_stage,
            "run_id": champion.run_id,
            "artifact_uri": champion.source,
            "model_type": run.data.params.get("model_type")
        },
        "performance_metrics": {
            "test_accuracy": run.data.metrics.get("test_accuracy"),
            "f1_weighted": run.data.metrics.get("f1_weighted"),
            "precision_weighted": run.data.metrics.get("precision_weighted"),
            "recall_weighted": run.data.metrics.get("recall_weighted")
        },
        "hyperparameters": run.data.params,
        "preprocessing_dependency": "models/preprocessor_pipeline.pkl",
        "dataset_lineage": "News_Category_Dataset_v3"
    }

    report_path = ARTIFACTS_DIR / "production_model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[REPORT] [SUCCESS] Production model report generated for Version {champion.version}.")
    print(f"  -> File: {report_path}")
    return report

def run_experiment_6_suite():
    """
    Executes Experiment 6:
    1. Registers candidate models (V1: Logistic Regression, V2: LinearSVC Champion)
    2. Runs automated Champion-Challenger evaluation
    3. Demonstrates lifecycle stage transition (Staging -> Production -> Archived)
    4. Demonstrates Rollback capability
    5. Exports Production Model Registry Report
    """
    print("==================================================")
    print("EXPERIMENT 6: MODEL REGISTRY & LIFECYCLE MANAGEMENT")
    print("==================================================")

    # Candidate 1: Logistic Regression Baseline
    train_and_register_candidate(
        version_name="Candidate_LogReg_V1",
        model_type="LogisticRegression",
        model_kwargs={"C": 1.0, "max_iter": 500, "random_state": RANDOM_STATE}
    )

    # Candidate 2: LinearSVC Tuned Challenger
    train_and_register_candidate(
        version_name="Candidate_LinearSVC_V2",
        model_type="LinearSVC",
        model_kwargs={"C": 0.5, "class_weight": "balanced", "random_state": RANDOM_STATE}
    )

    # Run Automated Champion vs Challenger Lifecycle Management
    manage_lifecycle_champion_challenger(primary_metric="f1_weighted")

    # Generate Audit Report
    report = generate_production_registry_report()
    return report

if __name__ == "__main__":
    run_experiment_6_suite()
