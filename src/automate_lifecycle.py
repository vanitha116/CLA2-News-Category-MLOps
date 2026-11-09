import json
import os
import mlflow
from mlflow.tracking import MlflowClient
import yaml


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def automate_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()

    model_name = config["registry"]["model_name"]
    metric_name = config["registry"]["decision_metric"]
    lifecycle_report_path = config["paths"]["lifecycle_report"]

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            print(f"[ERROR] No versions found for registered model: '{model_name}'.")
            return False

        # 1. Discover untracked models (Stage: None) and push to Staging
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]
        if new_models:
            print(f"[INFO] Found {len(new_models)} new model(s). Moving to Staging...")
            for mv in new_models:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                print(f"  -> Version {mv.version} moved to Staging.")

        # Refresh
        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        current_champion = production_models[0] if production_models else None

        if not staging_models:
            print("[INFO] No models in Staging to evaluate.")
            if current_champion:
                print(f"[INFO] Production Champion remains Version {current_champion.version}.")
            return True

        # 2. Evaluate Staging Models (Find Best Challenger)
        print(f"\n[INFO] Evaluating Staging candidates based on '{metric_name}'...")
        best_challenger = None
        best_score = -1.0

        for mv in staging_models:
            run = client.get_run(mv.run_id)
            score = float(run.data.metrics.get(metric_name, 0.0))
            print(f"  -> Staging Candidate V{mv.version} | {metric_name}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_challenger = mv

        print(f"[INFO] Best Challenger: Version {best_challenger.version} ({metric_name}={best_score:.4f})")

        # 3. Face-Off with Current Champion
        promote = False
        champ_score = None
        if not current_champion:
            print("[INFO] No current Production model. Challenger promoted by default!")
            promote = True
        else:
            champ_run = client.get_run(current_champion.run_id)
            champ_score = float(champ_run.data.metrics.get(metric_name, 0.0))
            print(f"[INFO] Current Production Champion V{current_champion.version} ({metric_name}={champ_score:.4f})")

            if best_score > champ_score:
                print(f"[SUCCESS] Challenger beat Champion! (+{best_score - champ_score:.4f})")
                promote = True
            else:
                print("[INFO] Challenger did not beat Champion. Champion retains title.")

        # 4. Execute Promotion and Archival
        if promote:
            print(f"[INFO] Promoting Version {best_challenger.version} to Production...")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=False
            )
            client.set_model_version_tag(
                name=model_name, version=best_challenger.version, key="status", value="CHAMPION_ACTIVE"
            )

            if current_champion:
                print(f"[INFO] Archiving superseded Champion Version {current_champion.version}...")
                client.transition_model_version_stage(
                    name=model_name, version=current_champion.version, stage="Archived", archive_existing_versions=False
                )

        # 5. Clean up remaining Staging models
        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining_staging = [mv for mv in final_versions if mv.current_stage == "Staging"]
        for mv in remaining_staging:
            print(f"[INFO] Archiving rejected candidate Version {mv.version}...")
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False
            )

        # 6. Save Lifecycle Report
        os.makedirs(os.path.dirname(lifecycle_report_path), exist_ok=True)
        report = {
            "stage": "Lab 6/8 - Model Lifecycle Management",
            "model_name": model_name,
            "decision_metric": metric_name,
            "promoted": promote,
            "candidate_version": int(best_challenger.version) if best_challenger else None,
            "candidate_score": best_score,
            "production_version_before": int(current_champion.version) if current_champion else None,
            "production_score_before": champ_score,
            "production_version_after": int(best_challenger.version) if promote else (int(current_champion.version) if current_champion else None)
        }

        with open(lifecycle_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Lifecycle Report written to {lifecycle_report_path}.")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to automate model lifecycle: {e}")
        return False


if __name__ == "__main__":
    automate_lifecycle()
