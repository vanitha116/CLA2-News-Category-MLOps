# Lab 9 — Pipeline Failure Handling and Self-Healing Recovery

## Goal

Deliberately inject failures across multiple pipeline stages and verify that the system halts safely or self-heals by recovering the Production model from the MLflow Model Registry.

## Run Failure Injection Suite

```bash
python pipelines/run_lab9_failure_handling.py
```

## Failure Scenarios Tested

| Scenario | Injected Defect | Expected Behavior |
| :--- | :--- | :--- |
| **Scenario 1** | Missing raw dataset (`news_category.json` renamed) | **HALT** with non-zero exit code |
| **Scenario 2** | Corrupted records (numeric headline, invalid category, malformed date) | **HALT** at Pandera schema validation |
| **Scenario 3** | Missing processed output matrix (`X_train_final.npz` moved) | **HALT** at output matrix validation |
| **Scenario 4** | Missing local model artifact (`models/news_model.pkl` deleted) | **DETECT & SELF-HEAL**: Model evaluation detects missing artifact, queries MLflow Model Registry, downloads active Production model, restores artifact, and completes evaluation |

The audit report is automatically saved to `reports/failure_report.json`.
