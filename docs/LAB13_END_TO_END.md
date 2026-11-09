# Lab 13 — End-to-End MLOps Integration Report

## 1. Objective

Integrate data ingestion, schema validation, reproducible preprocessing, MLflow tracking, quality gates, automated lifecycle management, serving, drift monitoring, and closed-loop retraining triggers into one unified platform.

---

## 2. Closed-Loop Monitoring & Retraining Architecture

```text
Scheduled Trigger (GitHub Actions / Local Timer)
                    │
                    ▼
     Run Drift Analysis (Stage 12)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Analysis Failed      Analysis Passed
          │                   │
          ▼                   ▼
   Fail with report     Hash incoming batch and
                        compare with data lineage
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
             No unseen batch       Unseen batch
                     │                   │
                     ▼                   ▼
        Wait if drift detected;    01 Calculate Hash & Ingest
        otherwise report idle              │
                                            ▼
                                    02 Raw Schema Validation
                                            │
                                            ▼
                                    03 Preprocessing Pipeline
                                            │
                                            ▼
                                    04 Output Matrix Validation
                                            │
                                            ▼
                                    05 Train Candidate
                                            │
                                            ▼
                                    06 Model Evaluation
                                            │
                                            ▼
                                    07 Quality Gate (F1 >= 0.45)
                                            │
                                            ▼
                                    08 Champion-Challenger Face-Off
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                       Not Promoted                    Promoted
                              │                           │
                              ▼                           ▼
                       Keep Champion            Prepare Deployment Bundle;
                                                Update Container/Rollout
```

---

## 3. Idempotent Ingestion Mechanics

The platform prevents redundant retraining:
1. When a batch is submitted, `01_load_data.py` calculates its SHA-256 fingerprint.
2. It queries `dataset_lineage.json`.
3. If the batch hash already exists, the pipeline halts immediately with status `already_ingested`.
4. Only fresh, unconsumed batches trigger model retraining and evaluation.

The Lab 13 monitor applies the same hash check before launching the training pipeline. An existing `new_news_category.json` file is not, on its own, evidence of new data: if its hash is already in `dataset_lineage.json`, retraining is skipped. The monitor reports `waiting_for_new_data` when drift is present but no unseen batch is available. A failed drift-analysis subprocess or training pipeline returns a failure status instead of reporting success.
