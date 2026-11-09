# News Category Classification MLOps Platform — Architecture

## 1. System Overview

This platform implements an enterprise-grade Machine Learning Operations (MLOps) system for automated News Category Classification, covering the entire lifecycle across **Labs 1 to 13**:
- Data ingestion, lineage hashing, and version control
- Strict schema validation and data quality checks (Pandera)
- Reproducible, idempotent text preprocessing pipelines
- Experiment tracking, parameter/metric logging, and reproducibility (MLflow)
- Model registry and automated Champion-Challenger lifecycle decisions
- Quality gate enforcement guarding promotion
- Failure handling and self-healing recovery
- RESTful microservice model serving with FastAPI
- Containerization with Docker and orchestration with Kubernetes
- Continuous drift monitoring (data drift, prediction drift, concept drift)
- Closed-loop automated retraining triggers via GitHub Actions

---

## 2. End-to-End Workflow Architecture

```text
Incoming News Batch (new_news_category.json)
              │
              ▼
01 Data Loading & SHA-256 Hash Ingestion
              │
              ├── Hash already exists? ──> [HALT / Skip Duplicate Batch]
              └── New batch verified
              │
              ▼
02 Raw Data Validation (Pandera Schema Guardrails)
              │
              ▼
03 Preprocessing Pipeline (TextCleaner + TF-IDF)
              │
              ▼
04 Processed Output Validation (Matrix Sanity & Idempotence)
              │
              ▼
05 Training + MLflow Tracking + Model Registry (Candidate V2)
              │
              ▼
06 Model Evaluation (Accuracy, Precision, Recall, F1)
              │
              ▼
07 Quality Gate Enforcement (F1 >= 0.45)
              │
              ├── Fails Gate? ──> [HALT / Reject Candidate]
              └── Passes Gate
              │
              ▼
08 Automated Model Lifecycle (Champion vs. Challenger)
              │
        ┌─────┴──────────────────┐
        │                        │
Candidate <= Champion     Candidate > Champion
        │                        │
        ▼                        ▼
Reject Candidate /        Promote Candidate to 'Production'
Archive Challenger        Archive Previous Champion
        │                        │
        │                        ▼
        │                 Prepare Deployment Bundle
        │                        │
        │                        ▼
        │                 Docker Image & Kubernetes Rollout
        ▼                        ▼
09 / 10 Production Serving & Real-Time Inference (FastAPI)
```

---

## 3. Component Responsibilities

| Stage | Script / Module | Purpose |
| :--- | :--- | :--- |
| **Lab 1** | `src/mlops_pipeline/01_load_data.py` | Ingests new batches, calculates SHA-256 hashes, archives prior versions, prevents duplicate ingestion. |
| **Lab 2** | `src/mlops_pipeline/02_validate_data.py` | Validates data against Pandera schema (min length, 42 valid categories, valid URLs). |
| **Lab 3** | `src/preprocess.py`, `train.py`, `evaluate.py` | Modular baseline pipeline with Git workflow documentation (`docs/GIT_WORKFLOW.md`). |
| **Lab 4** | `src/train_mlflow.py` | Tracks parameters, metrics, confusion matrices, and verifies reproducibility. |
| **Lab 5** | `src/preprocess_pipeline.py`, `04_validate_outputs.py` | Scikit-Learn pipeline, matrix validation, idempotence proof. |
| **Lab 6** | `src/train_registry.py`, `08_automate_lifecycle.py` | Pushes candidate to Model Registry, conducts Champion vs Challenger evaluation. |
| **Lab 7** | `src/mlops_pipeline/news_classification_pipeline.py` | Master sequential pipeline executing stages 01 through 09. |
| **Lab 8** | `src/mlops_pipeline/07_quality_gate.py` | Quality Gate requirement: candidate must achieve configured F1 minimum threshold. |
| **Lab 9** | `src/mlops_pipeline/10_failure_injection.py` | Injects 5 deliberate failure modes and validates self-healing recovery. |
| **Lab 10** | `api/main.py`, `api/schemas.py` | FastAPI application offering `/health`, `/metadata`, `/predict`, `/predict/batch`. |
| **Lab 11** | `deployment/deploy_production.py`, `k8s/` | Docker containerization, Kubernetes Deployment and Service manifests. |
| **Lab 12** | `src/mlops_pipeline/12_drift_analysis.py` | Evaluates data drift, prediction drift, and concept degradation. |
| **Lab 13** | `src/mlops_pipeline/12_monitor_trigger.py` | Scheduled closed-loop automation linking monitoring to retraining. |
