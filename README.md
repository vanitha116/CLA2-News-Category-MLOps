# News Category Classification — End-to-End MLOps Platform (Labs 1 to 13)

A production-grade Machine Learning Operations (MLOps) platform for **News Category Classification** implementing the complete curriculum across **Labs 1 to 13**. The platform mirrors the design principles, fail-fast mechanics, and lifecycle management of the reference enterprise architecture.

---

## 1. Table of Contents

- [Platform Overview](#platform-overview)
- [Labs 1 to 13 Architecture Summary](#labs-1-to-13-architecture-summary)
- [Quick Start: Master Runner](#quick-start-master-runner)
- [Individual Lab Execution Guide](#individual-lab-execution-guide)
- [Model Serving and API Engineering (FastAPI)](#model-serving-and-api-engineering-fastapi)
- [MLflow Experiment Tracking & Model Registry](#mlflow-experiment-tracking--model-registry)
- [Failure Handling and Self-Healing Recovery](#failure-handling-and-self-healing-recovery)
- [Containerization & Kubernetes](#containerization--kubernetes)
- [Monitoring, Drift Analysis & Retraining](#monitoring-drift-analysis--retraining)
- [Repository Layout](#repository-layout)

---

## 2. Platform Overview

The system classifies digital news into **42 canonical categories** (Politics, Business, Tech, Science, Entertainment, Sports, etc.). Rather than a static script, it wraps the machine learning workflow inside an automated, production-ready MLOps platform:

- **Idempotent Ingestion**: SHA-256 fingerprinting prevents re-ingestion and redundant training on duplicate batches.
- **Strict Data Guardrails**: Schema enforcement with Pandera blocks corrupted records before vectorization.
- **Reproducible NLP Pipelines**: Scikit-Learn `Pipeline` wrapping regex text cleaning and TF-IDF transformations.
- **Traceable Tracking**: SQLite-backed MLflow records parameters, metrics, confusion matrices, and model packages.
- **Quality-Gated Promotion**: Candidate models must exceed minimum F1 thresholds (`F1 >= 0.45`) before qualifying for lifecycle comparison.
- **Automated Lifecycle Decisions**: Best Challenger in `Staging` faces off against the `Production` Champion; promotion occurs only when performance strictly improves.
- **Self-Healing Fault Recovery**: Missing local models are detected and automatically restored from the MLflow Model Registry.
- **Real-Time Serving**: FastAPI REST microservice with Pydantic validation, standardized 422/503 error handlers, sub-2ms latency, and request logging.
- **Continuous Monitoring**: Automatic detection of vocabulary drift, prediction distribution drift, and concept degradation.
- **Closed-Loop Retraining**: Scheduled triggers detect drift, ingest new batches, retrain candidates, and execute conditional deployment.

---

## 3. Labs 1 to 13 Architecture Summary

| Lab | Title | Description | Runner Script |
| :--- | :--- | :--- | :--- |
| **Lab 1** | **Data Ingestion & Versioning** | SHA-256 batch hashing, dataset snapshots, lineage tracking in `dataset_lineage.json`. | `pipelines/run_lab1_data_loading.py` |
| **Lab 2** | **Data Validation with Pandera** | Schema rules: headline length $\ge 3$, valid categories, ISO dates, HTTP URLs. | `pipelines/run_lab2_preprocessing.py` |
| **Lab 3** | **Baseline ML Pipeline & Git** | Modular preprocessing, training, evaluation, and Git branching/conflict docs. | `pipelines/run_lab3_baseline.py` |
| **Lab 4** | **Experiment Tracking** | MLflow logging of parameters, metrics, plots, and deterministic reproducibility checks. | `pipelines/run_lab4_tracking.py` |
| **Lab 5** | **Reproducible Pipelines** | Reusable `TextCleanerTransformer` + `TfidfVectorizer`, matrix quality checks. | `pipelines/run_lab5_pipeline.py` |
| **Lab 6** | **Model Registry & Lifecycle** | Stage transitions (`Staging`, `Production`, `Archived`), Champion-Challenger promotion. | `pipelines/run_lab6_registry.py` |
| **Lab 7** | **ML Pipeline Engineering** | Automated 10-stage sequential pipeline with fail-fast execution and run logs. | `pipelines/run_lab7_pipeline.py` |
| **Lab 8** | **CI & Quality Gate Automation** | Automated Quality Gate (`F1 >= 0.45`), GitHub Actions workflow (`lab8_ci.yml`). | `pipelines/run_lab8_ci.py` |
| **Lab 9** | **Failure Handling & Recovery** | 5 failure injection scenarios; self-healing recovery of model from MLflow. | `pipelines/run_lab9_failure_handling.py` |
| **Lab 10** | **Model Serving & API Engineering** | FastAPI service (`/health`, `/metadata`, `/predict`, `/predict/batch`), test suite. | `pipelines/run_lab10_api.py` |
| **Lab 11** | **Containerization & Deployment** | Docker image definition (`Dockerfile`), Kubernetes Deployment/Service manifests. | `pipelines/run_lab11_deployment.py` |
| **Lab 12** | **Continuous Drift Monitoring** | Data drift, prediction drift, concept degradation, candidate retraining. | `pipelines/run_lab12_monitoring.py` |
| **Lab 13** | **End-to-End Retraining Trigger** | Closed-loop monitoring trigger linking drift detection to automated retraining. | `pipelines/run_lab13_end_to_end.py` |

---

## 4. Quick Start: Master Runner

Create the project environment and install the full pipeline and test dependencies:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the test suite before launching workflows:

```powershell
.venv\Scripts\python.exe -m pytest
```

The automated tests use a small in-memory classifier; they do not need your datasets, MLflow registry, or trained model files. Training and monitoring workflows do require the raw dataset configured at `data.raw_path` in `config.yaml`. If you have the original dataset at the project root, copy it into the configured location before running those workflows:

```powershell
New-Item -ItemType Directory -Force data\raw
Copy-Item News_Category_Dataset_v3.json data\raw\news_category.json
```

Datasets, trained models, generated reports, and the local MLflow database are kept out of Git.

To execute the entire 13-Lab platform sequentially:

```powershell
.venv\Scripts\python.exe pipelines/run_all_labs.py
```

---

## 5. Individual Lab Execution Guide

Run any individual lab directly in your VS Code terminal:

```powershell
# Lab 1: Data Ingestion and Lineage
.venv\Scripts\python.exe pipelines/run_lab1_data_loading.py

# Lab 2: Data Validation
.venv\Scripts\python.exe pipelines/run_lab2_preprocessing.py

# Lab 3: Baseline Pipeline
.venv\Scripts\python.exe pipelines/run_lab3_baseline.py

# Lab 4: MLflow Tracking & Reproducibility
.venv\Scripts\python.exe pipelines/run_lab4_tracking.py

# Lab 5: Production Preprocessor & Output Checks
.venv\Scripts\python.exe pipelines/run_lab5_pipeline.py

# Lab 6: Model Registry & Lifecycle Face-Off
.venv\Scripts\python.exe pipelines/run_lab6_registry.py

# Lab 7: Multi-Stage ML Pipeline
.venv\Scripts\python.exe pipelines/run_lab7_pipeline.py

# Lab 8: CI Quality Gate Check
.venv\Scripts\python.exe pipelines/run_lab8_ci.py

# Lab 9: Failure Injection & Self-Healing
.venv\Scripts\python.exe pipelines/run_lab9_failure_handling.py

# Lab 10: FastAPI Serving & Performance Tests
.venv\Scripts\python.exe pipelines/run_lab10_api.py

# Lab 11: Containerization & Deployment
.venv\Scripts\python.exe pipelines/run_lab11_deployment.py

# Lab 12: Drift Monitoring & Retraining
.venv\Scripts\python.exe pipelines/run_lab12_monitoring.py

# Lab 13: End-to-End Monitoring Trigger
.venv\Scripts\python.exe pipelines/run_lab13_end_to_end.py
```

---

## 6. Model Serving and API Engineering (FastAPI)

Launch the interactive API server in VS Code:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **Healthcheck**: `http://127.0.0.1:8000/health`
- **Model Metadata**: `http://127.0.0.1:8000/metadata`
- **Single Prediction**: `POST /predict`
- **Batch Prediction**: `POST /predict/batch`

---

## 7. MLflow Experiment Tracking & Model Registry

Launch the MLflow Web UI:

```powershell
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Navigate to `http://127.0.0.1:5000` to visualize run comparisons, scatter plots, confusion matrices, and the registered model `News_Category_Production_Model`.

---

## 8. Containerization & Kubernetes

Prepare the production model bundle, then build and run the API container:

```powershell
.venv\Scripts\python.exe pipelines/run_lab11_deployment.py
docker build -t news-category-api:1.0 .
docker run --rm -p 8000:8000 news-category-api:1.0
```

The deployment step packages the model, preprocessor, and metadata into `deployment/`. Build the image after that step so the bundle is included. To deploy to a configured Kubernetes cluster:

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 9. Repository Layout

The project follows the same nested MLOps layout as the original Churn Prediction project. The filenames and artifacts are specific to news classification; the datasets, models, and reports are not shared with that project.

```
cla2/
├── .github/
│   └── workflows/              # CI, failure testing, and monitoring
├── api/                        # FastAPI application, model loader, schemas
├── artifacts/                  # Evaluation plots and experiment artifacts
├── config.yaml
├── data/
│   ├── raw/
│   │   └── archive/            # Ingested data snapshots and lineage
│   ├── processed/              # Feature matrices, splits, and metadata
│   └── monitoring/drift/       # Monitoring reference data
├── deployment/                 # Production bundle builder
├── docs/                       # Architecture, runbook, and lab reports
├── k8s/                        # Kubernetes Deployment and Service
├── logs/
├── models/                     # Trained news classifier and preprocessors
├── notebooks/
├── outputs/
│   ├── drift/
│   └── monitoring/
├── pipelines/                  # Lab runners and experiment entrypoints
├── reports/                    # Evaluation, quality gate, and lifecycle
├── src/
│   ├── mlops_pipeline/         # Numbered lifecycle and monitoring stages
│   └── *.py                    # Reusable data, training, and validation modules
└── tests/                      # API, model consistency, and performance tests
```

The original dataset, trained models, MLflow database, and experiment outputs remain project-specific local assets; they are not moved into or replaced with Churn project files.
