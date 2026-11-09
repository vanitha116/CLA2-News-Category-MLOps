# MLOps Platform Operational Runbook (Labs 1 to 13)

This runbook provides complete operational instructions for running and verifying every lab in **VS Code**.

---

## 1. Quick Start: Master Sequential Runner

To run the entire platform across all 13 labs in one automated pass:

```powershell
.venv\Scripts\python.exe pipelines/run_all_labs.py
```

---

## 2. Individual Lab Execution Commands

| Lab | Goal | Command |
| :--- | :--- | :--- |
| **Lab 1** | Ingest new batch, calculate SHA-256 hash, and update lineage | `.venv\Scripts\python.exe pipelines/run_lab1_data_loading.py` |
| **Lab 2** | Validate raw dataset with Pandera schema guardrails | `.venv\Scripts\python.exe pipelines/run_lab2_preprocessing.py` |
| **Lab 3** | Execute baseline preprocessing, training, and evaluation | `.venv\Scripts\python.exe pipelines/run_lab3_baseline.py` |
| **Lab 4** | Run MLflow experiment tracking and verify reproducibility | `.venv\Scripts\python.exe pipelines/run_lab4_tracking.py` |
| **Lab 5** | Execute reusable preprocessing and matrix quality audit | `.venv\Scripts\python.exe pipelines/run_lab5_pipeline.py` |
| **Lab 6** | Register candidate model and run Champion-Challenger promotion | `.venv\Scripts\python.exe pipelines/run_lab6_registry.py` |
| **Lab 7** | Execute end-to-end multi-stage ML pipeline | `.venv\Scripts\python.exe pipelines/run_lab7_pipeline.py` |
| **Lab 8** | Run CI pipeline with automated Quality Gate check | `.venv\Scripts\python.exe pipelines/run_lab8_ci.py` |
| **Lab 9** | Execute failure injection and self-healing recovery suite | `.venv\Scripts\python.exe pipelines/run_lab9_failure_handling.py` |
| **Lab 10** | Run API test suite and latency/throughput benchmarks | `.venv\Scripts\python.exe pipelines/run_lab10_api.py` |
| **Lab 11** | Prepare deployment bundle and verify container readiness | `.venv\Scripts\python.exe pipelines/run_lab11_deployment.py` |
| **Lab 12** | Run data, prediction, and concept drift analysis | `.venv\Scripts\python.exe pipelines/run_lab12_monitoring.py` |
| **Lab 13** | Run closed-loop monitoring trigger and automated retraining | `.venv\Scripts\python.exe pipelines/run_lab13_end_to_end.py` |

---

## 3. Launching the Live FastAPI Server

In your VS Code terminal, start the server:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

Open your browser to:
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Healthcheck Endpoint: `http://127.0.0.1:8000/health`
- Model Metadata: `http://127.0.0.1:8000/metadata`

---

## 4. Launching the MLflow Tracking Web UI

In your VS Code terminal, start the UI:

```powershell
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` to inspect:
- Experiment comparison charts, scatter plots, and parallel coordinates
- Logged artifacts (confusion matrices, dataset lineage metadata)
- Registered model versions, stages (`Production`, `Staging`, `Archived`), and lineage
