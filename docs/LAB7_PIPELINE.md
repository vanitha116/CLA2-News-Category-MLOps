# Lab 7 — ML Pipeline Engineering

Lab 7 builds the core end-to-end News Category Classification MLOps pipeline.

## Standard Stages Executed:

```text
01 Load Data & Batch Lineage
02 Raw Data Validation (Pandera)
03 Preprocessing Pipeline (TextCleaner + TF-IDF)
04 Processed Output Validation
05 Training + MLflow + Model Registry
06 Model Evaluation
07 Model Quality Gate (F1 >= 0.45)
08 Automated Model Lifecycle
09 Deployment Update
10 Production Inference
```

## Running the Pipeline

Run the master pipeline in VS Code:

```bash
python pipelines/run_lab7_pipeline.py
```

Or run directly:

```bash
python src/mlops_pipeline/news_classification_pipeline.py --force
```

Execution logs are automatically preserved in `logs/news_classification_pipeline_run_log.json`.
