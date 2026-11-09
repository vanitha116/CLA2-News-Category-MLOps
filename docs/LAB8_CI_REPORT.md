# Lab 8 — Pipeline Automation and Continuous Integration (CI)

## Goal

Automate the execution of the full MLOps pipeline using GitHub Actions and enforce an automated **Quality Gate** before any candidate model can be evaluated for production promotion.

## CI Flow

```text
Git push / Pull Request / Manual Dispatch
                │
                ▼
        Install Dependencies
                │
                ▼
news_classification_pipeline.py
                │
                ▼
01 ──> 02 ──> 03 ──> 04 ──> 05 ──> 06 ──> 07 ──> 08 ──> 09
                │
                ▼
    Verify Quality Gate Status
                │
                ▼
     Upload Audit Artifacts
```

## Quality Gate Enforcement

Configured in `config.yaml`:

```yaml
quality_gate:
  metric: f1_score
  minimum_value: 0.45
```

If the candidate model's test F1-score falls below `0.45`, the pipeline exits with a non-zero exit code, blocking promotion and preventing faulty or degraded models from entering production.
