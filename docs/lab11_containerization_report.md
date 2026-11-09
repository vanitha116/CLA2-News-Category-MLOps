# Lab 11 — Containerization and Kubernetes Deployment Report

## 1. Objective

Containerize the News Category Classification FastAPI inference application with Docker, validate container portability, and orchestrate deployment using Kubernetes manifests.

---

## 2. Deployment Architecture

```text
MLflow Model Registry
         │
         ▼
Deployment Bundle (deployment/model.pkl, preprocessor.pkl, metadata.json)
         │
         ▼
Docker Image (news-category-api:1.0)
         │
         ▼
Kubernetes Deployment (k8s/deployment.yaml) ──> Replicas: 2 Pods
         │
         ▼
Kubernetes Service (k8s/service.yaml) ────────> NodePort: 30080
         │
         ▼
FastAPI REST API
```

---

## 3. Docker Container Specifications

- **Base Image**: `python:3.12-slim`
- **Workdir**: `/app`
- **Exposed Port**: `8000`
- **Container Entrypoint**: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

---

## 4. Kubernetes Manifests

- **`k8s/deployment.yaml`**: Defines a 2-replica deployment with liveness and readiness HTTP probes on `/health` and CPU/memory resource limits.
- **`k8s/service.yaml`**: Exposes the service via NodePort on port `30080`.
