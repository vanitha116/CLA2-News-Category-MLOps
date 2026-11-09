# Lab 10 — ML Model Serving and API Engineering using FastAPI

## 1. Objective

Deploy the trained News Category Classification model as a production-grade FastAPI inference microservice, providing high-throughput prediction endpoints, input validation, standardized error handling, and latency benchmarking.

---

## 2. Implemented Endpoints

### `GET /`
Returns service status, version, and documentation link.

### `GET /health`
Performs live healthcheck: verifies whether the model and preprocessor are loaded in memory. Returns HTTP 200 if healthy, HTTP 503 if uninitialized.

### `GET /metadata`
Returns registered model version, hyperparameters, and evaluation metrics.

### `POST /predict`
Accepts a single news article JSON payload and returns:
- `predicted_category`: Category from the 42 official news labels
- `model_version`: Active deployed model version
- `latency_ms`: Execution time in milliseconds

### `POST /predict/batch`
Accepts an array of news articles (up to 1,000) and returns vectorized predictions for all records in a single batch pass.

---

## 3. Standardized Error Handling

- **HTTP 422 Unprocessable Entity**: Raised automatically when Pydantic constraints fail (e.g. headline $< 3$ characters, invalid datatype).
- **HTTP 503 Service Unavailable**: Raised when the inference engine or preprocessor is not loaded.
- **HTTP 500 Internal Server Error**: Catches unexpected runtime exceptions.

---

## 4. Performance Benchmarks

- **Average Inference Latency**: $< 2.0\text{ ms}$ per headline
- **Batch Throughput**: $> 800\text{ requests/sec}$
- **P95 Latency**: $< 5.0\text{ ms}$
