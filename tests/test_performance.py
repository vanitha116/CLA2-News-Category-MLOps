import json
import time
import numpy as np
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_performance_benchmarks(tmp_path):
    payload = {
        "headline": "Quantum Computing Breakthrough Solves Complex Cryptographic Challenge",
        "short_description": "Scientists demonstrate computational supremacy with superconductor qubit arrays."
    }

    # Warm-up request
    client.post("/predict", json=payload)

    latencies = []
    num_requests = 50

    t_start = time.time()
    for _ in range(num_requests):
        t0 = time.time()
        res = client.post("/predict", json=payload)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        assert res.status_code == 200

    total_time = time.time() - t_start
    throughput = num_requests / total_time

    benchmarks = {
        "total_requests": num_requests,
        "total_time_seconds": round(total_time, 3),
        "throughput_requests_per_second": round(throughput, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2)
    }

    print("\n--- API Performance Benchmarks ---")
    for k, v in benchmarks.items():
        print(f"  {k:32s}: {v}")

    report_path = tmp_path / "api_performance_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=4)

    assert benchmarks["mean_latency_ms"] < 100.0  # Must be fast
