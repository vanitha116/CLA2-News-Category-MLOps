import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("outputs/monitoring")
LOG_FILE = LOG_DIR / "api_requests.jsonl"


def log_prediction_event(headline: str, predicted_category: str, latency_ms: float, model_version: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "headline_length": len(headline),
        "predicted_category": predicted_category,
        "latency_ms": latency_ms,
        "model_version": model_version
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass
