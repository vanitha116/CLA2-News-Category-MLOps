import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess_pipeline import run_preprocessing_pipeline

if __name__ == "__main__":
    print("[INFO] Starting Stage 03: Preprocessing Pipeline...")
    passed = run_preprocessing_pipeline()
    sys.exit(0 if passed else 1)
