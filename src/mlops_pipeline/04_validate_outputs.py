import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validate_outputs import validate_preprocessing_outputs

if __name__ == "__main__":
    print("[INFO] Starting Stage 04: Processed Output Validation...")
    passed = validate_preprocessing_outputs()
    sys.exit(0 if passed else 1)
