import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.automate_lifecycle import automate_lifecycle

if __name__ == "__main__":
    print("[INFO] Starting Stage 08: Automated Model Lifecycle...")
    passed = automate_lifecycle()
    sys.exit(0 if passed else 1)
