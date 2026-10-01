import sys
import os
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validate_exp5 import run_experiment_5_suite

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("  LAUNCHING EXPERIMENT 5: DATA VALIDATION & REPRODUCIBLE ML PIPELINE")
    print("#"*80)
    start_time = time.time()
    try:
        run_experiment_5_suite()
        elapsed = time.time() - start_time
        print(f"\n[PIPELINE SUCCESS] Experiment 5 completed successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        print(f"\n[PIPELINE ERROR] Experiment 5 execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
