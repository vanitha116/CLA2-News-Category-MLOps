import sys
import os
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_mlflow_exp4 import run_experiment_4_suite

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("  LAUNCHING EXPERIMENT 4: MLFLOW EXPERIMENT TRACKING & REPRODUCIBILITY")
    print("#"*80)
    start_time = time.time()
    try:
        run_experiment_4_suite()
        elapsed = time.time() - start_time
        print(f"\n[PIPELINE SUCCESS] Experiment 4 completed successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        print(f"\n[PIPELINE ERROR] Experiment 4 execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
