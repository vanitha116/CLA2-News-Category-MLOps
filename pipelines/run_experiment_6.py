import sys
import os
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.registry_lifecycle_exp6 import run_experiment_6_suite
from src.inference_service import NewsInferenceService

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("  LAUNCHING EXPERIMENT 6: MODEL REGISTRY & LIFECYCLE MANAGEMENT")
    print("#"*80)
    start_time = time.time()
    try:
        run_experiment_6_suite()

        print("\n" + "-"*80)
        print("  VERIFYING PRODUCTION INFERENCE FROM REGISTRY")
        print("-"*80)
        service = NewsInferenceService()
        test_articles = [
            ("Astronomers Discover New Exoplanet in Habitable Zone", "The Earth-sized planet orbits a star 40 light years away."),
            ("Championship Finals Conclude With Overtime Thriller", "The defending champions secured their victory in the closing seconds."),
            ("Inflation Cools Down In Latest Economic Indicators", "Federal reports show consumer prices stabilizing across major sectors.")
        ]
        for h, d in test_articles:
            res = service.predict(h, d)
            print(f"[{res['predicted_category']}] -> {res['headline']} (Latency: {res['inference_latency_ms']} ms)")

        elapsed = time.time() - start_time
        print(f"\n[PIPELINE SUCCESS] Experiment 6 completed successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        print(f"\n[PIPELINE ERROR] Experiment 6 execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
