import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_production_inference():
    print("[INFO] =========================================")
    print("[INFO] Starting Stage 09: Production Inference")
    print("[INFO] =========================================")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model"]
    prep_path = config["paths"]["preprocessor"]
    output_path = config["paths"]["inference_output"]

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        print(f"[ERROR] Required model ({model_path}) or preprocessor ({prep_path}) missing.")
        return False

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    sample_articles = [
        {"headline": "NASA James Webb Space Telescope Reveals Earliest Galaxies in the Cosmic Dawn", "short_description": "Astronomers discover primordial star clusters dating back to the Big Bang.", "source": "AP"},
        {"headline": "Federal Reserve Cuts Benchmark Interest Rates by 50 Basis Points as Inflation Stabilizes", "short_description": "Wall Street rallies after major policy shift by central bankers.", "source": "Reuters"},
        {"headline": "Hollywood Stars Dazzle on the Red Carpet at the Academy Awards Gala", "short_description": "Celebrities celebrate cinematic achievements in glamorous designer attire.", "source": "HuffPost"},
        {"headline": "National Team Clinches Championship Title With Thrilling Stoppage-Time Goal", "short_description": "Fans erupt into celebration as underdogs secure historic victory.", "source": "ESPN"}
    ]

    df = pd.DataFrame(sample_articles)
    df["combined_text"] = df["headline"] + " " + df["short_description"]

    print(f"[INFO] Running inference on {len(df)} production sample news records...")
    X_mat = preprocessor.transform(df["combined_text"])
    predictions = model.predict(X_mat)
    df["predicted_category"] = predictions

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[["headline", "predicted_category"]].to_csv(output_path, index=False)

    print("\n--- Production Predictions ---")
    for _, row in df.iterrows():
        print(f"  [{row['predicted_category']:15s}] -> {row['headline'][:65]}...")

    print(f"\n[SUCCESS] Inference output saved to {output_path}.")
    return True


if __name__ == "__main__":
    passed = run_production_inference()
    import sys
    sys.exit(0 if passed else 1)
