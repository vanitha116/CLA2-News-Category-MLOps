import json
import os
import re
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for text in X:
            if not isinstance(text, str):
                cleaned.append("")
                continue
            t = text.lower()
            t = re.sub(r"https?://\S+|www\.\S+", " ", t)
            t = re.sub(r"[^a-zA-Z\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return np.array(cleaned)


def build_preprocessor_pipeline(max_features=25000, ngram_range=(1, 2), sublinear_tf=True):
    return Pipeline([
        ("cleaner", TextCleanerTransformer()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words="english",
            min_df=2
        ))
    ])


def run_preprocessing_pipeline():
    print("[INFO] Starting Production Preprocessor Pipeline...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    articles = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    df = pd.DataFrame(articles)
    df["combined_text"] = df["headline"].fillna("") + " " + df["short_description"].fillna("")

    cat_counts = df["category"].value_counts()
    valid_cats = cat_counts[cat_counts >= 5].index
    df = df[df["category"].isin(valid_cats)].reset_index(drop=True)

    X = df[["combined_text", "headline", "short_description", "link"]]
    y = df["category"]

    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    feat_cfg = config["features"]
    pipeline = build_preprocessor_pipeline(
        max_features=feat_cfg.get("max_features", 25000),
        ngram_range=tuple(feat_cfg.get("ngram_range", [1, 2])),
        sublinear_tf=feat_cfg.get("sublinear_tf", True)
    )

    print("[INFO] Fitting and transforming training data with Pipeline...")
    X_train_final = pipeline.fit_transform(X_train_df["combined_text"])
    X_test_final = pipeline.transform(X_test_df["combined_text"])

    y_train = y_train_s.to_numpy()
    y_test = y_test_s.to_numpy()

    sp.save_npz(os.path.join(processed_dir, "X_train_final.npz"), X_train_final)
    sp.save_npz(os.path.join(processed_dir, "X_test_final.npz"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    X_train_df.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test_df.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    pipeline_path = config["paths"]["preprocessor"]
    joblib.dump(pipeline, pipeline_path)

    metadata = {
        "dataset_name": "News_Category_Classification",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "num_classes": int(len(np.unique(y_train))),
        "pipeline_stages": [name for name, _ in pipeline.steps]
    }
    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Production Preprocessing Pipeline executed and saved to {pipeline_path}.")
    return True


if __name__ == "__main__":
    run_preprocessing_pipeline()
