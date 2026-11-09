import json
import os
import re
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_preprocessing():
    print("[INFO] Starting Baseline Preprocessing Pipeline...")

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
    print(f"[INFO] Loaded {len(df)} records from {data_path}.")

    df["combined_text"] = (df["headline"].fillna("") + " " + df["short_description"].fillna("")).apply(clean_text)

    # Filter out categories with fewer than 5 records if needed
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
    max_features = feat_cfg.get("max_features", 25000)
    ngram_range = tuple(feat_cfg.get("ngram_range", [1, 2]))
    sublinear_tf = feat_cfg.get("sublinear_tf", True)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        stop_words="english",
        min_df=2
    )

    print(f"[INFO] Fitting TF-IDF Vectorizer (max_features={max_features}, ngrams={ngram_range})...")
    X_train_final = vectorizer.fit_transform(X_train_df["combined_text"])
    X_test_final = vectorizer.transform(X_test_df["combined_text"])

    y_train = y_train_s.to_numpy()
    y_test = y_test_s.to_numpy()

    # Save processed matrices
    sp.save_npz(os.path.join(processed_dir, "X_train_final.npz"), X_train_final)
    sp.save_npz(os.path.join(processed_dir, "X_test_final.npz"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    # Save raw splits for reference and drift detection
    X_train_df.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test_df.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    # Save vectorizer artifact
    joblib.dump(vectorizer, config["paths"]["preprocessor"])

    metadata = {
        "dataset_name": "News_Category_Classification",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "num_classes": int(len(np.unique(y_train))),
        "vocabulary_size": int(len(vectorizer.vocabulary_))
    }
    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing completed! Train: {X_train_final.shape}, Test: {X_test_final.shape}")
    print(f"[INFO] Preprocessor saved to {config['paths']['preprocessor']}.")
    return True


if __name__ == "__main__":
    run_preprocessing()
