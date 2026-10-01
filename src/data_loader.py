import json
import re
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, SAMPLE_SIZE, RANDOM_STATE, TEST_SIZE

def clean_text(text: str) -> str:
    """Standardizes and cleans text for NLP preprocessing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_dataset_hash(file_path=RAW_DATA_PATH) -> str:
    """Computes SHA-256 hash of raw data file for data lineage and provenance."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read first 1MB chunk to quickly generate a unique deterministic fingerprint
        chunk = f.read(1024 * 1024)
        hasher.update(chunk)
    return hasher.hexdigest()

def load_raw_dataframe(file_path=RAW_DATA_PATH, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE) -> pd.DataFrame:
    """
    Loads News_Category_Dataset_v3.json into a pandas DataFrame.
    If sample_size is provided, applies stratified sampling across categories.
    """
    print(f"[DATA LOADER] Loading raw dataset from: {file_path}")
    articles = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    df = pd.DataFrame(articles)
    total_records = len(df)
    print(f"[DATA LOADER] Total records in raw dataset: {total_records}")

    if sample_size and sample_size < total_records:
        print(f"[DATA LOADER] Performing stratified sampling to {sample_size} records (seed={random_state})...")
        # Ensure categories with very few samples don't fail stratified split
        counts = df['category'].value_counts()
        valid_cats = counts[counts >= 5].index
        df_filtered = df[df['category'].isin(valid_cats)]
        
        df, _ = train_test_split(
            df_filtered,
            train_size=sample_size,
            random_state=random_state,
            stratify=df_filtered['category']
        )
        df = df.reset_index(drop=True)
        print(f"[DATA LOADER] Sampled dataset shape: {df.shape}")

    return df

def prepare_train_test_split(df: pd.DataFrame, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Prepares text inputs and categorical targets, then splits into train and test splits.
    """
    # Create combined feature text
    df['combined_text'] = (df['headline'].fillna('') + " " + df['short_description'].fillna('')).apply(clean_text)
    
    X = df['combined_text'].values
    y = df['category'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[DATA LOADER] Split complete -> Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def get_data_lineage_metadata(df: pd.DataFrame) -> dict:
    """Returns metadata dictionary for MLflow lineage logging."""
    return {
        "dataset_name": "News_Category_Dataset_v3",
        "dataset_sha256_prefix": compute_dataset_hash(),
        "total_records_loaded": int(len(df)),
        "num_classes": int(df['category'].nunique()),
        "columns": list(df.columns),
        "random_state": RANDOM_STATE
    }
