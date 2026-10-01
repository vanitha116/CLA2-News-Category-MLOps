import os
import json
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_loader import clean_text
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Reusable, idempotent text cleaner transformer conforming to the Scikit-Learn API.
    Property: clean_text(clean_text(x)) == clean_text(x) (Idempotent).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Applies clean_text deterministically across all elements
        if isinstance(X, np.ndarray) or isinstance(X, list):
            return np.array([clean_text(str(text)) for text in X])
        elif hasattr(X, "apply"):
            return X.apply(lambda text: clean_text(str(text))).values
        else:
            return np.array([clean_text(str(X))])

def build_text_pipeline(max_features: int = 25000, ngram_range=(1, 2), sublinear_tf: bool = True) -> Pipeline:
    """
    Constructs an end-to-end reproducible NLP preprocessing pipeline.
    Pipeline stages:
      1. TextCleanerTransformer (Idempotent regex normalization)
      2. TfidfVectorizer (Sublinear TF scaling, n-grams, stop-words)
    """
    pipeline = Pipeline(steps=[
        ("cleaner", TextCleanerTransformer()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=2,
            max_df=0.95,
            stop_words="english"
        ))
    ])
    return pipeline

def run_preprocessor_pipeline(X_train, X_test, y_train, y_test, max_features: int = 25000, ngram_range=(1, 2)):
    """
    Executes the preprocessing pipeline, saves transformed arrays and serialized pipeline.
    """
    print(f"\n[PIPELINE] Building preprocessing pipeline (max_features={max_features}, ngrams={ngram_range})...")
    pipeline = build_text_pipeline(max_features=max_features, ngram_range=ngram_range)

    print("[PIPELINE] Fitting and transforming training data...")
    X_train_final = pipeline.fit_transform(X_train)

    print("[PIPELINE] Transforming testing data...")
    X_test_final = pipeline.transform(X_test)

    # Save sparse matrices efficiently
    X_train_path = PROCESSED_DATA_DIR / "X_train_final.npz"
    X_test_path = PROCESSED_DATA_DIR / "X_test_final.npz"
    y_train_path = PROCESSED_DATA_DIR / "y_train.npy"
    y_test_path = PROCESSED_DATA_DIR / "y_test.npy"

    sp.save_npz(X_train_path, X_train_final)
    sp.save_npz(X_test_path, X_test_final)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

    # Save pipeline object
    pipeline_path = MODELS_DIR / "preprocessor_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    # Save metadata configuration
    metadata = {
        "pipeline_stages": ["TextCleanerTransformer", "TfidfVectorizer"],
        "max_features": max_features,
        "ngram_range": list(ngram_range),
        "sublinear_tf": True,
        "stop_words": "english",
        "train_matrix_shape": list(X_train_final.shape),
        "test_matrix_shape": list(X_test_final.shape),
        "vocabulary_size": len(pipeline.named_steps['tfidf'].vocabulary_)
    }
    with open(PROCESSED_DATA_DIR / "preprocessor_config.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[PIPELINE] [SUCCESS] Preprocessing completed!")
    print(f"  -> X_train shape: {X_train_final.shape} | X_test shape: {X_test_final.shape}")
    print(f"  -> Pipeline saved to: {pipeline_path}")
    return pipeline, X_train_final, X_test_final
