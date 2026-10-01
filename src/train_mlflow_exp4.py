import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    ARTIFACTS_DIR, RANDOM_STATE, MODELS_DIR
)
from src.data_loader import (
    load_raw_dataframe, prepare_train_test_split, get_data_lineage_metadata
)
from src.preprocessor_pipeline import build_text_pipeline

def setup_mlflow():
    """Initializes MLflow tracking database and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"[MLFLOW] Tracking URI set to: {MLFLOW_TRACKING_URI}")
    print(f"[MLFLOW] Experiment active: '{MLFLOW_EXPERIMENT_NAME}'")

def evaluate_and_log_run(
    run_name: str,
    model,
    model_params: dict,
    vectorizer_params: dict,
    X_train_raw, X_test_raw,
    y_train, y_test,
    lineage_metadata: dict
):
    """
    Executes training, computes comprehensive metrics, logs diagnostic plots,
    registers model and artifacts to MLflow.
    """
    print(f"\n==================================================")
    print(f"[EXPERIMENT 4] Running: {run_name}")
    print(f"==================================================")

    # 1. Build and apply specific preprocessing configuration for this run
    t0 = time.time()
    preprocessor = build_text_pipeline(
        max_features=vectorizer_params["max_features"],
        ngram_range=vectorizer_params["ngram_range"],
        sublinear_tf=vectorizer_params.get("sublinear_tf", True)
    )

    print(f"[TRAIN] Fitting preprocessor pipeline on training text...")
    X_train_vec = preprocessor.fit_transform(X_train_raw)
    X_test_vec = preprocessor.transform(X_test_raw)
    prep_time = time.time() - t0

    # 2. Fit the model
    t_train_start = time.time()
    print(f"[TRAIN] Fitting model: {model.__class__.__name__}...")
    model.fit(X_train_vec, y_train)
    train_time = time.time() - t_train_start
    total_latency = prep_time + train_time

    # 3. Evaluate predictions
    y_pred_train = model.predict(X_train_vec)
    y_pred_test = model.predict(X_test_vec)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_weighted = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    metrics = {
        "train_accuracy": float(acc_train),
        "test_accuracy": float(acc_test),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "preprocessing_seconds": float(round(prep_time, 3)),
        "training_seconds": float(round(train_time, 3)),
        "total_latency_seconds": float(round(total_latency, 3))
    }

    print(f"[RESULTS] Test Accuracy: {acc_test:.4f} | F1 (weighted): {f1_weighted:.4f} | Latency: {total_latency:.2f}s")

    # 4. MLflow Logging
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_params({f"tfidf_{k}": v for k, v in vectorizer_params.items()})
        mlflow.log_param("model_family", model.__class__.__name__)
        mlflow.log_param("dataset_name", lineage_metadata["dataset_name"])
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("test_samples", len(y_test))

        # Log metrics
        mlflow.log_metrics(metrics)

        # 5. Diagnostic Artifacts: Confusion Matrix (Top 10 Classes for readability)
        top_classes = pd.Series(y_test).value_counts().head(10).index.tolist()
        mask = np.isin(y_test, top_classes) & np.isin(y_pred_test, top_classes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_test[mask], y_pred_test[mask],
            labels=top_classes,
            ax=ax,
            cmap="Blues",
            xticks_rotation=45
        )
        ax.set_title(f"Confusion Matrix (Top 10 Categories) - {run_name}", fontsize=12)
        plt.tight_layout()

        cm_filename = f"cm_{run_name.lower()}.png"
        cm_path = ARTIFACTS_DIR / cm_filename
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="diagnostic_plots")

        # Save classification report
        clf_report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(clf_report).transpose()
        report_path = ARTIFACTS_DIR / f"report_{run_name.lower()}.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(str(report_path), artifact_path="evaluation_reports")

        # Log dataset lineage metadata
        meta_path = ARTIFACTS_DIR / "dataset_lineage.json"
        with open(meta_path, "w") as f:
            json.dump(lineage_metadata, f, indent=4)
        mlflow.log_artifact(str(meta_path), artifact_path="lineage")

        # Log model artifact with input example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print(f"[MLFLOW] [SUCCESS] Run '{run_name}' tracked under Run ID: {run_id}")
        return run_id, metrics

def run_experiment_4_suite():
    """
    Executes the 4 systematic runs to study:
    1. Baseline Naive Bayes
    2. Effect of Advanced Preprocessing & Feature Selection
    3. Effect of Hyperparameter Tuning
    4. Alternative High-Margin Algorithm (LinearSVC)
    """
    setup_mlflow()

    # Load dataset
    df = load_raw_dataframe()
    lineage_metadata = get_data_lineage_metadata(df)
    X_train_raw, X_test_raw, y_train, y_test = prepare_train_test_split(df)

    runs_summary = []

    # Run 1: Baseline Naive Bayes (Simple unigram TF-IDF, max_features=10,000)
    nb_params = {"alpha": 1.0}
    nb_vec_params = {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": False}
    nb_model = MultinomialNB(**nb_params)
    run1_id, m1 = evaluate_and_log_run(
        "MultinomialNB_Baseline", nb_model, nb_params, nb_vec_params,
        X_train_raw, X_test_raw, y_train, y_test, lineage_metadata
    )
    runs_summary.append({"Run": "1. MultinomialNB Baseline", "Model": "MultinomialNB", **m1})

    # Run 2: Preprocessing & Feature Selection Effect (N-grams (1,2), sublinear TF, 25,000 features)
    lr2_params = {"C": 1.0, "max_iter": 500, "random_state": RANDOM_STATE}
    lr2_vec_params = {"max_features": 25000, "ngram_range": (1, 2), "sublinear_tf": True}
    lr2_model = LogisticRegression(**lr2_params)
    run2_id, m2 = evaluate_and_log_run(
        "LogisticRegression_Advanced_TFIDF", lr2_model, lr2_params, lr2_vec_params,
        X_train_raw, X_test_raw, y_train, y_test, lineage_metadata
    )
    runs_summary.append({"Run": "2. Preprocessing & Feature Selection Effect", "Model": "LogisticRegression", **m2})

    # Run 3: Hyperparameter Tuning Effect (C=2.0, class_weight='balanced')
    lr3_params = {"C": 2.0, "max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE}
    lr3_vec_params = {"max_features": 25000, "ngram_range": (1, 2), "sublinear_tf": True}
    lr3_model = LogisticRegression(**lr3_params)
    run3_id, m3 = evaluate_and_log_run(
        "LogisticRegression_Tuned_C", lr3_model, lr3_params, lr3_vec_params,
        X_train_raw, X_test_raw, y_train, y_test, lineage_metadata
    )
    runs_summary.append({"Run": "3. Hyperparameter Tuning Effect", "Model": "LogisticRegression (Tuned)", **m3})

    # Run 4: Alternative Model Architecture (Linear Support Vector Classifier)
    svc_params = {"C": 0.5, "class_weight": "balanced", "random_state": RANDOM_STATE}
    svc_vec_params = {"max_features": 25000, "ngram_range": (1, 2), "sublinear_tf": True}
    svc_model = LinearSVC(**svc_params)
    run4_id, m4 = evaluate_and_log_run(
        "LinearSVC_Classifier", svc_model, svc_params, svc_vec_params,
        X_train_raw, X_test_raw, y_train, y_test, lineage_metadata
    )
    runs_summary.append({"Run": "4. LinearSVC Margin Maximizer", "Model": "LinearSVC", **m4})

    # Build and print comparison table
    comparison_df = pd.DataFrame(runs_summary)
    comparison_csv = ARTIFACTS_DIR / "experiment_4_runs_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 4: SYSTEMATIC RUN COMPARISON TABLE")
    print("="*80)
    cols = ["Run", "Model", "test_accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "total_latency_seconds"]
    print(comparison_df[cols].to_string(index=False))
    print("="*80)
    print(f"[INFO] Comparison table exported to: {comparison_csv}")

    # Reproducibility verification test
    print("\n--- Performing Automated Reproducibility Verification ---")
    repro_model = MultinomialNB(**nb_params)
    repro_prep = build_text_pipeline(max_features=10000, ngram_range=(1,1), sublinear_tf=False)
    X_tr = repro_prep.fit_transform(X_train_raw)
    X_te = repro_prep.transform(X_test_raw)
    repro_model.fit(X_tr, y_train)
    repro_preds = repro_model.predict(X_te)
    repro_f1 = f1_score(y_test, repro_preds, average="weighted", zero_division=0)

    f1_diff = abs(repro_f1 - m1["f1_weighted"])
    assert f1_diff < 1e-6, f"Reproducibility verification failed! Delta: {f1_diff}"
    print(f"[REPRODUCIBILITY] [PASS] Verified deterministic execution across runs. Delta = {f1_diff:.8f}")

    return comparison_df

if __name__ == "__main__":
    run_experiment_4_suite()
