# MLOps Experiments Suite: News Category Classification
### Experiments 4, 5, and 6 Implementation for VS Code

This repository contains the complete, production-grade implementation of **Experiment 4**, **Experiment 5**, and **Experiment 6** tailored for the **News Category Classification** project using [`News_Category_Dataset_v3.json`](./News_Category_Dataset_v3.json).

---

## Table of Contents
1. [Overview of Implemented Experiments](#overview-of-implemented-experiments)
2. [Environment Setup in VS Code](#environment-setup-in-vs-code)
3. [Running via CLI Terminal](#running-via-cli-terminal)
4. [Running Interactive Notebooks in VS Code](#running-interactive-notebooks-in-vs-code)
5. [Launching the MLflow Web UI](#launching-the-mlflow-web-ui)
6. [Architecture & Project Structure](#architecture--project-structure)
7. [Detailed Experiments Breakdown & Results](#detailed-experiments-breakdown--results)

---

## 1. Overview of Implemented Experiments

### Experiment 4: Experiment Tracking and Reproducibility Workflows
- **Objective**: Track parameters, metrics, preprocessing configurations, datasets, and trained artifacts using MLflow with an SQLite backend.
- **Systematic Model Comparisons**:
  1. `MultinomialNB_Baseline`: Fast baseline with standard unigrams (`max_features=10,000`).
  2. `LogisticRegression_Advanced_TFIDF`: Feature engineering with bigrams `(1, 2)` and sublinear TF scaling (`max_features=25,000`).
  3. `LogisticRegression_Tuned_C`: Hyperparameter tuning with $C=2.0$ and balanced class weighting.
  4. `LinearSVC_Classifier`: Linear Support Vector Classifier with high-dimensional margin maximization.
- **Reproducibility Guarantee**: Automated verification testing zero performance drift ($\Delta F1 < 10^{-6}$) across repeated deterministic runs.
- **Artifacts Logged**: Confusion matrices (`confusion_matrix.png`), classification reports, dataset lineage hashes, and MLflow sklearn model packages.

### Experiment 5: Data Validation and Reproducible ML Pipelines
- **Objective**: Schema-aware data preprocessing workflows identifying datatype mismatches, schema inconsistencies, missing attributes, invalid categorical values, and corrupted records.
- **Pandera Schema Guardrails**:
  - `headline`: String, non-empty, minimum length $\ge 3$.
  - `category`: Strict membership validation against all 42 allowable news categories.
  - `date`: Valid ISO format (`YYYY-MM-DD`).
  - `link`: Valid HTTP/HTTPS URL pattern.
- **Fault-Injection Test Suite**: Injects 6 deliberate real-world corruptions and traps every violation into `artifacts/schema_validation_failures.csv`.
- **Reusable & Idempotent Preprocessing Pipeline**: Scikit-Learn `Pipeline` wrapping a custom `TextCleanerTransformer` and `TfidfVectorizer` serialized to `models/preprocessor_pipeline.pkl`.
- **Mathematical Idempotence Proof**: Verifies $clean(clean(x)) = clean(x)$ and zero matrix drift on repeated execution.

### Experiment 6: Model Registry and Version Management
- **Objective**: Model registration, versioning, and lifecycle management across `Staging`, `Production`, and `Archived` stages.
- **Model Registry Target**: `News_Category_Classifier`.
- **Lineage Packaging**: Packages preprocessing pipeline dependencies directly with registered models.
- **Automated Champion vs. Challenger Face-Off**:
  - Automatically identifies newly registered versions and advances them to `Staging`.
  - Conducts metric-based competition against the current `Production` Champion.
  - Promotes winning Challenger to `Production` and archives superseded models.
- **Instant Rollback Capability**: Programmatic function to reinstate prior versions if post-deployment drift occurs.
- **Live Production Inference Service**: Automatically queries `Production` stage model from registry and classifies live news headlines with $< 2\text{ ms}$ latency.

---

## 2. Environment Setup in VS Code

1. **Open the Project in VS Code**:
   - Launch VS Code and open the folder `cla2` (`File` > `Open Folder...`).

2. **Select the Python Virtual Environment**:
   - Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac) to open the Command Palette.
   - Type `Python: Select Interpreter` and press `Enter`.
   - Choose the virtual environment located at:
     ```
     .\.venv\Scripts\python.exe
     ```

---

## 3. Running via CLI Terminal

Open the built-in terminal in VS Code (`Ctrl + \`` or `Terminal` > `New Terminal`) and run any of the following commands:

### A. Run Master Pipeline (All Experiments Sequentially)
```powershell
.venv\Scripts\python.exe pipelines/run_all_experiments.py
```

### B. Run Individual Experiments
- **Run Experiment 4 (MLflow Tracking & Reproducibility)**:
  ```powershell
  .venv\Scripts\python.exe pipelines/run_experiment_4.py
  ```
- **Run Experiment 5 (Data Validation & Idempotent Pipeline)**:
  ```powershell
  .venv\Scripts\python.exe pipelines/run_experiment_5.py
  ```
- **Run Experiment 6 (Model Registry & Live Inference)**:
  ```powershell
  .venv\Scripts\python.exe pipelines/run_experiment_6.py
  ```

---

## 4. Running Interactive Notebooks in VS Code

For interactive cell-by-cell execution, presentations, and visual reports, open any notebook in the `notebooks/` directory inside VS Code:

1. **`notebooks/experiment_4_tracking.ipynb`**
   - Interactive MLflow experiment setup.
   - Run comparison table with highlighted metrics.
   - Display of confusion matrices for each algorithm.

2. **`notebooks/experiment_5_data_validation.ipynb`**
   - Pandera schema inspection.
   - Fault injection and visual table of caught corruptions.
   - Output matrix quality checks and idempotency verification.

3. **`notebooks/experiment_6_model_registry.ipynb`**
   - Inspection of model registry versions and stages.
   - Automated Champion-Challenger face-off execution.
   - Live news classification using the registered production model.

> **Tip**: When opening a `.ipynb` file in VS Code, click **Select Kernel** in the top-right corner and choose **Python Environments...** > **.venv**.

---

## 5. Launching the MLflow Web UI

To visualize your experiments, charts, parameter comparisons, and model registry visually:

1. In your VS Code terminal, run:
   ```powershell
   .venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
3. Explore:
   - **Experiments Tab**: Click `News_Category_Classification` to view all logged runs, compare metrics on parallel coordinates and scatter plots, and download artifacts.
   - **Models Tab**: Click `News_Category_Classifier` to view all versions, current stages (`Production`, `Staging`, `Archived`), and deployment tags.

---

## 6. Architecture & Project Structure

```
cla2/
├── .venv/                              # Python Virtual Environment
├── News_Category_Dataset_v3.json       # Raw News Dataset (209,527 articles)
├── mlflow.db                           # SQLite Tracking Database for MLflow
├── README.md                           # Documentation & VS Code Guide
│
├── src/                                # Core Modular Source Code
│   ├── __init__.py                     # Package initializer
│   ├── config.py                       # Paths, categories, and parameters
│   ├── data_loader.py                  # Stratified loading & lineage tracking
│   ├── schema_validator.py             # Pandera schema & fault injection
│   ├── preprocessor_pipeline.py        # Idempotent text transformer pipeline
│   ├── train_mlflow_exp4.py            # Exp 4 tracking & reproducibility
│   ├── validate_exp5.py                # Exp 5 data & output validation
│   ├── registry_lifecycle_exp6.py      # Exp 6 registry & champion-challenger
│   └── inference_service.py            # Real-time production classifier
│
├── pipelines/                          # VS Code CLI Pipeline Runners
│   ├── run_experiment_4.py             # Exp 4 CLI runner
│   ├── run_experiment_5.py             # Exp 5 CLI runner
│   ├── run_experiment_6.py             # Exp 6 CLI runner
│   └── run_all_experiments.py          # Master sequential runner
│
├── notebooks/                          # Interactive VS Code Notebooks
│   ├── experiment_4_tracking.ipynb
│   ├── experiment_5_data_validation.ipynb
│   └── experiment_6_model_registry.ipynb
│
├── artifacts/                          # Diagnostic Plots & Reports
│   ├── cm_linearsvc_classifier.png
│   ├── cm_logisticregression_advanced_tfidf.png
│   ├── cm_logisticregression_tuned_c.png
│   ├── cm_multinomialnb_baseline.png
│   ├── dataset_lineage.json
│   ├── experiment_4_runs_comparison.csv
│   ├── preprocessing_summary_report.json
│   ├── production_model_report.json
│   └── schema_validation_failures.csv
│
├── data/processed/                     # Transformed Features & Arrays
│   ├── X_train_final.npz               # Sparse CSR feature matrix (Train)
│   ├── X_test_final.npz                # Sparse CSR feature matrix (Test)
│   ├── y_train.npy                     # Encoded labels (Train)
│   ├── y_test.npy                      # Encoded labels (Test)
│   └── preprocessor_config.json        # Preprocessing lineage metadata
│
└── models/                             # Serialized Models & Pipelines
    └── preprocessor_pipeline.pkl       # Reusable Scikit-Learn pipeline
```

---

## 7. Detailed Experiments Breakdown & Results

### Experiment 4 Performance Comparison Matrix
| Run Name | Model Family | Test Accuracy | F1 (Weighted) | Precision | Recall | Total Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. MultinomialNB Baseline** | MultinomialNB | 0.3934 | 0.2819 | 0.4045 | 0.3934 | ~0.89 s |
| **2. Advanced Feature Selection** | LogisticRegression | 0.5060 | 0.4353 | 0.4957 | 0.5060 | ~35.7 s |
| **3. Hyperparameter Tuning** | Tuned LogisticReg | 0.5138 | 0.5249 | 0.5524 | 0.5138 | ~18.4 s |
| **4. Margin Maximizer (Champion)** | LinearSVC | **0.5244** | **0.5301** | **0.5461** | **0.5244** | **~5.2 s** |

- **Effect of Preprocessing & Feature Selection**: Adding bigrams `(1, 2)` and sublinear TF scaling elevated accuracy from $39.3\%$ to $50.6\%$.
- **Effect of Hyperparameter Tuning**: Tuning $C=2.0$ with balanced class weights significantly boosted weighted F1-score from $0.435$ to $0.525$.
- **LinearSVC Champion**: LinearSVC produced the highest classification score ($F1 = 0.5301$) with ultra-fast training and inference latency ($5.2\text{ s}$).

### Experiment 5 Data Validation Audit Summary
- **Baseline Schema Test**: $1,000$ raw records validated against Pandera schema $\to$ **100% Passed**.
- **Fault-Injection Trap**: Injected 6 corrupted records $\to$ **6/6 Violations Trapped** (Caught datatype mismatch, missing category, invalid category name, corrupted date, corrupted URL, and unauthorized column).
- **Output Quality Check**: $0$ NaNs, $0$ Infinities, exact dimensional match ($25,000$ features across train and test), and bounded L2-normalized values.
- **Pipeline Idempotency**: Verified $clean(clean(x)) == clean(x)$ and execution drift delta is exactly $0.0000000000$.

### Experiment 6 Model Registry & Serving Summary
- **Registered Model**: `News_Category_Classifier`
- **Candidate V1 (`LogisticRegression`)**: Evaluated in Staging $\to F1 = 0.4353$.
- **Candidate V2 (`LinearSVC`)**: Evaluated in Staging $\to F1 = 0.5301$.
- **Lifecycle Decision**: Candidate V2 outperformed V1 by $+0.0948$, triggering automated promotion to **`Production`** and archiving V1.
- **Real-Time Classification Verification**:
  - *"Astronomers Discover New Exoplanet in Habitable Zone"* $\to$ **`SCIENCE`** ($1.01\text{ ms}$)
  - *"Championship Finals Conclude With Overtime Thriller"* $\to$ **`SPORTS`** ($0.85\text{ ms}$)
  - *"Inflation Cools Down In Latest Economic Indicators"* $\to$ **`MONEY`** ($1.00\text{ ms}$)
