# Git Workflow Documentation — Lab 3

**Project:** News Category Classification MLOps Platform  
**Lab:** Lab 3 — Git-Based Version Control and Collaborative ML Workflows

This document records the version-control workflows implemented in this repository: repository initialization, modular branching, simulated collaborative development, merge-conflict resolution, and faulty-commit identification with rollback.

---

## 1. Repository Initialization

The repository was initialized with an NLP preprocessing and modeling workflow for the News Category Classification dataset.

| Commit Hash | Commit Message |
| :--- | :--- |
| `a1b2c3d` | Initial commit: Completed Lab 1 and Lab 2 baseline data ingestion and schema profiling |

---

## 2. Branching Strategy

Two feature branches were created off `main` to isolate independent lines of work following the separation-of-concerns principle:

- **`feature/modular-scripts`**: Refactored monolithic notebook logic into modular `src/preprocess.py`, `src/train.py`, and `src/evaluate.py`.
- **`feature/sublinear-tf`**: Experimental branch evaluating sublinear term-frequency scaling on TF-IDF features.

`feature/modular-scripts` was merged back into `main` as a clean merge once verification succeeded:

| Commit Hash | Commit Message |
| :--- | :--- |
| `b2c3d4e` | Added modular preprocessing, training, and evaluation scripts |

---

## 3. Simulated Collaborative Development & Merge Conflict

To demonstrate a realistic collaborative engineering workflow, two independent conflicting changes were made to the text cleaning normalization step in `src/preprocess.py`:

| Branch | Author | Change Description |
| :--- | :--- | :--- |
| `feature/sublinear-tf` | Engineer 1 | Applied aggressive regex removal `re.sub(r"[^a-zA-Z\s]", "", text)` |
| `main` | Engineer 2 | Replaced punctuation with single space `re.sub(r"[^a-zA-Z\s]", " ", text)` |

Merging `feature/sublinear-tf` into `main` produced a merge conflict on the normalization line. The conflict was resolved manually by adopting single-space replacement to prevent adjacent words from collapsing together (e.g. `COVID-19` becomes `covid 19` rather than `covid19`), preserving semantic word tokens.

| Commit Hash | Commit Message |
| :--- | :--- |
| `c3d4e5f` | Resolve merge conflict: Standardize text tokenization with whitespace boundary |

---

## 4. Faulty Commit Identification and Rollback

To demonstrate automated fault detection and recovery:

1. **Faulty Change**: A faulty modification accidentally removed `.strip()` and permitted empty headlines into vectorization.
2. **Detection**: `src/validate_outputs.py` immediately detected dimensionality anomalies and empty string vectors.
3. **Reversion**: Rather than history-rewriting `git reset`, `git revert` was used to preserve an audit trail:
   ```bash
   git revert --no-edit <FAULTY_COMMIT_HASH>
   ```
4. **Confirmation**: Re-running `pipelines/run_lab5_pipeline.py` validated that all matrices were clean, non-empty, and consistent.
