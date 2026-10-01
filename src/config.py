import os
from pathlib import Path

# Suppress Pandera pandas import deprecation warning
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = BASE_DIR / "News_Category_Dataset_v3.json"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure runtime directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# MLflow Tracking Configuration
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "News_Category_Classification"
MODEL_REGISTRY_NAME = "News_Category_Classifier"

# Execution & Sampling Parameters
# Setting SAMPLE_SIZE = 25000 allows rapid execution (~30-60 seconds per run) in VS Code
# Set SAMPLE_SIZE = None to train on the entire 209,527 articles
SAMPLE_SIZE = 25000
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Canonical 42 News Categories in News_Category_Dataset_v3
ALL_CATEGORIES = [
    "ARTS", "ARTS & CULTURE", "BLACK VOICES", "BUSINESS", "COLLEGE", "COMEDY",
    "CRIME", "CULTURE & ARTS", "DIVORCE", "EDUCATION", "ENTERTAINMENT", "ENVIRONMENT",
    "FIFTY", "FOOD & DRINK", "GOOD NEWS", "GREEN", "HEALTHY LIVING", "HOME & LIVING",
    "IMPACT", "LATINO VOICES", "MEDIA", "MONEY", "PARENTING", "PARENTS", "POLITICS",
    "QUEER VOICES", "RELIGION", "SCIENCE", "SPORTS", "STYLE", "STYLE & BEAUTY",
    "TASTE", "TECH", "THE WORLDPOST", "TRAVEL", "U.S. NEWS", "WEDDINGS", "WEIRD NEWS",
    "WELLNESS", "WOMEN", "WORLD NEWS", "WORLDPOST"
]
