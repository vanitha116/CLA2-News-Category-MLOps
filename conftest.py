"""
conftest.py - Pytest configuration for the News Category Classification MLOps Project.
Ensures the project root is on sys.path for all tests so imports like
`from api.main import app` work correctly.
"""
import sys
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Insert project root so `api.*`, `src.*` packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def test_model_artifacts():
    texts = [
        "astronomy telescope galaxy stars science",
        "scientists discover planets using a telescope",
        "stocks bank inflation market finance",
        "business markets report corporate earnings",
        "football team goal championship sports",
        "athletes win the league after a close game",
    ]
    labels = ["SCIENCE", "SCIENCE", "BUSINESS", "BUSINESS", "SPORTS", "SPORTS"]
    preprocessor = TfidfVectorizer().fit(texts)
    model = MultinomialNB().fit(preprocessor.transform(texts), labels)
    return model, preprocessor


@pytest.fixture(autouse=True)
def isolated_api_model(monkeypatch, test_model_artifacts):
    from api.main import loader

    loader.model, loader.preprocessor = test_model_artifacts
    loader.model_version = "test"
    loader.metadata = {"model_name": "test-model", "version": "test"}
    monkeypatch.setattr("api.main.log_prediction_event", lambda **kwargs: None)
