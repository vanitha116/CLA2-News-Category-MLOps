import json
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data


def test_single_prediction_valid():
    payload = {
        "headline": "Astronomers Discover New Exoplanet in Habitable Zone",
        "short_description": "The Earth-sized world orbits a nearby star in the goldilocks zone.",
        "authors": "Science Reporter",
        "date": "2026-10-15",
        "link": "https://www.huffpost.com/entry/science-exoplanet"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_category" in data["data"]
    assert data["data"]["predicted_category"] != ""


def test_single_prediction_invalid_headline():
    # Headline shorter than 3 characters should return 422
    payload = {
        "headline": "ab",
        "short_description": "Too short"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "Request validation failed" in data["message"]


def test_batch_prediction_valid():
    payload = {
        "articles": [
            {
                "headline": "Stock Markets Reach All Time High as Central Banks Cut Rates",
                "short_description": "Financial analysts report strong earnings across multiple sectors."
            },
            {
                "headline": "Underdog Team Triumphs in World Cup Final Overtime Thriller",
                "short_description": "Spectators witness historic victory in the closing minutes of play."
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
