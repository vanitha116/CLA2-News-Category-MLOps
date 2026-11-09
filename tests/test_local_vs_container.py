from api.main import loader

def test_local_model_consistency():
    assert loader.is_ready() is True

    test_article = "Artificial Intelligence Models Accelerate Genetic Medicine Discoveries"
    res1 = loader.predict_one(test_article)
    res2 = loader.predict_one(test_article)

    # Predictions must be deterministic
    assert res1["predicted_category"] == res2["predicted_category"]
    print(f"[TEST] Model consistency verified: {res1['predicted_category']}")
