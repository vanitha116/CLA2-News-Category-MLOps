import builtins
import json

import joblib
import yaml


def test_packaged_deployment_bundle_does_not_require_mlflow(
    tmp_path, monkeypatch, test_model_artifacts
):
    from api.model_loader import ModelLoader

    model, preprocessor = test_model_artifacts
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deployment").mkdir()
    joblib.dump(model, tmp_path / "deployment" / "model.pkl")
    joblib.dump(preprocessor, tmp_path / "deployment" / "preprocessor.pkl")
    (tmp_path / "deployment" / "model_metadata.json").write_text(
        json.dumps({"version": 7, "model_name": "news-test"}),
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "mlflow": {"tracking_uri": "sqlite:///mlflow.db"},
                "registry": {"model_name": "news-test"},
                "paths": {
                    "model": "models/news_model.pkl",
                    "preprocessor": "models/preprocessor.pkl",
                },
            }
        ),
        encoding="utf-8",
    )

    original_import = builtins.__import__

    def import_without_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("MLflow is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mlflow)
    loader = ModelLoader()

    assert loader.is_ready()
    assert loader.model_version == "7"
    assert loader.metadata["model_name"] == "news-test"
