import hashlib
from importlib import import_module
import json

monitor_trigger = import_module("src.mlops_pipeline.12_monitor_trigger")


def _write_batch_and_lineage(tmp_path):
    batch_path = tmp_path / "incoming.json"
    batch_path.write_text('{"headline":"new story"}\n', encoding="utf-8")
    batch_hash = hashlib.sha256(batch_path.read_bytes()).hexdigest()

    lineage_path = tmp_path / "dataset_lineage.json"
    lineage_path.write_text(
        json.dumps([{"sha256": batch_hash}]),
        encoding="utf-8",
    )
    return batch_path, lineage_path


def test_batch_pending_detection(tmp_path, monkeypatch):
    batch_path, lineage_path = _write_batch_and_lineage(tmp_path)
    monkeypatch.setattr(monitor_trigger, "NEW_DATA_PATH", batch_path)
    monkeypatch.setattr(monitor_trigger, "DATASET_LINEAGE_PATH", lineage_path)

    assert monitor_trigger.has_pending_batch() is False

    batch_path.write_text('{"headline":"another story"}\n', encoding="utf-8")
    assert monitor_trigger.has_pending_batch() is True


def test_monitor_does_not_retrain_for_already_ingested_batch(tmp_path, monkeypatch):
    batch_path, lineage_path = _write_batch_and_lineage(tmp_path)
    report_path = tmp_path / "monitor_trigger_report.json"
    monkeypatch.setattr(monitor_trigger, "NEW_DATA_PATH", batch_path)
    monkeypatch.setattr(monitor_trigger, "DATASET_LINEAGE_PATH", lineage_path)
    monkeypatch.setattr(monitor_trigger, "TRIGGER_REPORT", report_path)
    monkeypatch.setattr(monitor_trigger, "run_drift_analysis", lambda: True)
    monkeypatch.setattr(monitor_trigger, "detect_drift", lambda: True)
    monkeypatch.setattr(
        monitor_trigger.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pipeline must not run")),
    )

    assert monitor_trigger.run_monitor_trigger() is True

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "waiting_for_new_data"
    assert report["new_data_present"] is True
    assert report["new_data_pending"] is False
    assert report["pipeline_called"] is False
    assert report["retraining_executed"] is False
