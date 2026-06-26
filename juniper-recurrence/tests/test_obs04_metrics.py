"""Tests for OBS-04 domain Prometheus metrics (``juniper_recurrence.metrics``).

Train / predict / crossval counters + last-value gauges, asserted against the default
``prometheus_client`` REGISTRY (where ``register_or_reuse`` registers them) and
end-to-end through a ``/metrics`` scrape after a real (faked-data) training run.

Like ``test_metrics.py`` this ``importorskip``s ``juniper_observability`` so it is a
no-op without the optional ``[observability]`` extra; CI installs ``.[test,observability]``.
Assertions on counters are delta-based (the default REGISTRY is process-global and shared
across tests), so they are order-independent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from juniper_recurrence import metrics
from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings

pytest.importorskip("juniper_observability")

from prometheus_client import REGISTRY  # noqa: E402  (must follow the importorskip guard)


def test_module_enabled_with_observability():
    assert metrics.ENABLED is True


def test_record_train_updates_collectors():
    before = REGISTRY.get_sample_value("juniper_recurrence_train_runs_total") or 0.0
    metrics.record_train(1.25, {"r2": 0.91, "mse": 0.04, "loss": 0.04})
    assert REGISTRY.get_sample_value("juniper_recurrence_train_runs_total") == before + 1
    assert REGISTRY.get_sample_value("juniper_recurrence_train_last_duration_seconds") == 1.25
    assert REGISTRY.get_sample_value("juniper_recurrence_train_last_metric", {"metric": "r2"}) == 0.91
    assert REGISTRY.get_sample_value("juniper_recurrence_train_last_metric", {"metric": "mse"}) == 0.04


def test_record_predict_increments_counter():
    before = REGISTRY.get_sample_value("juniper_recurrence_predict_requests_total") or 0.0
    metrics.record_predict()
    metrics.record_predict()
    assert REGISTRY.get_sample_value("juniper_recurrence_predict_requests_total") == before + 2


def test_record_crossval_updates_collectors():
    before = REGISTRY.get_sample_value("juniper_recurrence_crossval_runs_total") or 0.0
    metrics.record_crossval(2.5, {"r2": 0.8})
    assert REGISTRY.get_sample_value("juniper_recurrence_crossval_runs_total") == before + 1
    assert REGISTRY.get_sample_value("juniper_recurrence_crossval_last_duration_seconds") == 2.5
    assert REGISTRY.get_sample_value("juniper_recurrence_crossval_last_metric", {"metric": "r2"}) == 0.8


def test_non_numeric_metric_values_skipped():
    # A non-numeric metric value must not raise or create a labelled series.
    metrics.record_train(0.5, {"note": "n/a", "r2": 0.7})
    assert REGISTRY.get_sample_value("juniper_recurrence_train_last_metric", {"metric": "note"}) is None
    assert REGISTRY.get_sample_value("juniper_recurrence_train_last_metric", {"metric": "r2"}) == 0.7


def test_record_helpers_noop_when_disabled(monkeypatch):
    # Simulate the [observability] extra being absent: the helpers must be safe no-ops.
    monkeypatch.setattr(metrics, "ENABLED", False)
    metrics.record_train(9.0, {"r2": 0.1})  # must not raise
    metrics.record_predict()
    metrics.record_crossval(9.0, {"r2": 0.1})


def test_train_route_surfaces_domain_metrics(fake_data):
    # End-to-end: a real (faked-data) train bumps the counter and the scrape exposes it.
    app = build_app(Settings(api_keys=None))
    client = TestClient(app, client=("127.0.0.1", 12345))
    before = REGISTRY.get_sample_value("juniper_recurrence_train_runs_total") or 0.0
    assert client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}}).status_code == 200
    assert REGISTRY.get_sample_value("juniper_recurrence_train_runs_total") == before + 1
    scrape = client.get("/metrics")
    assert scrape.status_code == 200
    assert "juniper_recurrence_train_runs_total" in scrape.text
    assert "juniper_recurrence_train_last_metric" in scrape.text
