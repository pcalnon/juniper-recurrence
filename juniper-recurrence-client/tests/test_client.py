"""Unit tests for ``JuniperRecurrenceClient`` (HTTP mocked with ``responses``)."""

from __future__ import annotations

import json

import pytest
import responses

from juniper_recurrence_client import JuniperRecurrenceClient

BASE_URL = "http://recurrence.test:8211"


def _client(**kwargs: object) -> JuniperRecurrenceClient:
    kwargs.setdefault("retries", 0)
    return JuniperRecurrenceClient(base_url=BASE_URL, **kwargs)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://recurrence.test:8211", "http://recurrence.test:8211"),
        ("http://recurrence.test:8211/", "http://recurrence.test:8211"),
        ("http://recurrence.test:8211/v1", "http://recurrence.test:8211"),
        ("recurrence.test:8211", "http://recurrence.test:8211"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert JuniperRecurrenceClient(base_url=raw).base_url == expected


@responses.activate
def test_train_posts_dataset_ref_and_hyperparams() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/train", json={"final_metrics": {"r2": 0.9}, "n_epochs": 1, "stopped_reason": None, "dataset": {}}, status=200)
    out = _client().train(name="equities", d=16, theta=2.0, ridge=0.1)
    assert out["final_metrics"]["r2"] == 0.9
    sent = json.loads(responses.calls[0].request.body)
    assert sent["dataset"] == {"split": "train", "name": "equities"}
    assert sent["d"] == 16 and sent["theta"] == 2.0 and sent["ridge"] == 0.1


@responses.activate
def test_training_status() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/training/status", json={"state": "trained", "final_metrics": {"r2": 0.9}, "stopped_reason": None, "events": []}, status=200)
    assert _client().training_status()["state"] == "trained"


@responses.activate
def test_predict_inline_x() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/predict", json={"predictions": [[1.0]], "shape": [1, 1]}, status=200)
    out = _client().predict(X=[[[1.0, 2.0]]], dt=[[0.0]])
    assert out["shape"] == [1, 1]
    sent = json.loads(responses.calls[0].request.body)
    assert sent["X"] == [[[1.0, 2.0]]] and sent["dt"] == [[0.0]] and "dataset" not in sent


@responses.activate
def test_predict_by_dataset_ref() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/predict", json={"predictions": [], "shape": [0, 1]}, status=200)
    _client().predict(dataset_id="ds-1")
    sent = json.loads(responses.calls[0].request.body)
    assert sent["dataset"] == {"split": "train", "dataset_id": "ds-1"} and "X" not in sent


@responses.activate
def test_crossval_passes_config() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"task_type": "regression", "n_folds": 3, "folds": [], "eval_aggregate": {}, "eval_std": {}, "dataset": {}}, status=200)
    out = _client().crossval(name="equities", n_folds=3, scheme="rolling", embargo=2, min_train=10)
    assert out["n_folds"] == 3
    sent = json.loads(responses.calls[0].request.body)
    assert sent["n_folds"] == 3 and sent["scheme"] == "rolling" and sent["embargo"] == 2 and sent["min_train"] == 10


@responses.activate
def test_crossval_status_model_dataset() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/crossval/status", json={"state": "done", "result": None}, status=200)
    responses.add(responses.GET, f"{BASE_URL}/v1/model", json={"topology": {"model_type": "lmu"}, "metrics": {}}, status=200)
    responses.add(responses.GET, f"{BASE_URL}/v1/dataset", json={"dataset_id": "ds-1", "split": "train"}, status=200)
    c = _client()
    assert c.crossval_status()["state"] == "done"
    assert c.get_model()["topology"]["model_type"] == "lmu"
    assert c.get_dataset()["dataset_id"] == "ds-1"


@responses.activate
def test_health_and_is_ready() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/health", json={"status": "ok"}, status=200)
    responses.add(responses.GET, f"{BASE_URL}/v1/health/ready", json={"status": "ready"}, status=200)
    c = _client()
    assert c.health_check()["status"] == "ok"
    assert c.is_ready() is True


@responses.activate
def test_is_ready_false_when_not_ready() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/health/ready", json={"status": "starting"}, status=200)
    assert _client().is_ready() is False


@responses.activate
def test_on_request_hook_fires_once() -> None:
    seen: list[tuple[str, object, object, object]] = []

    def hook(method: str, url: str, status: object, duration_ms: float, error: object) -> None:
        seen.append((method, url, status, error))

    responses.add(responses.GET, f"{BASE_URL}/v1/model", json={"topology": {}, "metrics": {}}, status=200)
    _client(on_request=hook).get_model()
    assert len(seen) == 1
    assert seen[0][0] == "GET" and seen[0][2] == 200 and seen[0][3] is None


def test_context_manager_closes() -> None:
    with _client() as client:
        assert client.session is not None
