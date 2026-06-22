"""Extra coverage: timeout / generic-request-error mapping, full predict & crossval bodies,
readiness polling (success + timeout), and the api-key file-read-error fallback."""

from __future__ import annotations

import json

import pytest
import requests
import responses

from juniper_recurrence_client import (
    JuniperRecurrenceClient,
    JuniperRecurrenceClientError,
    JuniperRecurrenceTimeoutError,
)
from juniper_recurrence_client.constants import API_KEY_ENV_VAR, API_KEY_FILE_ENV_VAR, API_KEY_HEADER_NAME

BASE_URL = "http://recurrence.test:8211"


def _client(**kwargs: object) -> JuniperRecurrenceClient:
    kwargs.setdefault("retries", 0)
    return JuniperRecurrenceClient(base_url=BASE_URL, **kwargs)


@responses.activate
def test_timeout_maps_to_timeout_error() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/model", body=requests.exceptions.Timeout("slow"))
    with pytest.raises(JuniperRecurrenceTimeoutError):
        _client().get_model()


@responses.activate
def test_generic_request_exception_maps_to_client_error() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/dataset", body=requests.exceptions.RequestException("boom"))
    with pytest.raises(JuniperRecurrenceClientError):
        _client().get_dataset()


@responses.activate
def test_predict_full_aux_body() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/predict", json={"predictions": [], "shape": [0, 1]}, status=200)
    _client().predict(X=[[[1.0]]], dt=[[0.0]], target_dt=[1.0], seq_lengths=[1])
    sent = json.loads(responses.calls[0].request.body)
    assert sent["target_dt"] == [1.0] and sent["seq_lengths"] == [1]


@responses.activate
def test_crossval_passes_hyperparams() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"task_type": "regression", "n_folds": 2, "folds": [], "eval_aggregate": {}, "eval_std": {}, "dataset": {}}, status=200)
    _client().crossval(generator="equities_seq", n_folds=2, d=8, theta=1.5, ridge=0.2)
    sent = json.loads(responses.calls[0].request.body)
    assert sent["dataset"]["generator"] == "equities_seq"
    assert sent["d"] == 8 and sent["theta"] == 1.5 and sent["ridge"] == 0.2


@responses.activate
def test_train_forwards_gcv_ridge() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/train", json={"final_metrics": {}, "n_epochs": 1, "stopped_reason": None, "dataset": {}}, status=200)
    _client().train(name="equities", ridge="gcv")
    sent = json.loads(responses.calls[0].request.body)
    assert sent["ridge"] == "gcv"


@responses.activate
def test_crossval_forwards_gcv_ridge() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"task_type": "regression", "n_folds": 2, "folds": [], "eval_aggregate": {}, "eval_std": {}, "dataset": {}}, status=200)
    _client().crossval(generator="equities_seq", n_folds=2, ridge="gcv")
    sent = json.loads(responses.calls[0].request.body)
    assert sent["ridge"] == "gcv"


@responses.activate
def test_train_forwards_readout_rff() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/train", json={"final_metrics": {}, "n_epochs": 1, "stopped_reason": None, "dataset": {}}, status=200)
    _client().train(name="equities", readout="rff", rff_features=64, rff_gamma="median")
    sent = json.loads(responses.calls[0].request.body)
    assert sent["readout"] == "rff" and sent["rff_features"] == 64 and sent["rff_gamma"] == "median"


@responses.activate
def test_crossval_forwards_readout_rff() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"task_type": "regression", "n_folds": 2, "folds": [], "eval_aggregate": {}, "eval_std": {}, "dataset": {}}, status=200)
    _client().crossval(generator="equities_seq", n_folds=2, readout="rff", rff_features=128, rff_gamma=0.5)
    sent = json.loads(responses.calls[0].request.body)
    assert sent["readout"] == "rff" and sent["rff_features"] == 128 and sent["rff_gamma"] == 0.5


@responses.activate
def test_wait_for_ready_polls_until_ready() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/health/ready", json={"status": "starting"}, status=200)
    responses.add(responses.GET, f"{BASE_URL}/v1/health/ready", json={"status": "ready"}, status=200)
    assert _client().wait_for_ready(timeout=2.0, poll_interval=0.01) is True


@responses.activate
def test_wait_for_ready_times_out() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/health/ready", json={"status": "starting"}, status=200)
    assert _client().wait_for_ready(timeout=0.03, poll_interval=0.01) is False


def test_api_key_file_read_error_falls_back_to_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # _FILE points at a missing path -> OSError on read -> fall back to the plain env var.
    monkeypatch.setenv(API_KEY_FILE_ENV_VAR, str(tmp_path / "does-not-exist"))
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    client = JuniperRecurrenceClient(base_url=BASE_URL)
    assert client.session.headers[API_KEY_HEADER_NAME] == "env-key"
