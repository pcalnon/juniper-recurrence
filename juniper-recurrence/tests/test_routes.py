"""Route tests for the juniper-recurrence API (plan §6 / §12).

Regression-generic throughout (RK-6): no ``accuracy`` key in any response, no
``argmax`` in any router. The data path is faked (synthetic 3-D arrays) so these
exercise the real model + synchronous lifecycle + serialization end-to-end.
"""

from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

import juniper_recurrence
from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings


def _client(**settings_kwargs) -> TestClient:
    return TestClient(build_app(Settings(**settings_kwargs)))


# --- train + status -------------------------------------------------------------------


def test_train_happy_path(fake_data):
    resp = _client(api_keys=None).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_epochs"] == 1
    assert body["stopped_reason"] == "converged"
    assert set(body["final_metrics"]) >= {"mse", "rmse", "mae", "r2", "loss"}
    assert "accuracy" not in body["final_metrics"]  # RK-6
    assert body["dataset"]["dataset_id"] == "ds-1"
    assert body["dataset"]["n_windows"] == 12
    assert body["dataset"]["n_features"] == 2
    assert body["dataset"]["has_target_dt"] is True


def test_status_idle_then_trained(fake_data):
    client = _client(api_keys=None)
    idle = client.get("/v1/training/status").json()
    assert idle["state"] == "idle"
    assert idle["events"] == []

    client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    trained = client.get("/v1/training/status").json()
    assert trained["state"] == "trained"
    assert [e["type"] for e in trained["events"]] == ["training_start", "epoch_end", "training_end"]
    seqs = [e["seq"] for e in trained["events"]]
    assert seqs == sorted(seqs)  # monotonic seq stamped by the lifecycle
    assert trained["events"][1]["payload"]["epoch"] == 0


def test_train_with_hyperparams(fake_data):
    client = _client(api_keys=None)
    resp = client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "d": 4, "ridge": 0.1})
    assert resp.status_code == 200
    topology = client.get("/v1/model").json()["topology"]
    assert topology["meta"]["d"] == 4


def test_train_409_when_lock_held(fake_data):
    app = build_app(Settings(api_keys=None))
    app.state.app_state.train_lock.acquire()  # simulate an in-progress run
    try:
        resp = TestClient(app).post("/v1/train", json={"dataset": {"dataset_id": "x"}})
        assert resp.status_code == 409
    finally:
        app.state.app_state.train_lock.release()


def test_train_invalid_dataset_ref_422(fake_data):
    resp = _client(api_keys=None).post("/v1/train", json={"dataset": {}})
    assert resp.status_code == 422  # DatasetRef validator: needs id/name/generator


# --- predict --------------------------------------------------------------------------


def test_predict_before_train_409():
    resp = _client(api_keys=None).post("/v1/predict", json={"X": [[[0.0, 0.0]]]})
    assert resp.status_code == 409


def test_predict_inline_with_dt(fake_data):
    client = _client(api_keys=None)
    client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    resp = client.post(
        "/v1/predict",
        json={
            "X": fake_data["X_train"].tolist(),
            "dt": fake_data["dt_train"].tolist(),
            "target_dt": fake_data["target_dt_train"].tolist(),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shape"] == [12, 1]
    assert len(body["predictions"]) == 12


def test_predict_via_dataset_ref(fake_data):
    client = _client(api_keys=None)
    client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    resp = client.post("/v1/predict", json={"dataset": {"dataset_id": "ds-1", "split": "train"}})
    assert resp.status_code == 200
    assert resp.json()["shape"][0] == 12


def test_predict_requires_x_or_dataset(fake_data):
    client = _client(api_keys=None)
    client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    assert client.post("/v1/predict", json={}).status_code == 422


def test_predict_bad_shape_422(fake_data):
    client = _client(api_keys=None)
    client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    # 2-D X violates the model's (n, T, F) contract -> ValueError -> 422.
    assert client.post("/v1/predict", json={"X": [[0.0, 0.0]]}).status_code == 422


# --- model + dataset ------------------------------------------------------------------


def test_model_route(fake_data):
    client = _client(api_keys=None)
    assert client.get("/v1/model").status_code == 409
    client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    body = client.get("/v1/model").json()
    assert body["topology"]["model_type"] == "lmu"
    assert "accuracy" not in body["metrics"]  # RK-6
    assert set(body["metrics"]) >= {"mse", "r2"}


def test_dataset_route(fake_data):
    client = _client(api_keys=None)
    assert client.get("/v1/dataset").status_code == 409
    client.post("/v1/train", json={"dataset": {"dataset_id": "ds-9", "split": "train"}})
    body = client.get("/v1/dataset").json()
    assert body["dataset_id"] == "ds-9"
    assert body["split"] == "train"
    assert body["n_windows"] == 12


# --- data-error mapping ---------------------------------------------------------------


def _client_with_download_error(monkeypatch, exc: Exception) -> TestClient:
    class _Client:
        def __init__(self, **kwargs):
            pass

        def get_latest(self, name):
            return {"dataset_id": "x"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "x"}

        def download_artifact_npz(self, dataset_id):
            raise exc

        def close(self):
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _Client)
    return _client(api_keys=None)


def test_train_not_found_404(monkeypatch):
    from juniper_data_client import JuniperDataNotFoundError

    client = _client_with_download_error(monkeypatch, JuniperDataNotFoundError("missing"))
    assert client.post("/v1/train", json={"dataset": {"dataset_id": "missing"}}).status_code == 404


def test_train_upstream_unreachable_502(monkeypatch):
    from juniper_data_client import JuniperDataConnectionError

    client = _client_with_download_error(monkeypatch, JuniperDataConnectionError("down"))
    assert client.post("/v1/train", json={"dataset": {"dataset_id": "x"}}).status_code == 502


def test_train_bad_contract_422(fake_data, monkeypatch):
    def _raise(arrays, **kw):
        raise ValueError("not a sequence")

    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", _raise)
    assert _client(api_keys=None).post("/v1/train", json={"dataset": {"dataset_id": "x"}}).status_code == 422


def test_map_data_error_covers_all_branches():
    from juniper_data_client import (
        JuniperDataClientError,
        JuniperDataConfigurationError,
        JuniperDataConnectionError,
        JuniperDataNotFoundError,
        JuniperDataTimeoutError,
        JuniperDataValidationError,
    )

    from juniper_recurrence.routers._common import map_data_error

    assert map_data_error(JuniperDataNotFoundError("x")).status_code == 404
    assert map_data_error(JuniperDataConnectionError("x")).status_code == 502
    assert map_data_error(JuniperDataTimeoutError("x")).status_code == 502
    assert map_data_error(JuniperDataValidationError("x")).status_code == 422
    assert map_data_error(ValueError("x")).status_code == 422
    assert map_data_error(JuniperDataConfigurationError("x")).status_code == 500
    assert map_data_error(JuniperDataClientError("x")).status_code == 502


# --- security + rate limit (plan §12) -------------------------------------------------


def test_routes_require_api_key(fake_data):
    client = _client(api_keys=["s3cret"])
    assert client.get("/v1/model").status_code == 401
    assert client.post("/v1/train", json={"dataset": {"dataset_id": "x"}}).status_code == 401
    # With the key, requests reach the handlers.
    assert client.get("/v1/model", headers={"X-API-Key": "s3cret"}).status_code == 409
    ok = client.post("/v1/train", json={"dataset": {"dataset_id": "x"}}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_rate_limit_429():
    client = _client(api_keys=None, rate_limit_requests_per_minute=2)
    assert client.get("/v1/model").status_code == 409  # allowed (1/2)
    assert client.get("/v1/model").status_code == 409  # allowed (2/2)
    assert client.get("/v1/model").status_code == 429  # exceeded


# --- RK-6 static guard ----------------------------------------------------------------


def test_no_argmax_call_in_routers():
    # RK-6: a router must never collapse continuous output to class labels via argmax.
    # We assert no argmax *call* (the ``argmax(`` syntax) — routers may still name the
    # constraint in docstrings ("never argmax"), which is documentation, not a leak.
    routers_dir = pathlib.Path(juniper_recurrence.__file__).parent / "routers"
    for module in routers_dir.glob("*.py"):
        assert "argmax(" not in module.read_text(), f"argmax() call in {module.name} violates RK-6"
