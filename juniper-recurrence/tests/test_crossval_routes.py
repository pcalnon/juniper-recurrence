"""Route tests for the cross-validation endpoint (``POST /v1/crossval`` + status).

Regression-generic (RK-6): no ``accuracy`` key, no ``argmax``. The data path is faked with a
synthetic 3-D ``_full`` artifact, so these exercise the real fold executor + LMURegressor end to
end. The status test covers the persisted-result behavior (the most recent CV result is held in
the in-process app state and returned by ``GET /v1/crossval/status``).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings


def _client(**settings_kwargs) -> TestClient:
    return TestClient(build_app(Settings(**settings_kwargs)))


@pytest.fixture
def full_npz_arrays() -> dict[str, np.ndarray]:
    """A contract-valid 3-D sequence NPZ mapping for the ``full`` split (20 windows)."""
    rng = np.random.default_rng(0)
    n_windows, lookback, n_features = 20, 5, 2
    X = rng.standard_normal((n_windows, lookback, n_features)).astype("float32")
    dt = np.zeros((n_windows, lookback), dtype="float32")
    dt[:, 1:] = rng.uniform(0.5, 2.0, size=(n_windows, lookback - 1)).astype("float32")
    y = rng.standard_normal((n_windows,)).astype("float32")
    target_dt = rng.uniform(0.5, 2.0, size=(n_windows,)).astype("float32")
    seq_lengths = np.full((n_windows,), lookback, dtype="int64")
    return {
        "X_full": X,
        "y_reg_full": y,
        "dt_full": dt,
        "target_dt_full": target_dt,
        "seq_lengths_full": seq_lengths,
    }


@pytest.fixture
def fake_full_data(monkeypatch, full_npz_arrays) -> dict[str, np.ndarray]:
    """Patch the data adapter to serve the ``_full`` arrays (no live juniper-data)."""

    class _FakeDataClient:
        def __init__(self, **kwargs) -> None:
            self.closed = False

        def get_latest(self, name):
            return {"dataset_id": f"latest-of-{name}"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "created-1"}

        def download_artifact_npz(self, dataset_id):
            return full_npz_arrays

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _FakeDataClient)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda arrays, **kw: "sequence")
    return full_npz_arrays


def test_crossval_happy_path(fake_full_data):
    resp = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_type"] == "regression"
    assert body["n_folds"] == 3
    assert len(body["folds"]) == 3
    assert [f["fold"] for f in body["folds"]] == [0, 1, 2]
    assert set(body["eval_aggregate"]) == {"mse", "rmse", "mae", "r2", "loss"}
    assert "accuracy" not in body["eval_aggregate"]  # RK-6
    assert set(body["eval_std"]) == set(body["eval_aggregate"])
    assert all(f["n_epochs"] == 1 for f in body["folds"])
    # CV always operates on the full chronological split.
    assert body["dataset"]["split"] == "full"
    assert body["dataset"]["n_windows"] == 20


def test_crossval_readout_rff(fake_full_data):
    # DP-3 P2c: the RFF nonlinear readout is reachable over /v1/crossval.
    resp = _client(api_keys=None).post(
        "/v1/crossval",
        json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4, "readout": "rff", "rff_features": 32},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_folds"] == 3


def test_crossval_rff_params_without_readout_422(fake_full_data):
    # rff_* params are rejected by the request schema unless readout="rff".
    resp = _client(api_keys=None).post(
        "/v1/crossval",
        json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "rff_gamma": 0.5},
    )
    assert resp.status_code == 422


def test_crossval_status_idle_then_persisted(fake_full_data):
    client = _client(api_keys=None)
    idle = client.get("/v1/crossval/status").json()
    assert idle["state"] == "idle"
    assert idle["result"] is None

    posted = client.post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4}).json()

    done = client.get("/v1/crossval/status").json()
    assert done["state"] == "done"
    assert done["result"] is not None
    # the persisted result matches what the POST returned (most-recent-result persistence)
    assert done["result"]["n_folds"] == 3
    assert done["result"]["eval_aggregate"] == posted["eval_aggregate"]
    assert done["result"]["dataset"]["split"] == "full"


def test_crossval_rolling_with_embargo_and_min_train(fake_full_data):
    resp = _client(api_keys=None).post(
        "/v1/crossval",
        json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4, "scheme": "rolling", "embargo": 1, "min_train": 2},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_folds"] == 3


def test_crossval_409_when_lock_held(fake_full_data):
    app = build_app(Settings(api_keys=None))
    app.state.app_state.crossval_lock.acquire()  # simulate an in-progress CV run
    try:
        resp = TestClient(app).post("/v1/crossval", json={"dataset": {"dataset_id": "x"}, "n_folds": 3})
        assert resp.status_code == 409
    finally:
        app.state.app_state.crossval_lock.release()


def test_crossval_too_many_folds_422(fake_full_data):
    # n_folds passes the pydantic ge=2 gate but exceeds what 20 windows support -> walk_forward_folds
    # raises ValueError -> mapped to 422.
    resp = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "x"}, "n_folds": 50})
    assert resp.status_code == 422


def test_crossval_n_folds_below_minimum_422(fake_full_data):
    # n_folds < 2 is rejected by the request schema (Field ge=2).
    resp = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "x"}, "n_folds": 1})
    assert resp.status_code == 422


def test_crossval_invalid_dataset_ref_422(fake_full_data):
    resp = _client(api_keys=None).post("/v1/crossval", json={"dataset": {}, "n_folds": 3})
    assert resp.status_code == 422  # DatasetRef validator: needs id/name/generator


def test_crossval_not_found_404(monkeypatch):
    from juniper_data_client import JuniperDataNotFoundError

    class _Client:
        def __init__(self, **kwargs):
            pass

        def get_latest(self, name):
            return {"dataset_id": "x"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "x"}

        def download_artifact_npz(self, dataset_id):
            raise JuniperDataNotFoundError("missing")

        def close(self):
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _Client)
    resp = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "missing"}, "n_folds": 3})
    assert resp.status_code == 404


def test_crossval_requires_api_key(fake_full_data):
    client = _client(api_keys=["s3cret"])
    assert client.get("/v1/crossval/status").status_code == 401
    assert client.post("/v1/crossval", json={"dataset": {"dataset_id": "x"}, "n_folds": 3}).status_code == 401
    ok = client.post("/v1/crossval", json={"dataset": {"dataset_id": "x"}, "n_folds": 3, "d": 4}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
