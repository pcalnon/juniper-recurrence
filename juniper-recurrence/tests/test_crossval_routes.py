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
    """A contract-valid 3-D sequence NPZ mapping: a ``train`` partition of 40 windows (the ``trainval`` pool is train | val; with no ``val`` it is the train set).

    40 windows against a look-back of 5 leaves every one of three folds a training set under the
    W5.2 default embargo (one look-back): fold 0 trains on 10 − 5 rows.
    """
    rng = np.random.default_rng(0)
    n_windows, lookback, n_features = 40, 5, 2
    X = rng.standard_normal((n_windows, lookback, n_features)).astype("float32")
    dt = np.zeros((n_windows, lookback), dtype="float32")
    dt[:, 1:] = rng.uniform(0.5, 2.0, size=(n_windows, lookback - 1)).astype("float32")
    y = rng.standard_normal((n_windows,)).astype("float32")
    target_dt = rng.uniform(0.5, 2.0, size=(n_windows,)).astype("float32")
    seq_lengths = np.full((n_windows,), lookback, dtype="int64")
    return {
        "X_train": X,
        "y_reg_train": y,
        "dt_train": dt,
        "target_dt_train": target_dt,
        "seq_lengths_train": seq_lengths,
    }


@pytest.fixture
def three_partition_npz_arrays() -> dict[str, np.ndarray]:
    """A two-ticker artifact with train / val / test partitions and the W5.2 keys, laid down split-major as juniper-data does."""
    rng = np.random.default_rng(1)
    lookback, n_features = 5, 2
    counts = {"train": 24, "val": 8, "test": 8}
    out: dict[str, np.ndarray] = {}
    for split, per_ticker in counts.items():
        n = 2 * per_ticker
        out[f"X_{split}"] = rng.standard_normal((n, lookback, n_features)).astype("float32")
        dt = np.zeros((n, lookback), dtype="float32")
        dt[:, 1:] = 1.0
        out[f"dt_{split}"] = dt
        out[f"y_reg_{split}"] = rng.standard_normal((n,)).astype("float32")
        out[f"target_dt_{split}"] = np.ones((n,), dtype="float32")
        out[f"seq_lengths_{split}"] = np.full((n,), lookback, dtype="int64")
        out[f"ticker_code_{split}"] = np.repeat(np.array([0, 1], dtype=np.int32), per_ticker)
    # Consecutive YYYYMMDD days per ticker, train before val before test, the same calendar for both tickers.
    base = np.datetime64("2020-01-01")
    offset = 0
    for split, per_ticker in counts.items():
        days = (base + np.arange(offset, offset + per_ticker).astype("timedelta64[D]")).astype("datetime64[D]")
        ymd = np.asarray([int(str(d)[0:4]) * 10000 + int(str(d)[5:7]) * 100 + int(str(d)[8:10]) for d in days], dtype=np.int32)
        out[f"window_end_date_{split}"] = np.tile(ymd, 2)
        offset += per_ticker
    return out


@pytest.fixture
def fake_three_partition_data(monkeypatch, three_partition_npz_arrays) -> dict[str, np.ndarray]:
    class _FakeDataClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_latest(self, name):
            return {"dataset_id": f"latest-of-{name}"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "created-1"}

        def download_artifact_npz(self, dataset_id):
            return three_partition_npz_arrays

        def close(self) -> None:
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _FakeDataClient)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda arrays, **kw: "sequence")
    return three_partition_npz_arrays


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
    # CV selects over the trainval pool (W5.2); with no val partition that is the train set.
    assert body["dataset"]["split"] == "trainval"
    assert body["dataset"]["n_windows"] == 40
    assert body["folding"] == {"lookback": 5, "embargo": 5, "embargo_days": None, "grouped": False, "ordered": False, "n_entities": None, "n_folds": 3, "dropped_label_availability": 0, "dropped_embargo_days": 0}


def test_crossval_pool_excludes_test_and_cuts_a_panel_on_the_date(fake_three_partition_data):
    # W5.2 at the route: the pool is train | val (never test), both tickers' keys cut the folds, and the
    # response's folding block says so.
    arrays = fake_three_partition_data
    body = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4}).json()
    assert body["dataset"]["split"] == "trainval"
    assert body["dataset"]["n_windows"] == arrays["X_train"].shape[0] + arrays["X_val"].shape[0]
    folding = body["folding"]
    assert folding["grouped"] is True and folding["ordered"] is True and folding["n_entities"] == 2
    assert folding["embargo"] == 5 and folding["n_folds"] == len(body["folds"]) >= 1
    assert folding["dropped_label_availability"] >= 0


def test_crossval_embargo_days_is_accepted_and_reported(fake_three_partition_data):
    body = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 2, "d": 4, "embargo": 0, "embargo_days": 2.5}).json()
    assert body["folding"]["embargo_days"] == 2.5 and body["folding"]["embargo"] == 0


def test_crossval_embargo_days_without_dates_422(fake_full_data):
    resp = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 2, "d": 4, "embargo_days": 2.0})
    assert resp.status_code == 422
    assert "embargo_days needs" in resp.text


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


def test_crossval_readout_mlp_without_torch_503(fake_full_data, monkeypatch):
    # DP-3 P3: a torch-less deployment rejects readout="mlp" up front (the pre-flight build) with a
    # clean 503 — toggling find_spec keeps this deterministic whether or not torch is installed.
    import importlib.util as iu

    real = iu.find_spec
    monkeypatch.setattr(
        "juniper_recurrence._readout.importlib.util.find_spec",
        lambda name, *a, **k: None if name == "torch" else real(name, *a, **k),
    )
    resp = _client(api_keys=None).post(
        "/v1/crossval",
        json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4, "readout": "mlp"},
    )
    assert resp.status_code == 503, resp.text
    assert "torch" in resp.text


def test_crossval_mlp_params_without_readout_422(fake_full_data):
    # mlp_* params are rejected by the request schema unless readout="mlp".
    resp = _client(api_keys=None).post(
        "/v1/crossval",
        json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "mlp_hidden": 64},
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
    assert done["result"]["dataset"]["split"] == "trainval"
    assert done["result"]["folding"] == posted["folding"]


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
