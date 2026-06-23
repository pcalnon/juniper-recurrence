"""DP-3 P2c — the HTTP readout enum.

Covers the :func:`juniper_recurrence._readout.build_lmu_regressor` translation helper (tagged enum
→ readout spec), the request-schema validation (``rff_*`` params only valid with ``readout="rff"``),
and the ``/v1/train`` route actually reaching the RFF nonlinear readout. The crossval route's RFF
path is exercised in ``test_crossval_routes.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from juniper_recurrence._readout import build_lmu_regressor
from juniper_recurrence.app import build_app
from juniper_recurrence.schemas import CrossValRequest, TrainRequest
from juniper_recurrence.settings import Settings


def _client(**settings_kwargs) -> TestClient:
    return TestClient(build_app(Settings(**settings_kwargs)))


def _tiny_fit_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, lookback, n_features = 24, 4, 1
    X = rng.standard_normal((n, lookback, n_features)).astype("float32")
    dt = np.zeros((n, lookback), dtype="float32")
    dt[:, 1:] = 1.0
    y = rng.standard_normal((n,)).astype("float32")
    target_dt = np.ones((n,), dtype="float32")
    return X, y, dt, target_dt


# --- build_lmu_regressor helper ---------------------------------------------------------


def test_build_lmu_regressor_rejects_unknown_readout() -> None:
    with pytest.raises(ValueError, match="unknown readout"):
        build_lmu_regressor(d=4, theta=None, readout="bogus", ridge=None, rff_features=None, rff_gamma=None, default_ridge=0.0)


def test_build_lmu_regressor_linear_has_coef_rff_does_not() -> None:
    # design §6: the linear readout exposes coefficients via model._coef; the nonlinear RFF readout
    # has no single coefficient vector, so model._coef is None.
    X, y, dt, target_dt = _tiny_fit_data()

    linear = build_lmu_regressor(d=4, theta=2.0, readout=None, ridge=None, rff_features=None, rff_gamma=None, default_ridge=0.0)
    linear.fit(X, y, dt=dt, target_dt=target_dt)
    assert linear._coef is not None

    rff = build_lmu_regressor(d=4, theta=2.0, readout="rff", ridge=None, rff_features=32, rff_gamma="median", default_ridge=0.0)
    rff.fit(X, y, dt=dt, target_dt=target_dt)
    assert rff._coef is None


def test_build_lmu_regressor_rff_explicit_params_construct() -> None:
    # Exercises the RFF branch with every param explicit (features / gamma / ridge all non-default).
    model = build_lmu_regressor(d=4, theta=2.0, readout="rff", ridge=0.1, rff_features=16, rff_gamma=0.5, default_ridge=0.0)
    assert type(model).__name__ == "LMURegressor"


def test_build_lmu_regressor_rejects_rff_params_without_rff_readout() -> None:
    # The shared helper rejects the RFF-only knobs on the linear readout (so the CLI behaves like the
    # HTTP edge) rather than silently dropping them.
    with pytest.raises(ValueError, match="rff_features / rff_gamma"):
        build_lmu_regressor(d=4, theta=None, readout=None, ridge=None, rff_features=64, rff_gamma=None, default_ridge=0.0)
    with pytest.raises(ValueError, match="rff_features / rff_gamma"):
        build_lmu_regressor(d=4, theta=None, readout="linear", ridge=None, rff_features=None, rff_gamma=0.5, default_ridge=0.0)


# --- request-schema validation ----------------------------------------------------------


def test_train_request_rejects_rff_params_without_rff_readout() -> None:
    with pytest.raises(ValidationError, match="rff_features / rff_gamma"):
        TrainRequest(dataset={"dataset_id": "x"}, rff_features=128)


def test_crossval_request_rejects_rff_params_without_rff_readout() -> None:
    with pytest.raises(ValidationError, match="rff_features / rff_gamma"):
        CrossValRequest(dataset={"dataset_id": "x"}, n_folds=3, rff_gamma=0.5)


def test_readout_request_accepts_rff_params() -> None:
    req = TrainRequest(dataset={"dataset_id": "x"}, readout="rff", rff_features=128, rff_gamma="median")
    assert req.readout == "rff" and req.rff_features == 128 and req.rff_gamma == "median"


# --- /v1/train route reaching the RFF readout -------------------------------------------


def test_train_route_readout_rff(fake_data) -> None:
    client = _client(api_keys=None)
    resp = client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "readout": "rff", "rff_features": 64})
    assert resp.status_code == 200, resp.text
    meta = client.get("/v1/model").json()["topology"]["meta"]
    assert meta["readout"]["kind"] == "rff"


def test_train_route_rff_params_without_readout_422(fake_data) -> None:
    resp = _client(api_keys=None).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "rff_features": 64})
    assert resp.status_code == 422
