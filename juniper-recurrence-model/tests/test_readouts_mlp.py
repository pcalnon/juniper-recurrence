"""Rung 2b — torch MLP readout (DP-3 P3). The whole module skips without the ``[torch]`` extra.

Covers the unit contract (fit/predict, within-machine determinism, the bit-exact save→load
round-trip, validation-driven early stopping) and a model-core conformance subclass driving the
MLP readout through the full ``TrainableModel`` contract via ``LMURegressor`` + ``LMUSerializer``.
Early stopping is unit-tested by passing validation arrays to the readout directly, and — since A2 —
end-to-end through ``LMURegressor.fit(X, y, X_val=…, y_val=…)``, asserting the readout's epoch and
stop diagnostics surface on ``TrainResult`` (``n_epochs`` / ``stopped_reason``).
"""

from __future__ import annotations

import numpy as np
import pytest
from juniper_model_core.conformance import TrainableModelConformance, tiny_regression_3d

from juniper_recurrence_model import LMURegressor, LMUSerializer, MLPReadoutSpec
from juniper_recurrence_model.readouts import build_readout_from_state

pytest.importorskip("torch")  # Rung 2b requires the [torch] extra; skip the whole module without it


def _toy(n: int = 64, p: int = 12, k: int = 0, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, p))
    extra = np.zeros((n, k))
    y = (np.sin(M[:, 0]) + 0.1 * M[:, 1] ** 2 + 0.05 * rng.standard_normal(n))[:, None]
    return M, extra, y


def _fit_predict(seed: int) -> np.ndarray:
    M, extra, y = _toy()
    ro = MLPReadoutSpec(hidden=16, max_epochs=20).make()
    ro.fit(M, extra, y, random_seed=seed)
    return ro.predict(M, extra)


def test_mlp_fit_predict_shape_and_finite() -> None:
    M, extra, y = _toy()
    ro = MLPReadoutSpec(hidden=16, max_epochs=20).make()
    assert not ro.is_fitted
    ro.fit(M, extra, y, random_seed=0)
    assert ro.is_fitted
    assert ro.coef is None  # nonlinear: no single linear coefficient vector
    pred = ro.predict(M, extra)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_mlp_deterministic_within_machine() -> None:
    # Same seed -> bit-identical predictions (CPU, float32, use_deterministic_algorithms, 1 thread).
    assert np.array_equal(_fit_predict(0), _fit_predict(0))


def test_mlp_serializer_roundtrip_bitexact() -> None:
    M, extra, y = _toy()
    ro = MLPReadoutSpec(hidden=16, max_epochs=20).make()
    ro.fit(M, extra, y, random_seed=0)
    before = ro.predict(M, extra)
    arrays, descriptor = ro.save_state()
    assert descriptor["kind"] == "mlp"
    restored = build_readout_from_state(arrays, descriptor)  # lazily registers "mlp"
    assert np.array_equal(before, restored.predict(M, extra))  # in-process bit-exact (no cross-machine claim)


def test_mlp_validation_early_stop() -> None:
    M, extra, y = _toy(n=96)
    rng = np.random.default_rng(7)
    M_val = rng.standard_normal(M.shape)
    y_val = rng.standard_normal((M.shape[0], 1))  # noise targets: no learnable signal -> val plateaus fast
    extra_val = np.zeros((M.shape[0], 0))
    early = MLPReadoutSpec(hidden=16, max_epochs=100, patience=3).make()
    early.fit(M, extra, y, M_val=M_val, extra_val=extra_val, y_val=y_val, random_seed=0)
    full = MLPReadoutSpec(hidden=16, max_epochs=100, patience=3).make()
    full.fit(M, extra, y, random_seed=0)  # no validation -> trains the full budget
    assert full.n_epochs_ == 100
    assert 1 <= early.n_epochs_ < full.n_epochs_  # validation early-stop curtails training
    assert early.stopped_reason_ == "early_stopping"
    assert full.stopped_reason_ == "max_epochs"


def test_mlp_via_lmuregressor_val_early_stop() -> None:
    """A2 plumbing end-to-end: LMURegressor feeds X_val/y_val into the MLP readout, and the
    readout's epoch / stop diagnostics surface through TrainResult. Noise validation targets make
    patience fire (n_epochs < max_epochs, stopped_reason "early_stopping"); the same fit with no
    validation runs the full budget ("max_epochs")."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 6, 3))
    y = (X[:, -1, 0] + 0.1 * X[:, -1, 1] ** 2)[:, None]
    X_val = rng.standard_normal((12, 6, 3))
    y_val = rng.standard_normal((12, 1))  # noise validation -> overfitting lifts val loss, patience fires
    # lr 1e-2 (not the 1e-3 default) lets the small net overfit within the budget, so early stopping
    # reliably fires well before max_epochs (verified stable across seeds: stops at epoch ~4-6).
    spec = MLPReadoutSpec(hidden=32, lr=1e-2, max_epochs=200, patience=3)

    res_val = LMURegressor(d=8, theta=6.0, readout=spec).fit(X, y, X_val=X_val, y_val=y_val)
    assert res_val.stopped_reason == "early_stopping"
    assert 1 <= res_val.n_epochs < 200

    res_full = LMURegressor(d=8, theta=6.0, readout=spec).fit(X, y)  # no validation -> full budget
    assert res_full.n_epochs == 200
    assert res_full.stopped_reason == "max_epochs"


def test_mlp_via_lmuregressor_end_to_end(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8, 3))
    y = rng.standard_normal(40)
    model = LMURegressor(d=4, readout=MLPReadoutSpec(hidden=8, max_epochs=15))
    model.fit(X, y)
    before = model.predict(X)
    serializer = LMUSerializer()
    path = tmp_path / "mlp_model.npz"
    serializer.save(model, path)
    restored = serializer.load(path)
    assert np.array_equal(before, restored.predict(X))  # full model round-trip via the generic serializer


class TestMLPLMUConformance(TrainableModelConformance):
    """Drive the LMU regressor with the **torch MLP readout** through the full ``TrainableModel`` contract.

    The gate that matters is the bit-exact lossless serialization round-trip (state persisted as named
    npz arrays, reloaded deterministically) plus the finite-prediction property.
    """

    def make_model(self):
        return LMURegressor(d=16, theta=30.0, readout=MLPReadoutSpec(hidden=16, max_epochs=20))

    def make_dataset(self):
        return tiny_regression_3d()

    def make_serializer(self):
        return LMUSerializer()
