"""Tests for the fixed-order LMU regressor (WS-4 model layer, PR-1).

Covers the model mechanics: the batched rollout's parity with the single-sequence oracle,
recovery of a target linear in the memory state, determinism, the overfit-tiny guarantee,
the bare ``predict(X)`` (uniform-dt) path the conformance kit uses, regression-only metrics
(RK-6), a renderable topology, and a lossless serializer round-trip. The full juniper-model-core
conformance suite and the R-Δt-3 / §9.1a Δt guardrails land in PR-2.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
from juniper_model_core.validation import legal_event_order, validate_metrics, validate_topology

from juniper_recurrence_model import LMURegressor, LMUSerializer, VariableStepLMUMemory


def _toy_3d(n: int = 32, n_steps: int = 8, n_features: int = 3, seed: int = 0):
    """A small (n, T, F) batch with strictly-positive irregular integer gaps ``dt``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_steps, n_features))
    dt = np.zeros((n, n_steps))
    dt[:, 1:] = rng.integers(1, 4, size=(n, n_steps - 1)).astype(float)
    return X, dt, rng


def test_batched_rollout_matches_single_seq_oracle():
    """The batched eigenbasis rollout matches per-(sample, feature) VariableStepLMUMemory.rollout()."""
    mem = VariableStepLMUMemory(d=12, theta=5.0)
    X, dt, _ = _toy_3d(n=6, n_steps=10, n_features=4, seed=11)
    batch = mem.rollout_batch(X, dt)  # (6, 10, 4, 12)
    assert batch.shape == (6, 10, 4, 12)
    for i in range(X.shape[0]):
        for f in range(X.shape[2]):
            oracle = mem.rollout(X[i, :, f], dt[i])  # (10, 12)
            assert np.allclose(batch[i, :, f, :], oracle, rtol=1e-6, atol=1e-9)


def test_rollout_batch_treats_zero_gap_as_held_step():
    """dt == 0 is a no-op step (padding-safe): the memory is unchanged across it."""
    mem = VariableStepLMUMemory(d=8, theta=4.0)
    X = np.ones((1, 4, 1))
    dt = np.array([[0.0, 1.0, 0.0, 1.0]])  # step 2 has a zero gap
    out = mem.rollout_batch(X, dt)
    assert np.allclose(out[0, 2, 0, :], out[0, 1, 0, :])  # held across the zero gap


def test_fit_predict_recovers_linear_over_memory():
    """A target linear in the final-step memory state is recovered with high R²."""
    mem = VariableStepLMUMemory(d=16, theta=10.0)
    X, dt, rng = _toy_3d(n=200, n_steps=12, n_features=3, seed=1)
    memory_state = mem.rollout_batch(X, dt)[:, -1].reshape(X.shape[0], -1)  # (n, F*d)
    weight = rng.normal(size=(memory_state.shape[1], 1))
    y = memory_state @ weight + 0.01 * rng.normal(size=(X.shape[0], 1))

    model = LMURegressor(d=16, theta=10.0)
    result = model.fit(X, y, dt=dt)
    assert result.n_epochs >= 1
    assert model.metrics()["r2"] > 0.99
    preds = model.predict(X, dt=dt)
    assert preds.shape == (X.shape[0], 1)
    assert tuple(model.input_shape) == (12, 3)
    assert tuple(model.output_shape) == (1,)


def test_fit_emits_legal_event_order():
    X, dt, rng = _toy_3d(seed=2)
    y = rng.normal(size=(X.shape[0], 1))
    events: list = []
    LMURegressor(d=10, theta=6.0).fit(X, y, dt=dt, on_event=events.append)
    assert legal_event_order(events)
    assert [e.type for e in events] == ["training_start", "epoch_end", "training_end"]


def test_determinism():
    X, dt, rng = _toy_3d(n=40, n_steps=8, n_features=3, seed=3)
    y = rng.normal(size=(40, 1))
    first = LMURegressor(d=10, theta=6.0)
    first.fit(X, y, dt=dt)
    second = LMURegressor(d=10, theta=6.0)
    second.fit(X, y, dt=dt)
    assert np.array_equal(first.predict(X, dt=dt), second.predict(X, dt=dt))


def test_overfit_tiny():
    """With F*d >> n the closed-form readout memorises a tiny set (near-zero MSE)."""
    X, dt, rng = _toy_3d(n=5, n_steps=6, n_features=4, seed=4)  # F*d = 4*16 = 64 >> 5
    y = rng.normal(size=(5, 1))
    model = LMURegressor(d=16, theta=8.0)
    model.fit(X, y, dt=dt)
    assert model.metrics()["mse"] < 1e-6


def test_predict_without_dt_uses_uniform_grid():
    """Bare predict(X) (no dt) works — the conformance kit calls predict this way."""
    X, dt, rng = _toy_3d(n=20, n_steps=8, n_features=3, seed=5)
    y = rng.normal(size=(20, 1))
    model = LMURegressor(d=12, theta=6.0)
    model.fit(X, y, dt=dt)
    preds = model.predict(X)  # no dt supplied
    assert preds.shape == (20, 1)
    assert np.all(np.isfinite(preds))


def test_target_dt_is_used_as_readout_feature_when_supplied():
    """Supplying target_dt at fit widens the readout; predict must then be consistent."""
    X, dt, rng = _toy_3d(n=30, n_steps=8, n_features=3, seed=6)
    target_dt = rng.integers(1, 5, size=30).astype(float)
    y = rng.normal(size=(30, 1))
    model = LMURegressor(d=10, theta=6.0)
    model.fit(X, y, dt=dt, target_dt=target_dt)
    assert model._uses_target_dt is True
    # readout coef has one extra row (target_dt) beyond F*d + bias
    assert model._coef.shape[0] == 3 * 10 + 1 + 1
    preds = model.predict(X, dt=dt, target_dt=target_dt)
    assert preds.shape == (30, 1)


def test_metrics_are_regression_only():
    X, dt, rng = _toy_3d(seed=7)
    y = rng.normal(size=(X.shape[0], 1))
    model = LMURegressor(d=8, theta=5.0)
    model.fit(X, y, dt=dt)
    validate_metrics("regression", model.metrics())  # raises if a classification key leaked
    assert "accuracy" not in model.metrics()
    assert model.task_type == "regression"


def test_topology_is_valid_and_renderable():
    X, dt, rng = _toy_3d(seed=8)
    y = rng.normal(size=(X.shape[0], 1))
    model = LMURegressor(d=8, theta=5.0)
    model.fit(X, y, dt=dt)
    topology = model.describe_topology()
    validate_topology(topology)  # raises if malformed
    assert topology["model_type"] == "lmu"
    assert topology["meta"]["task_type"] == "regression"
    assert any(node["kind"] == "memory" for node in topology["nodes"])


def test_serializer_roundtrip_lossless():
    X, dt, rng = _toy_3d(n=30, n_steps=8, n_features=3, seed=9)
    y = rng.normal(size=(30, 1))
    model = LMURegressor(d=12, theta=7.0)
    model.fit(X, y, dt=dt)
    before = model.predict(X, dt=dt)
    serializer = LMUSerializer()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "model")
        serializer.save(model, path)
        restored = serializer.load(path)
    after = restored.predict(X, dt=dt)
    assert np.array_equal(before, after)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LMURegressor(d=8, theta=5.0).predict(np.zeros((2, 4, 3)))


def test_fit_accepts_1d_targets():
    """A 1-D y is treated as a single regression target: (n,) -> (n, 1)."""
    X, dt, rng = _toy_3d(n=24, seed=20)
    y = rng.normal(size=24)
    model = LMURegressor(d=8, theta=5.0)
    model.fit(X, y, dt=dt)
    assert tuple(model.output_shape) == (1,)
    assert model.predict(X, dt=dt).shape == (24, 1)


def test_fit_rejects_dense_many_to_many():
    """Dense (n, T, output) targets are a deferred increment -> NotImplementedError."""
    X, dt, rng = _toy_3d(n=10, n_steps=6, seed=21)
    y = rng.normal(size=(10, 6, 1))
    with pytest.raises(NotImplementedError):
        LMURegressor(d=8, theta=5.0).fit(X, y, dt=dt)


def test_ridge_readout_path():
    """ridge > 0 uses the regularised normal-equation solve and still fits/predicts."""
    X, dt, rng = _toy_3d(n=60, n_steps=8, n_features=3, seed=22)
    memory_state = VariableStepLMUMemory(10, 6.0).rollout_batch(X, dt)[:, -1].reshape(60, -1)
    y = memory_state @ rng.normal(size=(memory_state.shape[1], 1))
    model = LMURegressor(d=10, theta=6.0, ridge=1e-3)
    model.fit(X, y, dt=dt)
    assert model.metrics()["r2"] > 0.9
    assert model.predict(X, dt=dt).shape == (60, 1)


def test_readout_mask_and_seq_lengths_select_last_valid_step():
    """seq_lengths and an equivalent readout_mask pick the same last-valid step."""
    X, dt, rng = _toy_3d(n=12, n_steps=8, n_features=2, seed=23)
    y = rng.normal(size=(12, 1))
    seq_lengths = np.full(12, 5)
    model = LMURegressor(d=8, theta=5.0)
    model.fit(X, y, dt=dt, seq_lengths=seq_lengths)
    via_lengths = model.predict(X, dt=dt, seq_lengths=seq_lengths)
    mask = np.zeros((12, 8), dtype=bool)
    mask[:, :5] = True
    via_mask = model.predict(X, dt=dt, readout_mask=mask)
    assert np.allclose(via_lengths, via_mask)


def test_predict_rejects_wrong_feature_count():
    X, dt, rng = _toy_3d(n=10, n_features=3, seed=24)
    y = rng.normal(size=(10, 1))
    model = LMURegressor(d=8, theta=5.0)
    model.fit(X, y, dt=dt)
    with pytest.raises(ValueError):
        model.predict(np.zeros((4, 8, 5)))  # 5 features != the 3 seen at fit


def test_rollout_batch_accepts_2d_single_channel():
    mem = VariableStepLMUMemory(d=6, theta=4.0)
    u = np.ones((3, 5))
    dt = np.tile(np.array([0.0, 1.0, 1.0, 1.0, 1.0]), (3, 1))
    assert mem.rollout_batch(u, dt).shape == (3, 5, 1, 6)


def test_rollout_batch_input_validation():
    mem = VariableStepLMUMemory(d=6, theta=4.0)
    with pytest.raises(ValueError):
        mem.rollout_batch(np.zeros((2, 3, 4, 1)), np.zeros((2, 3)))  # 4-D u
    with pytest.raises(ValueError):
        mem.rollout_batch(np.zeros((2, 5, 1)), np.zeros((2, 4)))  # dt shape mismatch
    with pytest.raises(ValueError):
        mem.rollout_batch(np.zeros((2, 5, 1)), -np.ones((2, 5)))  # negative dt


def test_serializer_rejects_non_lmu_and_unfitted():
    from juniper_model_core.conformance import ReferenceLinearModel

    serializer = LMUSerializer()
    with tempfile.TemporaryDirectory() as tmp:
        target = str(pathlib.Path(tmp) / "m")
        with pytest.raises(TypeError):
            serializer.save(ReferenceLinearModel(), target)  # not an LMURegressor
        with pytest.raises(RuntimeError):
            serializer.save(LMURegressor(d=8, theta=5.0), target)  # unfitted
