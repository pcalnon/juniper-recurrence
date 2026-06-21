"""Unit tests for the DP-3 readout spectrum (``juniper_recurrence_model.readouts``).

These exercise the readouts in isolation from the LMU memory: a readout maps a memory block
``M`` (n, p), a linear side-channel ``extra`` (n, k), and a target ``y`` to predictions, and
round-trips its fitted state losslessly via ``save_state`` / ``from_state`` (the bit-exact
serialization contract). Model-level integration (GCV write-back, spec/ridge equivalence, the
pre-DP-3 load fallback) lives in ``test_lmu_model.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from juniper_recurrence_model.readouts import READOUT_REGISTRY, LinearReadout, LinearReadoutSpec, build_readout_from_state


def _block(n: int = 80, p: int = 12, k: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, p))
    extra = rng.normal(size=(n, k)) if k else np.empty((n, 0))
    return M, extra, rng


def test_linear_spec_makes_unfitted_readout():
    ro = LinearReadoutSpec(ridge=0.5).make()
    assert isinstance(ro, LinearReadout)
    assert ro.kind == "linear"
    assert ro.ridge == 0.5
    assert ro.is_fitted is False
    assert ro.coef is None


def test_linear_spec_is_frozen():
    spec = LinearReadoutSpec(ridge="gcv")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.ridge = 1.0  # type: ignore[misc]


def test_linear_readout_lstsq_recovers_linear_target():
    M, extra, rng = _block(n=120, p=10, k=1, seed=1)
    design = np.concatenate([M, extra, np.ones((M.shape[0], 1))], axis=1)
    w = rng.normal(size=(design.shape[1], 2))
    y = design @ w
    ro = LinearReadout(ridge=0.0)
    ro.fit(M, extra, y)
    assert ro.is_fitted
    assert ro.coef.shape == (10 + 1 + 1, 2)  # p + extra + bias
    assert np.allclose(ro.predict(M, extra), y, atol=1e-6)


def test_linear_readout_predict_before_fit_raises():
    M, extra, _ = _block()
    with pytest.raises(RuntimeError):
        LinearReadout().predict(M, extra)


def test_linear_readout_save_state_unfitted_raises():
    with pytest.raises(RuntimeError):
        LinearReadout().save_state()


def test_ridge_shrinks_non_bias_coefficients():
    M, extra, rng = _block(n=60, p=8, seed=2)
    y = rng.normal(size=(60, 1))
    small = LinearReadout(ridge=1e-6)
    small.fit(M, extra, y)
    big = LinearReadout(ridge=1e3)
    big.fit(M, extra, y)
    assert np.linalg.norm(big.coef[:-1]) < np.linalg.norm(small.coef[:-1])


def test_gcv_selects_positive_lambda_and_writes_back():
    M, extra, rng = _block(n=200, p=16, k=1, seed=3)
    design = np.concatenate([M, extra, np.ones((200, 1))], axis=1)
    y = design @ rng.normal(size=(design.shape[1], 1)) + 0.1 * rng.normal(size=(200, 1))
    ro = LinearReadout(ridge="gcv")
    ro.fit(M, extra, y)
    assert isinstance(ro.ridge, float) and ro.ridge > 0.0  # "gcv" replaced by the selected λ
    assert ro.coef.shape == (16 + 1 + 1, 1)
    assert np.all(np.isfinite(ro.predict(M, extra)))


def test_gcv_generalizes_better_than_unregularized_on_noisy_near_overparam_data():
    """On a noisy near-overparameterised fit (p≈n), GCV ridge beats min-norm lstsq out-of-sample."""
    rng = np.random.default_rng(4)
    n, p = 50, 45
    M = rng.normal(size=(2 * n, p))
    w = rng.normal(size=(p + 1, 1))
    design_full = np.concatenate([M, np.ones((2 * n, 1))], axis=1)
    y = design_full @ w + 2.0 * rng.normal(size=(2 * n, 1))
    m_tr, m_te = M[:n], M[n:]
    y_tr, y_te = y[:n], y[n:]
    e_tr, e_te = np.empty((n, 0)), np.empty((n, 0))
    plain = LinearReadout(ridge=0.0)
    plain.fit(m_tr, e_tr, y_tr)
    gcv = LinearReadout(ridge="gcv")
    gcv.fit(m_tr, e_tr, y_tr)
    err_plain = float(np.mean((plain.predict(m_te, e_te) - y_te) ** 2))
    err_gcv = float(np.mean((gcv.predict(m_te, e_te) - y_te) ** 2))
    assert err_gcv < err_plain


def test_gcv_invalid_ridge_string_raises():
    M, extra, rng = _block()
    y = rng.normal(size=(M.shape[0], 1))
    ro = LinearReadout(ridge="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ro.fit(M, extra, y)


def test_gcv_roundtrip_is_lossless_after_writeback():
    M, extra, rng = _block(n=120, p=12, k=1, seed=7)
    y = rng.normal(size=(120, 1))
    ro = LinearReadout(ridge="gcv")
    ro.fit(M, extra, y)
    arrays, descriptor = ro.save_state()
    assert descriptor["ridge"] == ro.ridge  # the selected λ is persisted, not the symbolic "gcv"
    restored = LinearReadout.from_state(arrays, descriptor)
    assert np.array_equal(ro.predict(M, extra), restored.predict(M, extra))


def test_save_state_roundtrip_via_from_state():
    M, extra, rng = _block(n=50, p=10, k=1, seed=5)
    y = rng.normal(size=(50, 1))
    ro = LinearReadout(ridge=0.3)
    ro.fit(M, extra, y)
    arrays, descriptor = ro.save_state()
    assert descriptor["kind"] == "linear" and descriptor["ridge"] == 0.3
    restored = LinearReadout.from_state(arrays, descriptor)
    assert np.array_equal(ro.predict(M, extra), restored.predict(M, extra))


def test_build_readout_from_state_registry_roundtrip():
    M, extra, rng = _block(n=40, p=6, seed=6)
    y = rng.normal(size=(40, 1))
    ro = LinearReadout()
    ro.fit(M, extra, y)
    arrays, descriptor = ro.save_state()
    rebuilt = build_readout_from_state(arrays, descriptor)
    assert isinstance(rebuilt, LinearReadout)
    assert np.array_equal(ro.predict(M, extra), rebuilt.predict(M, extra))
    assert "linear" in READOUT_REGISTRY


def test_build_readout_from_state_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_readout_from_state({"coef": np.zeros((2, 1))}, {"kind": "nope"})
