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

from juniper_recurrence_model.readouts import READOUT_REGISTRY, LinearReadout, LinearReadoutSpec, RFFReadout, RFFReadoutSpec, build_readout_from_state


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


# ----- Rung 2a: RFF nonlinear readout (P2) -------------------------------------------


def test_rff_spec_frozen_and_makes_readout():
    spec = RFFReadoutSpec(n_features_out=64, gamma="median", ridge="gcv")
    ro = spec.make()
    assert isinstance(ro, RFFReadout) and ro.kind == "rff" and ro.n_features_out == 64
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.n_features_out = 1  # type: ignore[misc]


def test_rff_unfitted_state():
    ro = RFFReadout(n_features_out=20)
    assert ro.is_fitted is False and ro.coef is None  # nonlinear -> coef stays None
    with pytest.raises(RuntimeError):
        ro.predict(*_block(p=4)[:2])
    with pytest.raises(RuntimeError):
        ro.save_state()


def test_rff_fit_predict_finite_and_coef_none():
    M, extra, rng = _block(n=120, p=10, k=1, seed=10)
    y = rng.normal(size=(120, 1))
    ro = RFFReadout(n_features_out=64, ridge="gcv")
    ro.fit(M, extra, y, random_seed=0)
    pred = ro.predict(M, extra)
    assert pred.shape == (120, 1)
    assert np.all(np.isfinite(pred))  # NaN would fail the array_equal serialization contract
    assert ro.coef is None and ro.is_fitted is True


def test_rff_captures_nonlinear_product_a_linear_readout_cannot():
    """Capacity: a bilinear target y = M0·M1 is unreachable by a linear readout; RFF approximates it."""
    rng = np.random.default_rng(11)
    n, p = 400, 8
    M = rng.normal(size=(n, p))
    extra = np.empty((n, 0))
    y = (M[:, 0] * M[:, 1])[:, None] + 0.05 * rng.normal(size=(n, 1))

    def _r2(ro):
        pred = ro.predict(M, extra)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    lin = LinearReadout(ridge="gcv")
    lin.fit(M, extra, y)
    rff = RFFReadout(n_features_out=300, ridge="gcv")
    rff.fit(M, extra, y, random_seed=0)
    r2_lin, r2_rff = _r2(lin), _r2(rff)
    assert r2_lin < 0.25, f"linear readout should fail the bilinear target; got r2={r2_lin}"
    assert r2_rff > 0.6, f"RFF readout should capture the bilinear target; got r2={r2_rff}"
    assert r2_rff > r2_lin + 0.35


def test_rff_handles_zero_variance_column():
    rng = np.random.default_rng(12)
    M = rng.normal(size=(60, 5))
    M[:, 2] = 3.0  # a constant memory column -> std 0 -> would be 0/0 NaN without the guard
    extra = np.empty((60, 0))
    y = rng.normal(size=(60, 1))
    ro = RFFReadout(n_features_out=32, ridge="gcv")
    ro.fit(M, extra, y, random_seed=0)
    assert np.all(np.isfinite(ro.predict(M, extra)))


def test_rff_deterministic_from_seed():
    M, extra, rng = _block(n=80, p=10, k=1, seed=20)
    y = rng.normal(size=(80, 1))
    a = RFFReadout(n_features_out=48)
    a.fit(M, extra, y, random_seed=7)
    b = RFFReadout(n_features_out=48)
    b.fit(M, extra, y, random_seed=7)
    c = RFFReadout(n_features_out=48)
    c.fit(M, extra, y, random_seed=8)
    assert np.array_equal(a.predict(M, extra), b.predict(M, extra))  # same seed -> identical (cross-fold safe)
    assert not np.array_equal(a.predict(M, extra), c.predict(M, extra))  # different seed -> different W,b


def test_rff_save_state_roundtrip_lossless():
    M, extra, rng = _block(n=70, p=10, k=1, seed=21)
    y = rng.normal(size=(70, 1))
    ro = RFFReadout(n_features_out=40, ridge="gcv")
    ro.fit(M, extra, y, random_seed=3)
    arrays, descriptor = ro.save_state()
    assert descriptor["kind"] == "rff" and descriptor["n_features_out"] == 40 and "gamma" in descriptor
    restored = RFFReadout.from_state(arrays, descriptor)
    assert np.array_equal(ro.predict(M, extra), restored.predict(M, extra))  # bit-exact
    rebuilt = build_readout_from_state(arrays, descriptor)  # registry path
    assert isinstance(rebuilt, RFFReadout)
    assert np.array_equal(ro.predict(M, extra), rebuilt.predict(M, extra))


def test_rff_d_capped_to_fold_size():
    M, extra, rng = _block(n=30, p=8, seed=22)
    y = rng.normal(size=(30, 1))
    ro = RFFReadout(n_features_out=256)  # 256 > n -> capped to n
    ro.fit(M, extra, y, random_seed=0)
    _, descriptor = ro.save_state()
    assert descriptor["n_features_out"] == 30


def test_rff_invalid_ridge_string_raises():
    M, extra, rng = _block(seed=24)
    y = rng.normal(size=(M.shape[0], 1))
    with pytest.raises(ValueError):
        RFFReadout(ridge="bogus").fit(M, extra, y, random_seed=0)  # type: ignore[arg-type]


def test_rff_fixed_gamma_is_used_and_persisted():
    M, extra, rng = _block(n=50, p=6, seed=25)
    y = rng.normal(size=(50, 1))
    ro = RFFReadout(n_features_out=32, gamma=0.5, ridge=1.0)
    ro.fit(M, extra, y, random_seed=0)
    _, descriptor = ro.save_state()
    assert descriptor["gamma"] == 0.5  # explicit gamma bypasses the median heuristic


def test_rff_ridge_zero_uses_lstsq():
    M, extra, rng = _block(n=80, p=8, seed=26)
    y = rng.normal(size=(80, 1))
    ro = RFFReadout(n_features_out=32, gamma=0.5, ridge=0.0)  # unregularised RFF -> min-norm lstsq
    ro.fit(M, extra, y, random_seed=0)
    assert np.all(np.isfinite(ro.predict(M, extra)))
