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

from juniper_recurrence_model.readouts import _GCV_GRID, READOUT_REGISTRY, LinearReadout, LinearReadoutSpec, RFFReadout, RFFReadoutSpec, _gcv_select, _standardize_fit, build_readout_from_state


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


def test_rff_rejects_nonpositive_n_features():
    """RFFReadout(n_features_out < 1) must raise, not divide-by-zero at fit time (audit MODEL-04)."""
    with pytest.raises(ValueError, match="n_features_out"):
        RFFReadout(n_features_out=0)


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


# --- W5.9: the GCV grid's edges are reported, not moved --------------------------------------


def _noise_block(seed: int = 1, n: int = 200, p: int = 10):
    """A target with nothing to fit: the GCV optimum is the intercept-only (null) model."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), np.empty((n, 0)), rng.normal(size=(n, 1))


def test_gcv_select_reports_the_null_model_and_the_ceiling_on_pure_noise(caplog):
    M, extra, y = _noise_block()
    with caplog.at_level("WARNING", logger="juniper_recurrence_model.readouts"):
        selection = _gcv_select(np.concatenate([M, extra], axis=1), y)
    assert selection.ridge == float(_GCV_GRID[-1])  # the ceiling: no grid point beats the null model
    assert selection.at_edge is True
    assert selection.null_model is True
    assert selection.gcv_null < selection.gcv_best  # the one off-grid point that is free to evaluate
    n = M.shape[0]
    yc = y - y.mean(axis=0)
    assert selection.gcv_null == pytest.approx(n * float(np.sum(yc**2)) / (n - 1.0) ** 2)
    assert "intercept-only model" in caplog.text and "abstains at the ceiling" in caplog.text


def test_gcv_select_interior_optimum_sets_no_flags_and_logs_nothing(caplog):
    M, extra, rng = _block(n=200, p=16, k=1, seed=3)
    design = np.concatenate([M, extra, np.ones((200, 1))], axis=1)
    y = design @ rng.normal(size=(design.shape[1], 1)) + 0.1 * rng.normal(size=(200, 1))
    with caplog.at_level("WARNING", logger="juniper_recurrence_model.readouts"):
        selection = _gcv_select(np.concatenate([M, extra], axis=1), y)
    assert float(_GCV_GRID[0]) < selection.ridge < float(_GCV_GRID[-1])
    assert selection.at_edge is False
    assert selection.null_model is False
    assert selection.gcv_best < selection.gcv_null
    assert caplog.text == ""


def test_gcv_select_at_the_floor_is_an_edge_but_not_the_null_model(caplog):
    rng = np.random.default_rng(0)
    M = rng.normal(size=(200, 8))
    y = M @ rng.normal(size=(8, 1))  # exact linear target, no noise: the smallest penalty wins
    with caplog.at_level("WARNING", logger="juniper_recurrence_model.readouts"):
        selection = _gcv_select(M, y)
    assert selection.ridge == float(_GCV_GRID[0])
    assert selection.at_edge is True
    assert selection.null_model is False
    assert "edge of the grid" in caplog.text


def test_linear_gcv_fit_exposes_the_w5_9_flags_as_booleans():
    M, extra, y = _noise_block()
    ro = LinearReadout(ridge="gcv")
    assert ro.gcv_at_edge_ is None and ro.null_model_ is None  # not determined before a fit
    ro.fit(M, extra, y)
    assert ro.gcv_at_edge_ is True and ro.null_model_ is True
    assert ro.ridge == float(_GCV_GRID[-1])
    M2, extra2, rng = _block(n=200, p=16, k=1, seed=3)
    y2 = np.concatenate([M2, extra2, np.ones((200, 1))], axis=1) @ rng.normal(size=(18, 1)) + 0.1 * rng.normal(size=(200, 1))
    ro.ridge = "gcv"
    ro.fit(M2, extra2, y2)  # a re-fit resets and re-derives the flags
    assert ro.gcv_at_edge_ is False and ro.null_model_ is False


def test_fixed_ridge_fit_leaves_the_w5_9_flags_undetermined():
    M, extra, rng = _block(n=60, p=8, seed=2)
    y = rng.normal(size=(60, 1))
    for ridge in (0.0, 0.3):
        ro = LinearReadout(ridge=ridge)
        ro.fit(M, extra, y)
        assert ro.gcv_at_edge_ is None and ro.null_model_ is None  # not applicable: None, never False


def test_w5_9_flags_are_not_persisted_and_read_none_after_reload():
    M, extra, y = _noise_block()
    ro = LinearReadout(ridge="gcv")
    ro.fit(M, extra, y)
    arrays, descriptor = ro.save_state()
    assert "gcv_at_edge_" not in descriptor and "null_model_" not in descriptor
    restored = LinearReadout.from_state(arrays, descriptor)
    assert restored.gcv_at_edge_ is None and restored.null_model_ is None


def test_rff_gcv_fit_exposes_the_w5_9_flags_and_a_fixed_ridge_does_not():
    M, extra, y = _noise_block(p=6)
    gcv = RFFReadout(n_features_out=32, ridge="gcv")
    gcv.fit(M, extra, y, random_seed=0)
    assert isinstance(gcv.gcv_at_edge_, bool) and isinstance(gcv.null_model_, bool)
    assert gcv.gcv_at_edge_ == (gcv.ridge in (float(_GCV_GRID[0]), float(_GCV_GRID[-1])))
    fixed = RFFReadout(n_features_out=32, ridge=1.0)
    fixed.fit(M, extra, y, random_seed=0)
    assert fixed.gcv_at_edge_ is None and fixed.null_model_ is None


# --- W5.8 (B): the linear rung standardises the memory block on the train fold when ridge != 0 ---


def test_linear_ridge_zero_path_is_untouched_by_standardisation():
    M, extra, rng = _block(n=50, p=10, k=1, seed=5)
    y = rng.normal(size=(50, 1))
    ro = LinearReadout(ridge=0.0)
    ro.fit(M, extra, y)
    expected, *_ = np.linalg.lstsq(np.concatenate([M, extra, np.ones((50, 1))], axis=1), y, rcond=None)
    assert np.array_equal(ro.coef, expected)  # byte-identical to the raw min-norm solve (the conformance baseline)
    assert ro.standardized is False
    arrays, descriptor = ro.save_state()
    assert set(arrays) == {"coef"} and descriptor["standardized"] is False


@pytest.mark.parametrize("ridge", [0.3, "gcv"])
def test_linear_nonzero_ridge_standardises_with_train_fold_statistics(ridge):
    M, extra, rng = _block(n=80, p=6, k=1, seed=8)
    M = M * np.array([1.0, 10.0, 100.0, 1e3, 1e4, 1e5])  # columns of wildly different scale
    y = rng.normal(size=(80, 1))
    ro = LinearReadout(ridge=ridge)
    ro.fit(M, extra, y)
    mean, std = _standardize_fit(M)
    assert ro.standardized is True
    assert np.array_equal(ro._mean, mean) and np.array_equal(ro._std, std)
    # A later block is standardised with the TRAIN statistics, never its own.
    M_eval, extra_eval, _ = _block(n=30, p=6, k=1, seed=9)
    M_eval = M_eval * 7.0
    design_eval = np.concatenate([(M_eval - mean) / std, extra_eval, np.ones((30, 1))], axis=1)
    assert np.array_equal(ro.predict(M_eval, extra_eval), design_eval @ ro.coef)


def test_linear_fixed_ridge_is_column_scale_invariant_once_standardised():
    """The reason for W5.8 (B): a single penalty on raw columns depends on their units; standardised, it does not."""
    M, extra, rng = _block(n=120, p=5, k=0, seed=11)
    y = M @ rng.normal(size=(5, 1)) + rng.normal(size=(120, 1))
    scale = np.array([1.0, 1e-3, 1e3, 5.0, 0.02])
    a = LinearReadout(ridge=2.0)
    a.fit(M, extra, y)
    b = LinearReadout(ridge=2.0)
    b.fit(M * scale, extra, y)
    assert np.allclose(a.predict(M, extra), b.predict(M * scale, extra), rtol=1e-9, atol=1e-9)


def test_linear_standardised_fit_persists_its_statistics_and_reloads_bit_exact():
    M, extra, rng = _block(n=70, p=9, k=1, seed=12)
    y = rng.normal(size=(70, 1))
    ro = LinearReadout(ridge=0.5)
    ro.fit(M, extra, y)
    arrays, descriptor = ro.save_state()
    assert set(arrays) == {"coef", "mean", "std"} and descriptor["standardized"] is True
    restored = LinearReadout.from_state(arrays, descriptor)
    assert restored.standardized is True
    assert np.array_equal(ro.predict(M, extra), restored.predict(M, extra))


def test_linear_pre_0_5_0_state_without_statistics_predicts_on_the_raw_block():
    M, extra, rng = _block(n=40, p=4, k=0, seed=13)
    coef = rng.normal(size=(5, 1))
    restored = LinearReadout.from_state({"coef": coef}, {"kind": "linear", "ridge": 0.7})  # no mean / std, no flag
    assert restored.standardized is False
    assert np.array_equal(restored.predict(M, extra), np.concatenate([M, np.ones((40, 1))], axis=1) @ coef)


def test_linear_standardisation_guards_a_zero_variance_column():
    M, extra, rng = _block(n=40, p=4, k=0, seed=14)
    M[:, 2] = 3.0
    y = rng.normal(size=(40, 1))
    ro = LinearReadout(ridge="gcv")
    ro.fit(M, extra, y)
    assert np.all(np.isfinite(ro.predict(M, extra)))
    assert ro._std[2] == 1.0


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
