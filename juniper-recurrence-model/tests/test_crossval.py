"""The real second-implementer proof for the juniper-model-core cross-validation layer.

`juniper_model_core.crossval` (model-core 0.2.0) is generic by construction; the conformance
kit dogfoods it against the in-tree `ReferenceLinearModel` and a stub 3-D model. This module
closes the loop with a *real* model: it drives :class:`~juniper_recurrence_model.LMURegressor`
— a fixed-order, Δt-native LMU regressor over 3-D ``(n, T, F)`` windows — through
``cross_validate`` with ``aux={dt, target_dt, seq_lengths}``, proving the layer slices the
auxiliary sequence arrays per fold and engages the Δt path end-to-end on a genuine implementer.

The synthetic target is generated from the LMU memory state under the *true* per-step gaps, so a
correct fold executor that forwards ``dt`` recovers it (high held-out r2) — and a guardrail test
confirms that shuffling ``dt`` (breaking the true gaps) measurably degrades the score, i.e. the
timing channel is really being used, not ignored.
"""

from __future__ import annotations

import numpy as np
from juniper_model_core.crossval import cross_validate, walk_forward_folds

from juniper_recurrence_model import LMURegressor
from juniper_recurrence_model.units import VariableStepLMUMemory

_D = 8
_THETA = 20.0


def _memory_linear_3d(n: int = 160, timesteps: int = 6, n_features: int = 2, seed: int = 0):
    """A 3-D sequence problem whose target is linear in the LMU memory under the true gaps.

    Returns ``(X, y, dt, target_dt, seq_lengths)``. Because ``y`` is built from the order-``_D``
    memory at ``theta=_THETA`` over the true ``dt``, an :class:`LMURegressor` constructed with the
    same ``(d, theta)`` recovers it with a closed-form readout — so held-out r2 is high when the
    executor forwards the correct ``dt``.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, timesteps, n_features))
    dt = np.zeros((n, timesteps))
    dt[:, 1:] = rng.integers(1, 5, size=(n, timesteps - 1)).astype(float)  # positive irregular gaps; dt[:,0]==0
    memory_state = VariableStepLMUMemory(_D, _THETA).rollout_batch(X, dt)[:, -1].reshape(n, n_features * _D)
    weight = rng.normal(size=(n_features * _D, 1))
    y = memory_state @ weight + 0.01 * rng.normal(size=(n, 1))
    target_dt = rng.uniform(1.0, 5.0, size=n)
    seq_lengths = np.full(n, timesteps)
    return X, y, dt, target_dt, seq_lengths


def _factory(_fold: int) -> LMURegressor:
    return LMURegressor(d=_D, theta=_THETA)


def test_lmu_crossval_real_second_implementer():
    # cross_validate drives a real Δt-native LMU over 3-D windows with all three aux arrays.
    X, y, dt, target_dt, seq_lengths = _memory_linear_3d()
    folds = walk_forward_folds(X.shape[0], n_folds=4)
    result = cross_validate(_factory, X, y, folds, aux={"dt": dt, "target_dt": target_dt, "seq_lengths": seq_lengths})

    assert result.task_type == "regression"
    assert len(result.folds) == 4
    assert [fold_result.fold for fold_result in result.folds] == [0, 1, 2, 3]
    assert set(result.eval_aggregate) == {"mse", "rmse", "mae", "r2", "loss"}
    assert "accuracy" not in result.eval_aggregate  # RK-6: regression-only, never a label metric
    assert result.eval_aggregate["r2"] > 0.9  # the Δt path recovers the linear-in-memory target
    assert all(fold_result.n_epochs == 1 for fold_result in result.folds)
    # aggregate is the numpy mean/std of the per-fold eval metrics
    for key in result.eval_aggregate:
        values = np.asarray([fr.eval_metrics[key] for fr in result.folds], dtype=np.float64)
        assert result.eval_aggregate[key] == float(values.mean())
        assert result.eval_std[key] == float(values.std())


def test_crossval_engages_the_dt_path():
    # Guardrail: the target was built under the TRUE gaps, so forwarding the correct dt must
    # out-score a run where dt is shuffled (gaps broken) -- proving the timing channel is used.
    X, y, dt, target_dt, seq_lengths = _memory_linear_3d()
    folds = walk_forward_folds(X.shape[0], n_folds=4)

    rng = np.random.default_rng(123)
    dt_shuffled = dt.copy()
    for row in range(dt.shape[0]):
        dt_shuffled[row, 1:] = rng.permutation(dt[row, 1:])  # break the true per-step gaps; keep dt[:,0]==0

    common = {"target_dt": target_dt, "seq_lengths": seq_lengths}
    r_true = cross_validate(_factory, X, y, folds, aux={"dt": dt, **common})
    r_shuffled = cross_validate(_factory, X, y, folds, aux={"dt": dt_shuffled, **common})

    assert r_true.eval_aggregate["r2"] > r_shuffled.eval_aggregate["r2"]


def test_crossval_is_deterministic():
    X, y, dt, target_dt, seq_lengths = _memory_linear_3d()
    folds = walk_forward_folds(X.shape[0], n_folds=3)
    aux = {"dt": dt, "target_dt": target_dt, "seq_lengths": seq_lengths}
    first = cross_validate(_factory, X, y, folds, aux=aux)
    second = cross_validate(_factory, X, y, folds, aux=aux)
    assert [fr.eval_metrics for fr in first.folds] == [fr.eval_metrics for fr in second.folds]
    assert first.eval_aggregate == second.eval_aggregate
