"""Smoke tests for the bench harness (run via ``pytest bench/``).

``importorskip``s ``juniper_data`` so the suite is a no-op without the ``[bench]`` extra (the app's
unit CI doesn't install it). The harness imports are safe without juniper-data — ``bench.datasets``
imports the generators lazily inside each function — so the skip guard only needs to gate the tests
that actually generate data.
"""

from __future__ import annotations

import numpy as np
import pytest
from juniper_model_core.crossval import cross_validate, walk_forward_folds
from juniper_recurrence_model import LMURegressor

from bench import baselines, datasets

pytest.importorskip("juniper_data")


def test_uniform_dt_is_unit_grid():
    dt = np.array([[0.0, 1.5, 2.0], [0.0, 0.5, 3.0]])
    u = baselines.uniform_dt(dt)
    assert u.shape == dt.shape
    assert np.all(u[:, 0] == 0.0) and np.all(u[:, 1:] == 1.0)


def test_irregular_sine_contract():
    ds = datasets.irregular_sine(n_steps=240, lookback=12, seed=0)
    assert ds.X.ndim == 3
    assert ds.dt.shape == ds.X.shape[:2]
    assert ds.y.shape[0] == ds.X.shape[0]
    assert ds.target_dt.shape[0] == ds.X.shape[0]
    assert np.all(ds.dt[:, 0] == 0.0)


def _cv(factory, ds, dt):
    folds = walk_forward_folds(ds.X.shape[0], n_folds=3, embargo=2)
    return cross_validate(
        factory, ds.X, ds.y, folds, aux={"dt": dt, "target_dt": ds.target_dt}
    )


def test_lmu_beats_naive_on_irregular():
    ds = datasets.irregular_sine(n_steps=400, lookback=16, seed=0)
    theta = float(np.median(ds.dt.sum(axis=1)))
    lmu = _cv(lambda i: LMURegressor(d=16, theta=theta), ds, ds.dt)
    naive = _cv(lambda i: baselines.NaivePersistence(), ds, ds.dt)
    assert lmu.eval_aggregate["r2"] > naive.eval_aggregate["r2"]


def test_variable_dt_beats_fixed_dt_on_irregular():
    ds = datasets.irregular_sine(n_steps=400, lookback=16, seed=0)
    theta = float(np.median(ds.dt.sum(axis=1)))
    var = _cv(lambda i: LMURegressor(d=16, theta=theta), ds, ds.dt)
    fixed = _cv(
        lambda i: LMURegressor(d=16, theta=theta), ds, baselines.uniform_dt(ds.dt)
    )
    assert var.eval_aggregate["rmse"] < fixed.eval_aggregate["rmse"]


def test_noise_std_perturbs_signal_but_keeps_contract():
    """The noise-sweep extension: noise_std>0 adds observation noise without breaking the contract."""
    clean = datasets.irregular_sine(n_steps=400, lookback=16, seed=0)
    noisy = datasets.irregular_sine(n_steps=400, lookback=16, noise_std=0.25, seed=0)
    assert noisy.X.shape == clean.X.shape
    assert np.all(noisy.dt[:, 0] == 0.0)
    assert not np.allclose(noisy.X, clean.X)  # the signal is genuinely perturbed


def test_dataset_registry_covers_primary_and_extensions():
    """DATASETS spans the pre-registered primary set plus the noise + real-data extensions."""
    assert set(datasets.PRIMARY_DATASETS) <= set(datasets.DATASETS)
    assert "equities_seq" in datasets.DATASETS
    assert sum("noise" in k for k in datasets.DATASETS) == 4
