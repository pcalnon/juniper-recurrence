"""Smoke tests for the bench harness (run via ``pytest bench/``).

``importorskip``s ``juniper_data`` so the suite is a no-op without the ``[bench]`` extra (the app's
unit CI doesn't install it). The harness imports are safe without juniper-data — ``bench.datasets``
imports the generators lazily inside each function — so the skip guard only needs to gate the tests
that actually generate data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from juniper_model_core.crossval import cross_validate, walk_forward_folds
from juniper_recurrence_model import LMURegressor, MLPReadoutSpec, RFFReadoutSpec

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


def test_ar_p_contract():
    """W-5 linear-floor extension: regular-Δt AR(2) rows honour the Dataset contract."""
    ds = datasets.ar_p(n_steps=240, lookback=12, seed=0)
    assert ds.name == "ar_p"
    assert ds.grid == "regular"
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


def test_delay_product_capacity_gap_rff_beats_linear():
    """DP-3 §8a capacity signature: on the delay_product dataset the nonlinear RFF readout fits the
    bilinear target the linear readout provably cannot, so RFF r² >> linear r² (a measured gap ≈ 0.7
    at this config). The same RFF readout merely ties the linear one on the near-linear synthetics."""
    ds = datasets.delay_product(n_steps=800, lookback=24, lag1=2, lag2=8, seed=0)
    theta = float(np.median(ds.dt.sum(axis=1)))
    linear = _cv(lambda i: LMURegressor(d=16, theta=theta), ds, ds.dt)
    rff = _cv(
        lambda i: LMURegressor(d=16, theta=theta, readout=RFFReadoutSpec()), ds, ds.dt
    )
    assert rff.eval_aggregate["r2"] > linear.eval_aggregate["r2"] + 0.2


def test_delay_product_capacity_gap_mlp_beats_linear():
    """DP-3 Rung 2b capacity: the torch MLP readout (default spec, full-budget — no early stopping in
    CV) fits the bilinear delay_product target the linear readout provably cannot, so MLP r² >> linear
    r² (measured gap ≈ 0.7). Requires the optional [torch] extra (`.[bench,bench-torch]`); skipped
    without it, so the torch-free bench CI lane is unaffected."""
    pytest.importorskip("torch")
    ds = datasets.delay_product(n_steps=800, lookback=24, lag1=2, lag2=8, seed=0)
    theta = float(np.median(ds.dt.sum(axis=1)))
    linear = _cv(lambda i: LMURegressor(d=16, theta=theta), ds, ds.dt)
    mlp = _cv(
        lambda i: LMURegressor(d=16, theta=theta, readout=MLPReadoutSpec()), ds, ds.dt
    )
    assert mlp.eval_aggregate["r2"] > linear.eval_aggregate["r2"] + 0.2


def test_noise_std_perturbs_signal_but_keeps_contract():
    """The noise-sweep extension: noise_std>0 adds observation noise without breaking the contract."""
    clean = datasets.irregular_sine(n_steps=400, lookback=16, seed=0)
    noisy = datasets.irregular_sine(n_steps=400, lookback=16, noise_std=0.25, seed=0)
    assert noisy.X.shape == clean.X.shape
    assert np.all(noisy.dt[:, 0] == 0.0)
    assert not np.allclose(noisy.X, clean.X)  # the signal is genuinely perturbed


def test_results_dir_flag_redirects_output(tmp_path, monkeypatch):
    """W-7/H-6: ``--results-dir`` redirects the per-dataset JSON + REPORT.md.

    The heavy pieces (``run_dataset``/``evaluate_bands``/``_render_report``) are
    stubbed — this pins ONLY the output-routing seam, not the benchmark itself.
    """
    from bench import run_benchmark

    monkeypatch.setattr(
        run_benchmark.datasets,
        "DATASETS",
        {
            "irregular_sine": lambda: datasets.irregular_sine(
                n_steps=240, lookback=12, seed=0
            )
        },
    )
    monkeypatch.setattr(run_benchmark, "run_dataset", lambda ds: {"stub": True})
    monkeypatch.setattr(run_benchmark, "evaluate_bands", lambda results: [])
    monkeypatch.setattr(run_benchmark, "_render_report", lambda *a: "stub-report")
    out = tmp_path / "alt-results"
    run_benchmark.main(["--results-dir", str(out)])
    assert (out / "irregular_sine.json").is_file()
    assert (out / "REPORT.md").read_text() == "stub-report"


def test_results_dir_default_flows_from_module_constant(tmp_path, monkeypatch):
    """No flag ⇒ output lands in ``_RESULTS`` (bench/results in real runs) — the
    default target is unchanged by W-7. Patched here so the test never writes
    into the checkout's committed baseline home."""
    from bench import run_benchmark

    assert (
        run_benchmark._RESULTS
        == Path(run_benchmark.__file__).resolve().parent / "results"
    )
    monkeypatch.setattr(run_benchmark.datasets, "DATASETS", {})
    monkeypatch.setattr(run_benchmark, "evaluate_bands", lambda results: [])
    monkeypatch.setattr(run_benchmark, "_render_report", lambda *a: "stub-report")
    monkeypatch.setattr(run_benchmark, "_RESULTS", tmp_path / "default-home")
    run_benchmark.main([])
    assert (tmp_path / "default-home" / "REPORT.md").is_file()


def test_dataset_registry_covers_primary_and_extensions():
    """DATASETS spans the pre-registered primary set plus the noise / capacity / real-data extensions."""
    assert set(datasets.PRIMARY_DATASETS) <= set(datasets.DATASETS)
    assert "equities_seq" in datasets.DATASETS
    assert "delay_product" in datasets.DATASETS
    assert "ar_p" in datasets.DATASETS
    assert (
        "ar_p" not in datasets.PRIMARY_DATASETS
    )  # W-5: linear-floor extension, never scored
    assert sum("noise" in k for k in datasets.DATASETS) == 4
