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


def test_full_view_is_derived_from_the_three_partitions():
    """Decision 11 retired ``*_full``; ``_full`` must rebuild it, not read a producer key.

    juniper-data#369 stopped every generator emitting ``X_full`` / ``y_full``, and
    ``bench/datasets.py`` read those keys directly -- so all seven datasets raised
    ``KeyError: 'X_full'``. The path-scoped bench CI lane never saw it, because the
    change that broke bench landed in juniper-data, not under ``bench/``.

    Pins both halves of the contract: the derived view is exactly the three partitions
    concatenated, and a producer's own ``*_full`` is preserved rather than recomputed.
    """
    from juniper_data.generators.irregular_sine import (
        IrregularSineGenerator,
        IrregularSineParams,
    )

    out = IrregularSineGenerator.generate(
        IrregularSineParams(n_steps=200, lookback=8, jitter=0.6, noise_std=0.0, seed=0)
    )
    assert "X_full" not in out, (
        "generator emitted a retired *_full key -- decision 11 regressed upstream"
    )
    partitions = [out[f"X_{s}"] for s in ("train", "val", "test")]

    derived = datasets._full(dict(out), "X")
    assert len(derived) == sum(len(p) for p in partitions)
    assert np.array_equal(derived, np.concatenate(partitions, axis=0))

    # A legacy artifact keeps the producer's own array, byte for byte.
    sentinel = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
    legacy = dict(out)
    legacy["X_full"] = sentinel
    assert np.array_equal(datasets._full(legacy, "X"), sentinel)


# --- W5.2: the trainval pool, the window keys, and date-cut folds per entity ---------------------------------


def _panel_arrays(n_per: int = 30, lookback: int = 4) -> dict:
    """Two tickers x three partitions with the W5.2 keys -- the shape juniper-data's equities_seq emits.

    Each partition holds ticker 0's windows then ticker 1's (split-major outside, entity blocks inside).
    ``window_end_date`` is a ``YYYYMMDD``-style integer that rises with the window; the splits sit in
    successive years so ``test`` is recognisable by its dates.
    """
    rng = np.random.default_rng(0)
    out: dict = {}
    for split, n in (("train", n_per), ("val", n_per // 2), ("test", n_per // 3)):
        parts: dict = {
            k: []
            for k in ("X", "y_reg", "dt", "target_dt", "ticker_code", "window_end_date")
        }
        base = {"train": 20200101, "val": 20210101, "test": 20220101}[split]
        for code in (0, 1):
            parts["X"].append(rng.standard_normal((n, lookback, 2)).astype(np.float32))
            parts["y_reg"].append(rng.standard_normal((n, 1)).astype(np.float32))
            dt = np.ones((n, lookback), dtype=np.float32)
            dt[:, 0] = 0.0
            parts["dt"].append(dt)
            parts["target_dt"].append(np.ones(n, dtype=np.float32))
            parts["ticker_code"].append(np.full(n, code, dtype=np.int64))
            parts["window_end_date"].append(base + np.arange(n, dtype=np.int64))
        for key, chunks in parts.items():
            out[f"{key}_{split}"] = np.concatenate(chunks, axis=0)
    return out


def test_trainval_view_excludes_test_and_carries_the_window_keys():
    out = _panel_arrays()
    arrays = dict(out)
    X = datasets._composed(arrays, "X", "trainval")
    assert X.shape[0] == out["X_train"].shape[0] + out["X_val"].shape[0]
    groups = datasets._composed(arrays, "ticker_code", "trainval")
    order = datasets._composed(arrays, "window_end_date", "trainval")
    assert sorted(set(groups.tolist())) == [0, 1]
    assert np.all(np.diff(groups) >= 0), "the composed view is entity-major"
    assert order.max() < 20220101, "no test window in the pool"
    # The full view is still the three partitions, and _full is its "full" case.
    assert datasets._full(dict(out), "X").shape[0] == sum(
        out[f"X_{s}"].shape[0] for s in ("train", "val", "test")
    )


def test_equities_dataset_pools_trainval_and_cuts_folds_on_the_date(monkeypatch):
    """``equities_seq`` loads the trainval pool with the keys and embargo = look-back; model-core then cuts
    every fold on one global date with the per-entity embargo (no network: the generator is stubbed)."""
    import juniper_data.generators.equities_seq as eq

    out = _panel_arrays(n_per=30, lookback=4)
    monkeypatch.setattr(
        eq.EquitiesSeqGenerator, "generate", staticmethod(lambda params: dict(out))
    )
    ds = datasets.equities_seq(
        symbols=("AAPL", "MSFT"), lookback=4, name="equities_seq_panel"
    )
    assert ds.name == "equities_seq_panel" and ds.grid == "irregular"
    assert ds.embargo == 4
    assert ds.X.shape[0] == 2 * (30 + 15), "train + val of both tickers; test excluded"
    assert ds.order is not None and ds.groups is not None
    folds = walk_forward_folds(
        ds.X.shape[0], n_folds=3, embargo=ds.embargo, order=ds.order, groups=ds.groups
    )
    assert folds, "the panel yields folds"
    for fold in folds:
        train, ev = np.asarray(fold.train_idx), np.asarray(fold.eval_idx)
        cut = int(ds.order[ev].min())
        assert ds.order[train].max() < cut, (
            "every train window precedes every eval window on the global date"
        )
        for code in (0, 1):
            before = np.sort(ds.order[(ds.groups == code) & (ds.order < cut)])
            kept = np.sort(ds.order[train[ds.groups[train] == code]])
            assert np.array_equal(
                kept, before[: max(0, before.shape[0] - ds.embargo)]
            ), "the last `embargo` windows of each entity before the cut are dropped"


def test_run_dataset_passes_the_keys_and_the_row_embargo_to_model_core(monkeypatch):
    from bench import run_benchmark

    captured: dict = {}
    real = run_benchmark.walk_forward_folds

    def spy(n, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return real(n, **kwargs)

    monkeypatch.setattr(run_benchmark, "walk_forward_folds", spy)
    monkeypatch.setattr(run_benchmark, "_TORCH_AVAILABLE", False)
    out = _panel_arrays(n_per=60, lookback=4)
    arrays = dict(out)
    ds = datasets.Dataset(
        "panel",
        "irregular",
        datasets._composed(arrays, "X", "trainval"),
        datasets._composed(arrays, "y_reg", "trainval"),
        datasets._composed(arrays, "dt", "trainval"),
        datasets._composed(arrays, "target_dt", "trainval"),
        order=datasets._composed(arrays, "window_end_date", "trainval"),
        groups=datasets._composed(arrays, "ticker_code", "trainval"),
        embargo=4,
    )
    result = run_benchmark.run_dataset(ds)
    assert (
        captured["embargo"] == 4
        and captured["order"] is ds.order
        and captured["groups"] is ds.groups
    )
    assert result["folding"] == {
        "embargo": 4,
        "ordered": True,
        "grouped": True,
        "n_entities": 2,
        "n_folds": run_benchmark._N_FOLDS,
    }
    # The synthetic path is unchanged: row-index folds, the harness embargo.
    run_benchmark.run_dataset(datasets.irregular_sine(n_steps=240, lookback=12, seed=0))
    assert (
        captured["embargo"] == run_benchmark._EMBARGO
        and captured["order"] is None
        and captured["groups"] is None
    )


def test_panel_row_is_a_registered_extension_and_band_2b_covers_it():
    from bench import run_benchmark

    assert "equities_seq_panel" in datasets.DATASETS
    assert "equities_seq_panel" not in datasets.PRIMARY_DATASETS
    assert datasets.DATASETS["equities_seq_panel"].keywords == {
        "symbols": ("AAPL", "MSFT"),
        "name": "equities_seq_panel",
    }
    var = f"lmu_var_d{run_benchmark._HEADLINE_D}"
    rkey = f"lmu_var_d{run_benchmark._HEADLINE_D}_ridge{run_benchmark._RIDGE_VARIANT}"
    row = {
        "mean": {"rmse": 0.02, "r2": -0.1},
        "std": {"rmse": 0.001, "r2": 0.03},
        "n_folds": 5,
    }
    models = {
        var: row,
        f"lmu_fixed_d{run_benchmark._HEADLINE_D}": row,
        "naive_persistence": row,
        "linear_ridge": row,
        rkey: row,
    }
    bands = run_benchmark.evaluate_bands(
        {"equities_seq_panel": {"grid": "irregular", "models": models}}
    )
    assert any(
        b["band"].startswith("2b — equities_seq_panel") and not b["primary"]
        for b in bands
    )
