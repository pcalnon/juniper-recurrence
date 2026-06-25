"""Producer/consumer contract test for the bench orchestration (audit TEST-03).

``bench/run_benchmark.py``'s leaf helpers are covered (``test_evaluate_bands`` exercises the band
arithmetic against hand-built dicts; ``test_bench_smoke`` exercises the dataset/baseline contracts),
but the **orchestration** -- ``run_dataset`` building the per-model CV dict that ``evaluate_bands``
then indexes *by string key* -- had no test. A drift in either the keys ``run_dataset`` emits or the
keys ``evaluate_bands`` reads would ship green. This runs ``run_dataset`` on the smallest dataset and
feeds its real output straight into ``evaluate_bands``, closing that gap.

``importorskip``s ``juniper_data`` so it is a no-op without the ``[bench]`` extra (the app's unit CI
does not install it); the dedicated bench CI lane does.
"""

from __future__ import annotations

import pytest

pytest.importorskip("juniper_data")  # the irregular_sine generator


def test_run_dataset_output_feeds_evaluate_bands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench import datasets, run_benchmark

    # Bound runtime + keep it deterministic across environments: collapse the d-sensitivity sweep to
    # the headline d (the comprehension drops d == _HEADLINE_D, so no extra sweep rows) and force the
    # optional torch-MLP leg off so the test never depends on whether [torch] is installed.
    monkeypatch.setattr(run_benchmark, "_D_GRID", (run_benchmark._HEADLINE_D,))
    monkeypatch.setattr(run_benchmark, "_TORCH_AVAILABLE", False)

    ds = datasets.irregular_sine(n_steps=240, lookback=12, seed=0)
    result = run_benchmark.run_dataset(ds)

    # --- Producer: the dict shape run_dataset promises -------------------------------------------
    assert {"name", "grid", "n_windows", "theta", "models"} <= set(result)
    assert result["name"] == ds.name
    assert result["grid"] == ds.grid
    assert result["models"], "run_dataset returned no model rows"
    headline_var = f"lmu_var_d{run_benchmark._HEADLINE_D}"
    headline_fixed = f"lmu_fixed_d{run_benchmark._HEADLINE_D}"
    for required in (headline_var, headline_fixed, "naive_persistence", "linear_ridge"):
        assert required in result["models"], f"missing model row {required!r}"
    for name, row in result["models"].items():
        assert {"mean", "std", "n_folds"} <= set(row), f"row {name!r} has wrong shape"
        assert "r2" in row["mean"] and "rmse" in row["mean"], (
            f"row {name!r} missing metrics"
        )

    # --- Consumer: evaluate_bands indexes the above by string ds-name -> must not KeyError --------
    bands = run_benchmark.evaluate_bands({ds.name: result})
    assert isinstance(bands, list) and bands, "evaluate_bands produced no bands"
    assert all({"band", "value", "pass", "primary"} <= set(b) for b in bands)
    # irregular_sine is grid="irregular" and the registered primary, so Band 1 must be present + primary.
    assert any(b["band"].startswith("1 ") and b["primary"] for b in bands)
