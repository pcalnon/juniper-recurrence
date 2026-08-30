"""Tests for the G-5 plotting module (``bench/plots.py``).

Two skip guards, for different reasons:

* ``juniper_data`` — the bench convention (see ``test_bench_smoke.py``): the app's unit CI does
  not install the ``[bench]`` extra, so dataset generation is unavailable there.
* ``matplotlib`` — the ``[bench-plots]`` extra. The bench CI lane installs it so these tests
  actually execute; without it they skip rather than fail, matching how the torch readout row
  degrades in ``run_benchmark``.

``test_pyplot_missing_is_actionable`` deliberately does NOT skip on matplotlib: the whole point
is the absent-dependency path, which is simulated rather than requiring an uninstalled env.
"""

from __future__ import annotations

import builtins
import importlib.util
import json

import numpy as np
import pytest

from bench import plots

pytest.importorskip("juniper_data")

# NOT importorskip: that raises at module level and would skip the ENTIRE file, including the
# absent-dependency tests above that are precisely about matplotlib being missing.
_HAS_MPL = importlib.util.find_spec("matplotlib") is not None
needs_mpl = pytest.mark.skipif(not _HAS_MPL, reason="requires the [bench-plots] extra")


def _small_dataset():
    from bench import datasets

    return datasets.irregular_sine(n_steps=240, lookback=12, seed=0)


def test_flat_collapses_multioutput():
    """A (n, k) target becomes (n,) by mean, so a future multi-output row degrades to a summary."""
    y = np.array([[1.0, 3.0], [0.0, 2.0]])
    out = plots._flat(y)
    assert out.shape == (2,)
    assert np.allclose(out, [2.0, 1.0])


def test_flat_passes_through_1d():
    y = np.array([1.0, 2.0, 3.0])
    assert np.allclose(plots._flat(y), y)


def test_pyplot_missing_is_actionable(monkeypatch):
    """An absent extra must yield the install hint, not a bare ModuleNotFoundError."""
    real_import = builtins.__import__

    def _no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)
    with pytest.raises(plots.PlotsUnavailable) as excinfo:
        plots._pyplot()
    assert "bench-plots" in str(excinfo.value)


def test_main_exits_2_when_extra_missing(monkeypatch, tmp_path, capsys):
    """The CLI reports the hint on stderr and exits 2 rather than raising."""

    def _unavailable():
        raise plots.PlotsUnavailable(plots._INSTALL_HINT)

    monkeypatch.setattr(plots, "_pyplot", _unavailable)
    rc = plots.main(["--out-dir", str(tmp_path), "--skip-forecast"])
    assert rc == 2
    assert "bench-plots" in capsys.readouterr().err


def test_model_comparison_returns_none_on_empty_dir(tmp_path):
    """No results must yield no figure — an empty chart would read as 'everything scored zero'."""
    assert plots.plot_model_comparison(tmp_path / "missing", tmp_path / "out") is None
    (tmp_path / "empty").mkdir()
    assert plots.plot_model_comparison(tmp_path / "empty", tmp_path / "out") is None


@needs_mpl
def test_model_comparison_writes_figure(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "toy.json").write_text(
        json.dumps(
            {
                "name": "toy",
                "models": {
                    "lmu_var_d16": {"mean": {"r2": 0.9, "rmse": 0.1}},
                    "naive_persistence": {"mean": {"r2": -0.2, "rmse": 0.5}},
                },
            }
        )
    )
    out = plots.plot_model_comparison(results, tmp_path / "out")
    assert out is not None and out.exists() and out.stat().st_size > 0


@needs_mpl
def test_model_comparison_survives_unusable_json(tmp_path):
    """One corrupt file must not lose the whole figure."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "broken.json").write_text("{not json")
    (results / "ok.json").write_text(
        json.dumps({"name": "ok", "models": {"m": {"mean": {"r2": 0.5}}}})
    )
    out = plots.plot_model_comparison(results, tmp_path / "out")
    assert out is not None and out.exists()


@needs_mpl
def test_dataset_figure_written(tmp_path):
    ds = _small_dataset()
    out = plots.plot_dataset(ds, tmp_path)
    assert out.exists() and out.stat().st_size > 0
    assert out.name.endswith("_dataset.png")


@needs_mpl
def test_forecast_figure_written(tmp_path):
    """The illustrative single-fold refit renders; predictions are not available from CV."""
    ds = _small_dataset()
    out = plots.plot_forecast_and_residuals(ds, tmp_path, d=8)
    assert out.exists() and out.stat().st_size > 0
    assert out.name.endswith("_forecast.png")


@needs_mpl
def test_main_skip_forecast_writes_no_forecast(tmp_path):
    rc = plots.main(
        [
            "--out-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "no-results"),
            "--datasets",
            "irregular_sine",
            "--skip-forecast",
        ]
    )
    assert rc == 0
    assert not list(tmp_path.glob("*_forecast.png"))
    assert list(tmp_path.glob("*_dataset.png"))


@needs_mpl
def test_main_unknown_dataset_is_skipped_not_fatal(tmp_path, capsys):
    rc = plots.main(
        [
            "--out-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "no-results"),
            "--datasets",
            "definitely_not_a_dataset",
            "--skip-forecast",
        ]
    )
    assert rc == 0
    assert "no such dataset" in capsys.readouterr().out
