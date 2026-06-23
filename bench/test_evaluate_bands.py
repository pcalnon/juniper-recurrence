"""Unit tests for ``bench.run_benchmark.evaluate_bands`` — the PASS/MISS band arithmetic.

``evaluate_bands`` turns a per-dataset metrics dict into the ratified OQ-14 acceptance
bands (plus the informational extensions). It is the logic that gates the Δt verdict, so
its pass/fail arithmetic is exercised here directly with hand-built ``results`` dicts (no
model training): the RMSE-reduction thesis (band 1), beats-naive / matches-linear (band
2), the equities regularized-readout band (2b), the regular-grid penalty (band 3), and
the DP-3 RFF capacity-vs-tie band (band 4).
"""

from __future__ import annotations

from typing import Any

from bench.run_benchmark import _HEADLINE_D, _RIDGE_VARIANT, evaluate_bands

_VAR = f"lmu_var_d{_HEADLINE_D}"
_FIXED = f"lmu_fixed_d{_HEADLINE_D}"
_RIDGE = f"lmu_var_d{_HEADLINE_D}_ridge{_RIDGE_VARIANT}"
_RFF = f"lmu_var_d{_HEADLINE_D}_rff"


def _m(rmse: float = 1.0, r2: float = 0.0, r2_std: float = 0.0) -> dict[str, Any]:
    """One model's CV summary (only the fields ``evaluate_bands`` reads)."""
    return {"mean": {"rmse": rmse, "r2": r2}, "std": {"rmse": 0.0, "r2": r2_std}}


def _dataset(grid: str, **models: Any) -> dict[str, Any]:
    """A results entry, pre-filled with the four models every band iterates."""
    base = {
        _VAR: _m(),
        _FIXED: _m(),
        "naive_persistence": _m(r2=-1.0),
        "linear_ridge": _m(),
    }
    base.update(models)
    return {"grid": grid, "models": base}


def _band(bands: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """The single band whose label starts with ``prefix`` (e.g. '1', '2 ', '2b')."""
    matches = [b for b in bands if b["band"].startswith(prefix)]
    assert len(matches) == 1, f"want 1 band {prefix!r}, got {len(matches)}"
    return matches[0]


def test_band1_thesis_passes_at_25pct_reduction() -> None:
    # fixed rmse 1.0 -> var rmse 0.70 == 30% reduction, over the 25% bar.
    results = {
        "irregular_sine": _dataset(
            "irregular", **{_VAR: _m(rmse=0.70), _FIXED: _m(rmse=1.0)}
        ),
    }
    band = _band(evaluate_bands(results), "1")
    assert band["pass"] is True
    assert band["primary"] is True  # irregular_sine is the canonical primary


def test_band1_thesis_misses_below_25pct_reduction() -> None:
    # 10% reduction -> miss; a non-canonical irregular dataset is informational.
    results = {
        "delay_product": _dataset(
            "irregular", **{_VAR: _m(rmse=0.90), _FIXED: _m(rmse=1.0)}
        ),
    }
    band = _band(evaluate_bands(results), "1")
    assert band["pass"] is False
    assert band["primary"] is False


def test_band1_zero_fixed_rmse_is_not_a_division_error() -> None:
    results = {"irregular_sine": _dataset("irregular", **{_FIXED: _m(rmse=0.0)})}
    band = _band(evaluate_bands(results), "1")
    assert band["pass"] is False  # improvement defaults to 0.0 when fixed rmse <= 0


def test_band2_passes_when_beats_naive_and_matches_linear() -> None:
    results = {
        "multi_sine": _dataset(
            "regular",
            **{
                _VAR: _m(r2=0.95),
                "naive_persistence": _m(r2=0.10),
                "linear_ridge": _m(r2=0.95),
            },
        )
    }
    band = _band(evaluate_bands(results), "2 ")  # trailing space avoids '2b'
    assert band["pass"] is True
    assert band["primary"] is True  # multi_sine is pre-registered


def test_band2_matches_linear_within_combined_std() -> None:
    # var r2 slightly below linear, but inside the summed std band -> a match.
    results = {
        "irregular_sine": _dataset(
            "irregular",
            **{
                _VAR: _m(r2=0.80, r2_std=0.05),
                "naive_persistence": _m(r2=0.10),
                "linear_ridge": _m(r2=0.83, r2_std=0.05),
            },
        )
    }
    band = _band(evaluate_bands(results), "2 ")
    assert band["pass"] is True


def test_band2_fails_when_not_beating_naive() -> None:
    results = {
        "mackey_glass": _dataset(
            "regular",
            **{
                _VAR: _m(r2=0.50),
                "naive_persistence": _m(r2=0.60),  # naive wins
                "linear_ridge": _m(r2=0.50),
            },
        )
    }
    band = _band(evaluate_bands(results), "2 ")
    assert band["pass"] is False


def test_band2b_equities_regularized_readout_beats_linear() -> None:
    results = {
        "equities_seq": _dataset(
            "irregular",
            **{
                _VAR: _m(r2=-0.50),  # ridge0 overfits the non-stationary target
                "linear_ridge": _m(r2=0.01),
                _RIDGE: _m(r2=0.02),  # regularized readout >= linear
            },
        )
    }
    band = _band(evaluate_bands(results), "2b")
    assert band["pass"] is True
    assert band["primary"] is False


def test_band3_regular_grid_no_penalty() -> None:
    # var within 10% of fixed on a regular grid -> pass.
    results = {
        "multi_sine": _dataset(
            "regular", **{_VAR: _m(rmse=1.05), _FIXED: _m(rmse=1.0)}
        ),
    }
    band = _band(evaluate_bands(results), "3")
    assert band["pass"] is True
    assert band["primary"] is True


def test_band3_regular_grid_penalty_too_large() -> None:
    results = {
        "mackey_glass": _dataset(
            "regular", **{_VAR: _m(rmse=1.30), _FIXED: _m(rmse=1.0)}
        ),
    }
    band = _band(evaluate_bands(results), "3")
    assert band["pass"] is False
    assert band["primary"] is False  # only multi_sine is the primary regular band


def test_band4_rff_capacity_gap_on_delay_product() -> None:
    # delay_product is the capacity dataset: RFF must beat linear by >= 0.10.
    results = {
        "delay_product": _dataset("irregular", **{_VAR: _m(r2=0.20), _RFF: _m(r2=0.90)})
    }
    band = _band(evaluate_bands(results), "4")
    assert band["pass"] is True  # 0.70 gap >= 0.10


def test_band4_rff_ties_linear_on_near_linear_dataset() -> None:
    # On a near-linear dataset RFF should merely tie linear (|gap| <= 0.10).
    results = {
        "irregular_sine": _dataset(
            "irregular", **{_VAR: _m(r2=0.95), _RFF: _m(r2=0.97)}
        )
    }
    band = _band(evaluate_bands(results), "4")
    assert band["pass"] is True  # 0.02 gap is a tie


def test_band4_rff_capacity_miss_when_no_gap() -> None:
    results = {
        "delay_product": _dataset(
            "irregular",
            **{_VAR: _m(r2=0.50), _RFF: _m(r2=0.52)},  # no real gap
        )
    }
    band = _band(evaluate_bands(results), "4")
    assert band["pass"] is False
