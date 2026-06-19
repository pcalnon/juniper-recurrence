"""Dataset generation for the recurrence benchmark.

Generates the §3 datasets directly via the ``juniper-data`` generators (no live service),
returning the full ordered window set + the per-step Δt. Uses the ``*_full`` arrays (every
window, chronologically ordered) so the walk-forward CV does its own splitting.

Generator output (verified): ``X_full (n,T,F)``, ``y_full (n,output_dim)``, ``dt_full (n,T)``,
``target_dt_full (n,)`` (+ ``observed_mask_full``, ``*_train`` / ``*_test`` splits we ignore).
For ``equities_seq`` the regression target is ``y_reg_full`` (next-day close), not the one-hot
``y_full`` direction label.

The ``DATASETS`` set is the pre-registered benchmark scope (DP-5 guardrail). Beyond the three
synthetic datasets of the ratified eval it adds two **extensions**: a noise-robustness sweep
(``*_noise0.10`` / ``*_noise0.25`` — does the Δt advantage survive observation noise?) and a
real-data irregular-Δt sanity check (``equities_seq``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

__all__ = ["Dataset", "irregular_sine", "multi_sine", "mackey_glass", "equities_seq", "DATASETS", "PRIMARY_DATASETS"]


@dataclass(frozen=True)
class Dataset:
    """A generated, chronologically-ordered window set for the benchmark."""

    name: str
    grid: str  # "irregular" | "regular"
    X: np.ndarray  # (n, T, F)
    y: np.ndarray  # (n, output_dim)
    dt: np.ndarray  # (n, T) real per-step gaps (dt[:, 0] == 0)
    target_dt: np.ndarray  # (n,) irregular forecast horizon


def _full(out: dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.asarray(out[f"{key}_full"])


def irregular_sine(*, n_steps: int = 2000, lookback: int = 32, jitter: float = 0.6, noise_std: float = 0.0, seed: int = 0, name: str = "irregular_sine") -> Dataset:
    """Irregular-Δt superimposed sinusoids — the thesis dataset (timing varies window-to-window).

    ``noise_std`` adds Gaussian observation noise (0 = the exact closed-form signal); the
    robustness sweep raises it to check the Δt advantage survives a noisy signal.
    """
    from juniper_data.generators.irregular_sine import IrregularSineGenerator, IrregularSineParams

    out = IrregularSineGenerator.generate(IrregularSineParams(n_steps=n_steps, lookback=lookback, jitter=jitter, noise_std=noise_std, seed=seed))
    return Dataset(name, "irregular", _full(out, "X"), _full(out, "y"), _full(out, "dt"), _full(out, "target_dt"))


def multi_sine(*, n_steps: int = 2000, lookback: int = 32, noise_std: float = 0.0, seed: int = 0, name: str = "multi_sine") -> Dataset:
    """Regular-Δt superimposed sinusoids — the control (Δt-awareness must not hurt here)."""
    from juniper_data.generators.multi_sine import MultiSineGenerator, MultiSineParams

    out = MultiSineGenerator.generate(MultiSineParams(n_steps=n_steps, lookback=lookback, noise_std=noise_std, seed=seed))
    return Dataset(name, "regular", _full(out, "X"), _full(out, "y"), _full(out, "dt"), _full(out, "target_dt"))


def mackey_glass(*, n_steps: int = 2000, lookback: int = 32, seed: int = 0) -> Dataset:
    """Regular-Δt Mackey-Glass chaotic series — a harder regular-Δt sanity check."""
    from juniper_data.generators.mackey_glass import MackeyGlassGenerator, MackeyGlassParams

    out = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=n_steps, lookback=lookback, seed=seed))
    return Dataset("mackey_glass", "regular", _full(out, "X"), _full(out, "y"), _full(out, "dt"), _full(out, "target_dt"))


def equities_seq(*, symbols: tuple[str, ...] = ("AAPL",), start_date: str = "2010-01-01", end_date: str = "2022-01-01", lookback: int = 32) -> Dataset:
    """Real irregular-Δt sequences from one equity's daily history (calendar gaps = genuine Δt).

    Uses a **single ticker** so the concatenated window set is one chronological series (clean
    walk-forward CV — no cross-ticker interleaving). The regression target is ``y_reg``
    (next-day close), not the one-hot direction label. Requires the ``[equities]`` extra
    (yfinance / pandas) and network access at generation time (Yahoo Finance + SEC EDGAR); a
    network failure is reported by the runner, not fatal.

    Known gap: juniper-data 0.7.0's wheel omits the bundled ``sp500_constituents.csv`` (a
    packaging defect — the equities generators ship without their data file), so a pip install
    of ``juniper-data[equities]==0.7.0`` raises ``FileNotFoundError`` here and the runner SKIPs
    this row. Until juniper-data 0.7.1 ships the CSV, generate the equities row from a
    source / editable install of juniper-data.
    """
    from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams

    out = EquitiesSeqGenerator.generate(EquitiesSeqParams(symbols=list(symbols), start_date=start_date, end_date=end_date, lookback=lookback, normalize_features=True, use_cache=True))
    return Dataset("equities_seq", "irregular", _full(out, "X"), _full(out, "y_reg"), _full(out, "dt"), _full(out, "target_dt"))


#: The three pre-registered datasets the ratified OQ-14 bands are scored against (DP-5 guardrail).
PRIMARY_DATASETS = ("irregular_sine", "multi_sine", "mackey_glass")

#: name -> generator factory. The primary set plus the two extensions (noise sweep + real data).
#: Noise variants probe whether the Δt advantage survives observation noise; ``equities_seq`` is
#: the real-data irregular-Δt sanity check. Extension results are reported as informational —
#: they are not scored against the ratified bands (which were pre-registered for the primary set).
DATASETS = {
    "irregular_sine": irregular_sine,
    "irregular_sine_noise0.10": partial(irregular_sine, noise_std=0.10, name="irregular_sine_noise0.10"),
    "irregular_sine_noise0.25": partial(irregular_sine, noise_std=0.25, name="irregular_sine_noise0.25"),
    "multi_sine": multi_sine,
    "multi_sine_noise0.10": partial(multi_sine, noise_std=0.10, name="multi_sine_noise0.10"),
    "multi_sine_noise0.25": partial(multi_sine, noise_std=0.25, name="multi_sine_noise0.25"),
    "mackey_glass": mackey_glass,
    "equities_seq": equities_seq,
}
