"""Dataset generation for the recurrence benchmark.

Generates the §3 datasets directly via the ``juniper-data`` generators (no live service),
returning the full ordered window set + the per-step Δt. Uses the ``*_full`` arrays (every
window, chronologically ordered) so the walk-forward CV does its own splitting.

Generator output (verified): ``X_full (n,T,F)``, ``y_full (n,output_dim)``, ``dt_full (n,T)``,
``target_dt_full (n,)`` (+ ``observed_mask_full``, ``*_train`` / ``*_test`` splits we ignore).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Dataset", "irregular_sine", "multi_sine", "mackey_glass", "DATASETS"]


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


def irregular_sine(*, n_steps: int = 2000, lookback: int = 32, jitter: float = 0.6, seed: int = 0) -> Dataset:
    """Irregular-Δt superimposed sinusoids — the thesis dataset (timing varies window-to-window)."""
    from juniper_data.generators.irregular_sine import IrregularSineGenerator, IrregularSineParams

    out = IrregularSineGenerator.generate(IrregularSineParams(n_steps=n_steps, lookback=lookback, jitter=jitter, seed=seed))
    return Dataset("irregular_sine", "irregular", _full(out, "X"), _full(out, "y"), _full(out, "dt"), _full(out, "target_dt"))


def multi_sine(*, n_steps: int = 2000, lookback: int = 32, seed: int = 0) -> Dataset:
    """Regular-Δt superimposed sinusoids — the control (Δt-awareness must not hurt here)."""
    from juniper_data.generators.multi_sine import MultiSineGenerator, MultiSineParams

    out = MultiSineGenerator.generate(MultiSineParams(n_steps=n_steps, lookback=lookback, seed=seed))
    return Dataset("multi_sine", "regular", _full(out, "X"), _full(out, "y"), _full(out, "dt"), _full(out, "target_dt"))


def mackey_glass(*, n_steps: int = 2000, lookback: int = 32, seed: int = 0) -> Dataset:
    """Regular-Δt Mackey-Glass chaotic series — a harder regular-Δt sanity check."""
    from juniper_data.generators.mackey_glass import MackeyGlassGenerator, MackeyGlassParams

    out = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=n_steps, lookback=lookback, seed=seed))
    return Dataset("mackey_glass", "regular", _full(out, "X"), _full(out, "y"), _full(out, "dt"), _full(out, "target_dt"))


#: name -> generator factory (the benchmark's pre-registered dataset set, DP-5 guardrail).
DATASETS = {
    "irregular_sine": irregular_sine,
    "multi_sine": multi_sine,
    "mackey_glass": mackey_glass,
}
