"""Load a 3-D sequence NPZ artifact (the WS-1 contract) into arrays for the regressor.

The authoritative, full-contract validator is juniper-data-client's
``validate_npz_contract`` — the juniper-recurrence *app* calls it on the data-fetch path.
This module is the lean, **numpy-only model-side reader**: it pulls the per-split arrays
:class:`~juniper_recurrence_model.LMURegressor` consumes (``X`` / ``y`` / ``dt`` /
``target_dt`` / ``seq_lengths``) out of the NPZ key layout (per-split suffixes
``_train`` / ``_test`` / ``_full``) and applies the minimal ``dt`` rules the model relies on.
It deliberately takes **no** juniper-data-client dependency, keeping this package numpy-only.

The WS-1 3-D contract (juniper-data#168; ``DELTA_T_HANDLING`` §6): ``X_{split}`` is ``(W, L, F)``;
``dt_{split}`` is ``(W, L)`` with ``dt[:, 0] == 0`` and ``dt >= 0`` (or absolute ``t_{split}``,
from which ``dt`` is derived); ``y_reg_{split}`` is the regression target (one per window);
``target_dt_{split}`` (horizon) and ``seq_lengths_{split}`` (valid step count) are optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SequenceData", "load_sequence_npz", "sequence_data_from_arrays"]


@dataclass(frozen=True)
class SequenceData:
    """One split of a 3-D sequence artifact, ready for :class:`LMURegressor`.

    ``X`` is ``(W, L, F)``; ``y`` is ``(W, output_dim)``; ``dt`` is ``(W, L)`` with
    ``dt[:, 0] == 0``. ``target_dt`` ``(W,)`` and ``seq_lengths`` ``(W,)`` are optional.
    """

    X: np.ndarray
    y: np.ndarray
    dt: np.ndarray
    target_dt: np.ndarray | None = None
    seq_lengths: np.ndarray | None = None

    def fit_kwargs(self) -> dict[str, Any]:
        """The auxiliary-array keywords for ``LMURegressor.fit`` / ``predict`` (the D3 contract)."""
        kwargs: dict[str, Any] = {"dt": self.dt}
        if self.target_dt is not None:
            kwargs["target_dt"] = self.target_dt
        if self.seq_lengths is not None:
            kwargs["seq_lengths"] = self.seq_lengths
        return kwargs


def load_sequence_npz(path: Any, split: str = "train") -> SequenceData:
    """Read one ``split`` (``"train"`` / ``"test"`` / ``"full"``) of a 3-D sequence ``.npz``."""
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    return sequence_data_from_arrays(arrays, split)


def sequence_data_from_arrays(arrays: dict[str, np.ndarray], split: str = "train") -> SequenceData:
    """Build a :class:`SequenceData` from an in-memory NPZ array mapping.

    Reads ``X_{split}`` (required, 3-D), the regression target ``y_reg_{split}`` (preferred;
    falls back to ``y_{split}``), and the timing channel ``dt_{split}`` (or derives it from
    ``t_{split}``). ``target_dt_{split}`` / ``seq_lengths_{split}`` are read when present.
    Applies the minimal model-side ``dt`` checks (a strict subset of ``validate_npz_contract``).
    """
    if f"X_{split}" not in arrays:
        raise ValueError(f"NPZ artifact is missing required key 'X_{split}'")
    X = np.asarray(arrays[f"X_{split}"])
    if X.ndim != 3:
        raise ValueError(f"X_{split} must be 3-D (W, L, F) for a sequence artifact; got {X.ndim}-D")
    n_windows, lookback = int(X.shape[0]), int(X.shape[1])

    # Regression target: prefer y_reg, fall back to y.
    if f"y_reg_{split}" in arrays:
        y = np.asarray(arrays[f"y_reg_{split}"])
    elif f"y_{split}" in arrays:
        y = np.asarray(arrays[f"y_{split}"])
    else:
        raise ValueError(f"missing regression target: neither 'y_reg_{split}' nor 'y_{split}' present")
    if y.ndim == 1:
        y = y[:, None]

    # Timing: dt directly, or derived from absolute t (matches the contract's t/dt consistency).
    dt_key, t_key = f"dt_{split}", f"t_{split}"
    if dt_key in arrays:
        dt = np.asarray(arrays[dt_key], dtype=float)
    elif t_key in arrays:
        t = np.asarray(arrays[t_key], dtype=float)
        dt = np.zeros_like(t)
        dt[:, 1:] = np.diff(t, axis=1)
    else:
        raise ValueError(f"a 3-D artifact needs at least one of 'dt_{split}' / 't_{split}'")
    if dt.shape != (n_windows, lookback):
        raise ValueError(f"{dt_key} shape {dt.shape} != {(n_windows, lookback)}")
    if np.any(dt < 0):
        raise ValueError(f"{dt_key} has negative gaps")
    if n_windows and np.any(dt[:, 0] != 0):
        raise ValueError(f"{dt_key}[:, 0] must be 0 by convention")

    target_dt = np.asarray(arrays[f"target_dt_{split}"]).reshape(n_windows) if f"target_dt_{split}" in arrays else None
    seq_lengths = np.asarray(arrays[f"seq_lengths_{split}"]).reshape(n_windows) if f"seq_lengths_{split}" in arrays else None

    return SequenceData(X=X, y=y, dt=dt, target_dt=target_dt, seq_lengths=seq_lengths)
