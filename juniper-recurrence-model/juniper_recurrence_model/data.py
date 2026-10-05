"""Load a 3-D sequence NPZ artifact (the WS-1 contract) into arrays for the regressor.

The authoritative, full-contract validator is juniper-data-client's
``validate_npz_contract`` — the juniper-recurrence *app* calls it on the data-fetch path.
This module is the lean, **numpy-only model-side reader**: it pulls the per-split arrays
:class:`~juniper_recurrence_model.LMURegressor` consumes (``X`` / ``y`` / ``dt`` /
``target_dt`` / ``seq_lengths``) out of the NPZ key layout (per-split suffixes
``_train`` / ``_val`` / ``_test`` / ``_full``, the last **derived** from the partitions by
:func:`derive_full_split` when the artifact does not carry it — decision 11 retired the
``*_full`` family, juniper-data#369) and applies the minimal timing rules the model relies on
(``dt``, plus ``target_dt`` / ``seq_lengths`` when present). It deliberately takes **no**
juniper-data-client dependency, keeping this package numpy-only.

The WS-1 3-D contract (juniper-data#168; ``DELTA_T_HANDLING`` §6): ``X_{split}`` is ``(W, L, F)``;
``dt_{split}`` is ``(W, L)`` with ``dt[:, 0] == 0`` and ``dt >= 0`` (or absolute ``t_{split}``,
from which ``dt`` is derived); ``y_reg_{split}`` is the regression target (one per window);
``target_dt_{split}`` (horizon) and ``seq_lengths_{split}`` (valid step count) are optional.

The target is **selected explicitly** -- ``target=`` on :func:`sequence_data_from_arrays` and
:func:`load_sequence_npz`, defaulting to :data:`DEFAULT_TARGET` -- rather than by a silent
fallback from ``y_reg_{split}`` to the one-hot ``y_{split}`` (W1.3, finding F-S2 of juniper-ml
``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = ["DEFAULT_TARGET", "TARGET_MODES", "SequenceData", "TargetMode", "derive_full_split", "load_sequence_npz", "sequence_data_from_arrays"]

logger = logging.getLogger(__name__)

#: Which per-split key the reader takes as the target the regressor fits (W1.3). ``"reg"``
#: requires ``y_reg_{split}``; ``"class"`` requires ``y_{split}`` (the producer's primary label --
#: one-hot on a classification artifact such as equities' direction label); ``"auto"`` prefers
#: ``y_reg_{split}`` and falls back to ``y_{split}`` with a logged WARNING.
TargetMode = Literal["reg", "class", "auto"]

#: Every accepted ``target=`` value.
TARGET_MODES: tuple[str, ...] = ("reg", "class", "auto")

#: The ``target=`` default of every public entry in this module -- the single place the ruling
#: lives. ``"reg"`` applies the plan's recommended ruling R8 pending the owner's ruling; the
#: alternative ruling (keep ``"auto"`` as the default for one more release) is this one line.
DEFAULT_TARGET: TargetMode = "reg"


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


def load_sequence_npz(path: Any, split: str = "train", *, target: TargetMode = DEFAULT_TARGET) -> SequenceData:
    """Read one ``split`` (``"train"`` / ``"val"`` / ``"test"`` / ``"full"``) of a 3-D sequence ``.npz``.

    ``"full"`` is served from the artifact's own ``*_full`` family when it has one and is
    otherwise derived by :func:`derive_full_split`; ``target`` selects the target key. Both
    behave exactly as in :func:`sequence_data_from_arrays`.
    """
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    return sequence_data_from_arrays(arrays, split, target=target)


#: Partition suffixes that compose the whole dataset, in the order juniper-data laid them
#: down within each entity. Order is load-bearing -- see :func:`derive_full_split`.
_FULL_COMPONENT_SPLITS = ("train", "val", "test")

#: Per-window key naming the entity a window belongs to, when the generator emits one.
#: Its presence is what makes the legacy ``_full`` row order exactly reconstructible.
_ENTITY_KEY = "ticker_code"


def derive_full_split(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return ``arrays`` plus a synthesized ``*_full`` family built from the partitions.

    Decision 11 retires the ``*_full`` family from the NPZ contract (juniper-data#369), but
    cross-validation legitimately needs the whole set: ``POST /v1/crossval`` derives its
    walk-forward folds from it (D-CV-4). This rebuilds it consumer-side.

    **Row order is the whole difficulty, and it is not a concatenation.** juniper-data built
    ``X_full`` ENTITY-major -- each ticker's train windows, then its val, then its test,
    concatenated across tickers -- while the partitions themselves are SPLIT-major
    (every ticker's train, then every ticker's val, ...). See ``_assemble`` in
    ``juniper_data/generators/equities_seq/generator.py``. A plain
    ``concat(train, val, test)`` therefore holds the same rows in a DIFFERENT order
    whenever an artifact spans more than one ticker, and cross-validation slices by row
    index -- so a naive concatenation would silently change which windows land in which
    fold. That is a change to what the science measures, not a refactor.

    So: concatenate, then STABLE-sort by ``ticker_code``. Within one ticker the stable sort
    preserves the concatenation order (train, then val, then test), and within each
    (ticker, split) block it preserves chronological order -- which is exactly the
    entity-major layout juniper-data wrote. The sort must be stable; numpy's default
    quicksort would permute equal keys arbitrarily and silently reorder windows.

    When the artifact carries no ``ticker_code`` the plain concatenation is used. That is
    exact for any single-entity artifact (the only order there is), and it is the best
    available reconstruction otherwise.

    Keys already present in ``arrays`` are never overwritten: a legacy artifact that still
    ships ``*_full`` keeps the producer's own arrays, byte for byte.
    """
    present = [s for s in _FULL_COMPONENT_SPLITS if f"X_{s}" in arrays]
    if not present:
        return arrays

    # Every base key available in ALL present partitions can be composed; one that is
    # missing from any of them cannot, and is skipped rather than half-built.
    base_keys = {key[: -len(f"_{present[0]}")] for key in arrays if key.endswith(f"_{present[0]}")}
    composable = sorted(base for base in base_keys if all(f"{base}_{s}" in arrays for s in present))

    derived: dict[str, np.ndarray] = {}
    for base in composable:
        if f"{base}_full" in arrays:
            continue  # the producer's own array wins
        derived[f"{base}_full"] = np.concatenate([np.asarray(arrays[f"{base}_{s}"]) for s in present], axis=0)

    if not derived:
        return arrays

    entity_parts = [np.asarray(arrays[f"{_ENTITY_KEY}_{s}"]).reshape(-1) for s in present] if all(f"{_ENTITY_KEY}_{s}" in arrays for s in present) else None
    if entity_parts is not None:
        order = np.argsort(np.concatenate(entity_parts), kind="stable")
        derived = {key: value[order] for key, value in derived.items()}

    return {**arrays, **derived}


def sequence_data_from_arrays(arrays: dict[str, np.ndarray], split: str = "train", *, target: TargetMode = DEFAULT_TARGET) -> SequenceData:
    """Build a :class:`SequenceData` from an in-memory NPZ array mapping.

    Reads ``X_{split}`` (required, 3-D), the target ``target`` selects (below), and the timing
    channel ``dt_{split}`` (or derives it from ``t_{split}``). ``target_dt_{split}`` /
    ``seq_lengths_{split}`` are read when present. Applies the model-side timing checks: the
    ``dt`` rules (``(W, L)``, finite, ``>= 0``, ``dt[:, 0] == 0``) and, mirroring
    juniper-data-client's ``validate_npz_contract`` (W1.4), ``target_dt`` ``(W,)`` finite
    ``>= 0`` and ``seq_lengths`` ``(W,)`` of an integer dtype with every value in ``[1, L]``.

    ``target`` (keyword-only; default :data:`DEFAULT_TARGET`) chooses the target key:

    * ``"reg"`` -- the regression target ``y_reg_{split}``, required. An artifact without it
      raises ``ValueError("regression target 'y_reg_{split}' missing")``.
    * ``"class"`` -- the producer's primary label ``y_{split}``, required (one-hot on a
      classification artifact, e.g. equities' direction label).
    * ``"auto"`` -- the pre-W1.3 behaviour: ``y_reg_{split}`` when present, otherwise
      ``y_{split}`` with a logged WARNING naming the split and both keys. On a classification
      artifact that fallback turns a regression fit into a fit of the one-hot direction label
      (F-S2), which is why it is no longer silent and no longer the default.

    ``split`` must be ``"train"`` / ``"val"`` / ``"test"`` / ``"full"`` -- the four the app's
    request schema admits; anything else is refused before the artifact is read.
    ``split="full"`` is served from the artifact's own ``*_full`` family when it has one, and
    otherwise from :func:`derive_full_split`. juniper-data stopped emitting that family in
    decision 11, so without the fallback every post-#369 artifact would fail this read --
    which is how ``POST /v1/crossval`` would break.
    """
    if target not in TARGET_MODES:
        raise ValueError(f"target must be one of {TARGET_MODES}; got {target!r}")
    # The four splits the app's request schema admits (``SplitName``). Spelled as an inline literal
    # rather than derived from _FULL_COMPONENT_SPLITS: ``split`` reaches the target-fallback WARNING,
    # and a comparison against a literal display is what static analysis (CodeQL
    # ``py/log-injection``) recognises as validating a caller-supplied string.
    if split not in ("train", "val", "test", "full"):
        raise ValueError(f"split must be one of 'train' / 'val' / 'test' / 'full'; got {split!r}")
    if split == "full" and f"X_{split}" not in arrays:
        arrays = derive_full_split(arrays)
    if f"X_{split}" not in arrays:
        raise ValueError(f"NPZ artifact is missing required key 'X_{split}'")
    X = np.asarray(arrays[f"X_{split}"])
    if X.ndim != 3:
        raise ValueError(f"X_{split} must be 3-D (W, L, F) for a sequence artifact; got {X.ndim}-D")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"X_{split} has non-finite values (NaN/Inf)")
    n_windows, lookback = int(X.shape[0]), int(X.shape[1])

    y = _select_target(arrays, split, target)
    if y.ndim == 1:
        y = y[:, None]
    # ``X`` and ``dt`` are both checked for finiteness; ``y`` was not. That
    # asymmetry is the whole point of this line: a guard that exists to stop
    # non-finite values entering the model left the TARGET unguarded, so a NaN
    # there produced a NaN loss on the first backward pass -- the exact failure
    # the X check is written to prevent, arriving by the one door left open.
    #
    # The producer side made this worth closing rather than merely tidy:
    # juniper-data#378 defaults `equities` to `fundamentals_fill="nan"`, and
    # `y_reg` is derived from a price column, so a target built over a
    # pre-filing span is NaN by construction rather than by accident.
    if not np.all(np.isfinite(y)):
        raise ValueError(f"regression target for split '{split}' has non-finite values (NaN/Inf)")

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
    if not np.all(np.isfinite(dt)):
        raise ValueError(f"{dt_key} has non-finite gaps (NaN/Inf)")
    if np.any(dt < 0):
        raise ValueError(f"{dt_key} has negative gaps")
    if n_windows and np.any(dt[:, 0] != 0):
        raise ValueError(f"{dt_key}[:, 0] must be 0 by convention")

    target_dt = _read_target_dt(arrays, split, n_windows)
    seq_lengths = _read_seq_lengths(arrays, split, n_windows, lookback)

    return SequenceData(X=X, y=y, dt=dt, target_dt=target_dt, seq_lengths=seq_lengths)


def _select_target(arrays: dict[str, np.ndarray], split: str, target: str) -> np.ndarray:
    """Return the target array ``target`` selects for ``split`` (see :func:`sequence_data_from_arrays`)."""
    reg_key, class_key = f"y_reg_{split}", f"y_{split}"
    if target == "reg":
        if reg_key not in arrays:
            raise ValueError(f"regression target '{reg_key}' missing")
        return np.asarray(arrays[reg_key])
    if target == "class":
        if class_key not in arrays:
            raise ValueError(f"classification target '{class_key}' missing")
        return np.asarray(arrays[class_key])
    # "auto": the pre-W1.3 preference order, with the fallback made audible rather than silent.
    if reg_key in arrays:
        return np.asarray(arrays[reg_key])
    if class_key in arrays:
        logger.warning("split %r: regression target %r missing; target='auto' falls back to %r. On a classification artifact that key is the one-hot label, so the fit becomes a direction fit -- pass target='reg' or target='class' to choose explicitly.", split, reg_key, class_key)
        return np.asarray(arrays[class_key])
    raise ValueError(f"missing regression target: neither '{reg_key}' nor '{class_key}' present")


def _read_target_dt(arrays: dict[str, np.ndarray], split: str, n_windows: int) -> np.ndarray | None:
    """``target_dt_{split}`` when present: ``(W,)``, finite, ``>= 0`` (the W1.4 validator mirror).

    The horizon enters the readout's design matrix as a linear side-channel, so a NaN there
    surfaced as an opaque ``LinAlgError`` ("SVD did not converge") from the solve rather than
    as a contract error; a mis-shaped array used to be reshaped into place without a word.
    """
    key = f"target_dt_{split}"
    if key not in arrays:
        return None
    target_dt = np.asarray(arrays[key])
    if target_dt.shape != (n_windows,):
        raise ValueError(f"{key} shape {target_dt.shape} != {(n_windows,)}")
    if not np.all(np.isfinite(target_dt)):
        raise ValueError(f"{key} has non-finite horizons (NaN/Inf)")
    if np.any(target_dt < 0):
        raise ValueError(f"{key} has negative horizons")
    return target_dt


def _read_seq_lengths(arrays: dict[str, np.ndarray], split: str, n_windows: int, lookback: int) -> np.ndarray | None:
    """``seq_lengths_{split}`` when present: ``(W,)``, integer dtype, every value in ``[1, L]`` (W1.4 mirror).

    :class:`~juniper_recurrence_model.LMURegressor` reads each window's memory at step ``seq_lengths - 1`` and CLIPS that
    index into ``[0, L - 1]`` (``model.py`` ``_readout_index``), so an out-of-range length silently
    reads the wrong step instead of failing, and a float length is truncated the same way.
    """
    key = f"seq_lengths_{split}"
    if key not in arrays:
        return None
    seq_lengths = np.asarray(arrays[key])
    if seq_lengths.shape != (n_windows,):
        raise ValueError(f"{key} shape {seq_lengths.shape} != {(n_windows,)}")
    if not np.issubdtype(seq_lengths.dtype, np.integer):
        raise ValueError(f"{key} must be an integer dtype, got {seq_lengths.dtype}")
    if np.any((seq_lengths < 1) | (seq_lengths > lookback)):
        raise ValueError(f"{key} values must be in [1, {lookback}]")
    return seq_lengths
