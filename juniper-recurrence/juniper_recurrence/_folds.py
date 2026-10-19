"""Fold construction for ``POST /v1/crossval`` (plan W5.2): chronological, entity-grouped, ``test``-free.

Before W5.2 the route cut row-index folds over the ``full`` view. On a panel artifact (several
tickers) that is wrong twice over: ``full`` is laid down entity-major (every window of ticker A, then
every window of ticker B), so a row-index cut puts ticker A's 2021 in train and ticker B's 2015 in
eval; and its last fold evaluates the ``test`` partition the plan reserves for one out-of-sample read
(W5.3). Here the pool is the composed ``trainval`` view, and when the artifact carries the per-window
keys ``juniper-recurrence-model`` 0.6.0 exposes -- ``window_end_date`` (a ``YYYYMMDD`` integer) and
``ticker_code`` -- the folds are cut on the **global date** through model-core's ``order=`` /
``groups=``: every entity's windows before the cut are train, after it eval, and the embargo purge is
applied **per entity** (its last ``embargo`` windows before the cut), which is what keeps one ticker's
look-back from straddling the boundary.

Three rules the plan sets, all encoded here:

* **``embargo`` is in windows and defaults to the look-back.** A window's input spans its last
  ``lookback`` steps, so a smaller gap lets an eval window's history overlap a train window's. An
  explicit smaller value is honoured (it is a methodological choice, and the E-G suite sweeps it) but
  logged at WARNING; the resolved value is on the response.
* **Label availability**: a train window whose label is not yet known at the cut -- its
  ``window_end_date + target_dt`` reaches the first eval day -- is dropped from train.
* **``embargo_days``** is a separately named *duration* gap: train windows ending within that many
  days before the cut are dropped as well.

A single-entity artifact (one ticker, or no keys at all) reduces to the row-index folds of before,
with the same per-window embargo. The dates are converted to day ordinals for the arithmetic;
``YYYYMMDD`` integers sort chronologically as they are, so the sort key needs no conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from juniper_model_core.crossval import Fold, walk_forward_folds
from juniper_recurrence_model import SequenceData

__all__ = ["POOL_SPLIT", "FoldingSpec", "build_folds", "yyyymmdd_to_days"]

logger = logging.getLogger(__name__)

#: The split ``POST /v1/crossval`` selects over: train | val, never ``test`` (plan W5.2 / W5.3).
POOL_SPLIT = "trainval"


@dataclass(frozen=True)
class FoldingSpec:
    """How the folds were built -- the record the response carries (``CrossValFolding``)."""

    lookback: int
    embargo: int
    embargo_days: float | None
    grouped: bool
    ordered: bool
    n_entities: int | None
    n_folds: int
    dropped_label_availability: int
    dropped_embargo_days: int


def yyyymmdd_to_days(values: np.ndarray) -> np.ndarray:
    """``YYYYMMDD`` integers -> days since 1970-01-01 (``int64``), refusing anything that is not a calendar date."""
    v = np.asarray(values).astype(np.int64)
    year, month, day = v // 10000, (v // 100) % 100, v % 100
    if v.size and bool(np.any((year < 1900) | (year > 2200) | (month < 1) | (month > 12) | (day < 1) | (day > 31))):
        raise ValueError("window_end_date must hold YYYYMMDD calendar dates (e.g. 20150406)")
    months = (year - 1970).astype("datetime64[Y]").astype("datetime64[M]") + (month - 1).astype("timedelta64[M]")
    dates = months.astype("datetime64[D]") + (day - 1).astype("timedelta64[D]")
    if v.size and bool(np.any(dates.astype("datetime64[M]") != months)):
        raise ValueError("window_end_date holds a day that does not exist in its month (e.g. 20200230)")
    return dates.astype(np.int64)


def build_folds(sequence: SequenceData, *, n_folds: int, scheme: str, embargo: int | None, min_train: int | None, embargo_days: float | None) -> tuple[list[Fold], FoldingSpec]:
    """Cut the walk-forward folds for ``sequence`` (the ``trainval`` pool) per the W5.2 rules.

    Raises ``ValueError`` (the route answers 422) when model-core refuses the configuration, when
    ``embargo_days`` is asked of an artifact with no ``window_end_date``, or when the date filters
    leave no fold with a training set.
    """
    n, lookback = int(sequence.X.shape[0]), int(sequence.X.shape[1])
    resolved_embargo = lookback if embargo is None else int(embargo)
    if embargo is not None and embargo < lookback:
        logger.warning("crossval embargo=%d is below the look-back of %d windows, so an eval window's history can overlap a train window's; omit embargo for the default of one look-back (plan W5.2)", embargo, lookback)
    order = sequence.window_end_date
    groups = sequence.ticker_code
    if embargo_days is not None and order is None:
        raise ValueError("embargo_days needs the artifact's window_end_date; this artifact has none (use embargo, in windows)")

    folds = walk_forward_folds(n, n_folds=n_folds, scheme=scheme, embargo=resolved_embargo, min_train=min_train, order=order, groups=groups)

    dropped_label = dropped_days = 0
    if order is not None:
        days = yyyymmdd_to_days(order)
        horizon = np.zeros(n, dtype=float) if sequence.target_dt is None else np.asarray(sequence.target_dt, dtype=float)
        kept: list[Fold] = []
        for fold in folds:
            train, ev = np.asarray(fold.train_idx), np.asarray(fold.eval_idx)
            cut = int(days[ev].min())  # the first eval day: the cut every entity shares
            keep = (days[train] + horizon[train]) < cut  # the label is known strictly before the cut
            dropped_label += int(np.sum(~keep))
            if embargo_days is not None:
                gap = days[train] < cut - float(embargo_days)
                dropped_days += int(np.sum(keep & ~gap))
                keep &= gap
            train = train[keep]
            if train.shape[0] < 1 or (scheme == "expanding" and min_train is not None and train.shape[0] < min_train):
                continue
            kept.append(Fold(train_idx=train, eval_idx=ev))
        if not kept:
            raise ValueError("no valid folds remain after the label-availability / embargo_days filters; reduce n_folds, embargo_days or min_train")
        folds = kept

    spec = FoldingSpec(
        lookback=lookback,
        embargo=resolved_embargo,
        embargo_days=None if embargo_days is None else float(embargo_days),
        grouped=groups is not None,
        ordered=order is not None,
        n_entities=None if groups is None else int(np.unique(groups).shape[0]),
        n_folds=len(folds),
        dropped_label_availability=dropped_label,
        dropped_embargo_days=dropped_days,
    )
    return folds, spec
