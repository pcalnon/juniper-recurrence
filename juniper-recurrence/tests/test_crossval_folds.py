"""W5.2: the fold builder behind ``POST /v1/crossval`` -- chronological, entity-grouped, label-aware.

The four tests the plan names (two tickers: no eval row precedes a train row in time; per-ticker
look-back non-overlap; label availability; ``test`` never in a fold -- the last one at the route,
``test_crossval_routes.py``) plus the date conversion, the embargo default and ``embargo_days``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from juniper_model_core.crossval import walk_forward_folds
from juniper_recurrence_model import SequenceData

from juniper_recurrence._folds import build_folds, yyyymmdd_to_days

LOOKBACK = 4


def _dates(start_days: int, n: int) -> np.ndarray:
    """``n`` consecutive calendar days from ``start_days`` (days since 1970), as YYYYMMDD int32."""
    days = (np.datetime64("1970-01-01") + np.arange(start_days, start_days + n).astype("timedelta64[D]")).astype("datetime64[D]")
    ymd = [(int(str(d)[0:4]) * 10000 + int(str(d)[5:7]) * 100 + int(str(d)[8:10])) for d in days]
    return np.asarray(ymd, dtype=np.int32)


def _panel(n_per_ticker: int = 30, tickers: int = 2, horizon: int = 1, keys: bool = True) -> SequenceData:
    """An entity-major two-ticker pool: ticker 0's windows (consecutive days), then ticker 1's (same days)."""
    rng = np.random.default_rng(0)
    n = n_per_ticker * tickers
    x = rng.standard_normal((n, LOOKBACK, 2)).astype("float32")
    y = rng.standard_normal((n, 1)).astype("float32")
    dt = np.zeros((n, LOOKBACK), dtype="float32")
    dt[:, 1:] = 1.0
    target_dt = np.full(n, float(horizon), dtype="float32")
    ticker = np.repeat(np.arange(tickers), n_per_ticker).astype(np.int32)
    dates = np.tile(_dates(18000, n_per_ticker), tickers)
    return SequenceData(X=x, y=y, dt=dt, target_dt=target_dt, ticker_code=ticker if keys else None, window_end_date=dates if keys else None)


def test_yyyymmdd_to_days_is_exact_and_ordered():
    days = yyyymmdd_to_days(np.array([20191231, 20200101, 20200229, 20200301], dtype=np.int32))
    assert np.array_equal(np.diff(days), [1, 59, 1])  # a leap year, counted
    assert days[0] == (np.datetime64("2019-12-31") - np.datetime64("1970-01-01")).astype(int)


@pytest.mark.parametrize("bad", [20200230, 20201301, 1234, 20200100])
def test_yyyymmdd_to_days_refuses_a_non_date(bad):
    with pytest.raises(ValueError, match="window_end_date"):
        yyyymmdd_to_days(np.array([20200101, bad], dtype=np.int32))


def test_two_tickers_no_eval_row_precedes_a_train_row_in_time():
    seq = _panel()
    folds, spec = build_folds(seq, n_folds=3, scheme="expanding", embargo=None, min_train=None, embargo_days=None)
    days = yyyymmdd_to_days(seq.window_end_date)
    assert spec.grouped and spec.ordered and spec.n_entities == 2 and spec.embargo == LOOKBACK
    for fold in folds:
        assert days[fold.train_idx].max() < days[fold.eval_idx].min()
        # Both tickers sit on both sides of every cut: the cut is a date, not a row position.
        assert set(seq.ticker_code[fold.train_idx].tolist()) == {0, 1} == set(seq.ticker_code[fold.eval_idx].tolist())


def test_per_ticker_lookback_windows_never_overlap_across_the_boundary():
    seq = _panel()
    folds, _ = build_folds(seq, n_folds=3, scheme="expanding", embargo=None, min_train=None, embargo_days=None)
    days = yyyymmdd_to_days(seq.window_end_date)
    for fold in folds:
        for ticker in (0, 1):
            train_end = days[fold.train_idx][seq.ticker_code[fold.train_idx] == ticker]
            eval_end = days[fold.eval_idx][seq.ticker_code[fold.eval_idx] == ticker]
            if train_end.size and eval_end.size:
                # An eval window's input spans [end - (LOOKBACK - 1), end]; no train window's span may touch it.
                assert train_end.max() < eval_end.min() - (LOOKBACK - 1)


def test_a_train_window_whose_label_crosses_the_cut_is_excluded():
    seq = _panel(horizon=3)
    folds, spec = build_folds(seq, n_folds=3, scheme="expanding", embargo=0, min_train=None, embargo_days=None)
    days = yyyymmdd_to_days(seq.window_end_date)
    assert spec.dropped_label_availability > 0  # with no window embargo, only this rule stands between the horizon and the cut
    for fold in folds:
        cut = days[fold.eval_idx].min()
        assert np.all(days[fold.train_idx] + seq.target_dt[fold.train_idx] < cut)


def test_embargo_days_drops_train_windows_inside_the_gap():
    seq = _panel()
    folds, spec = build_folds(seq, n_folds=3, scheme="expanding", embargo=0, min_train=None, embargo_days=5.0)
    days = yyyymmdd_to_days(seq.window_end_date)
    assert spec.embargo_days == 5.0 and spec.dropped_embargo_days > 0
    for fold in folds:
        assert days[fold.train_idx].max() < days[fold.eval_idx].min() - 5.0


def test_embargo_days_needs_dates():
    with pytest.raises(ValueError, match="embargo_days needs"):
        build_folds(_panel(keys=False), n_folds=3, scheme="expanding", embargo=None, min_train=None, embargo_days=2.0)


def test_an_explicit_embargo_below_the_lookback_is_honoured_and_logged(caplog):
    seq = _panel()
    with caplog.at_level(logging.WARNING, logger="juniper_recurrence._folds"):
        _, spec = build_folds(seq, n_folds=3, scheme="expanding", embargo=1, min_train=None, embargo_days=None)
    assert spec.embargo == 1
    assert "below the look-back" in caplog.text


def test_a_keyless_single_series_reduces_to_the_row_index_folds():
    seq = _panel(tickers=1, keys=False)
    folds, spec = build_folds(seq, n_folds=3, scheme="expanding", embargo=None, min_train=None, embargo_days=None)
    expected = walk_forward_folds(seq.X.shape[0], n_folds=3, scheme="expanding", embargo=LOOKBACK)
    assert not spec.grouped and not spec.ordered and spec.n_entities is None
    assert [f.train_idx.tolist() for f in folds] == [f.train_idx.tolist() for f in expected]
    assert [f.eval_idx.tolist() for f in folds] == [f.eval_idx.tolist() for f in expected]


def test_filters_that_empty_every_fold_are_refused():
    with pytest.raises(ValueError, match="no valid folds remain"):
        build_folds(_panel(n_per_ticker=12), n_folds=2, scheme="expanding", embargo=0, min_train=None, embargo_days=365.0)
