"""Tests for the 3-D sequence NPZ loader + the end-to-end irregular-Δt consumer path (§9.1c).

Synthesises an ``equities_seq``-shaped 3-D artifact (per-split ``X`` / ``y_reg`` / ``dt`` /
``target_dt`` / ``seq_lengths``), loads it via :func:`load_sequence_npz`, and trains + predicts
:class:`LMURegressor` on it — the juniper-recurrence consumer ingesting the shipped WS-1 contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from juniper_recurrence_model import LMURegressor, SequenceData, load_sequence_npz, sequence_data_from_arrays


def _make_equities_seq_arrays(splits=("train", "test"), w=40, lookback=12, n_features=4, seed=0):
    """An equities_seq-shaped 3-D NPZ array mapping (one regression target per window)."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for split in splits:
        n = w if split == "train" else w // 2
        arrays[f"X_{split}"] = rng.normal(size=(n, lookback, n_features)).astype(np.float32)
        dt = np.zeros((n, lookback), dtype=np.float32)
        dt[:, 1:] = rng.integers(1, 4, size=(n, lookback - 1)).astype(np.float32)  # calendar-day gaps
        arrays[f"dt_{split}"] = dt
        arrays[f"y_reg_{split}"] = rng.normal(size=(n, 1)).astype(np.float32)
        arrays[f"target_dt_{split}"] = rng.integers(1, 4, size=n).astype(np.float32)
        arrays[f"seq_lengths_{split}"] = np.full(n, lookback, dtype=np.int64)
    return arrays


def test_load_sequence_npz_roundtrip(tmp_path):
    path = tmp_path / "equities_seq.npz"
    np.savez(path, **_make_equities_seq_arrays())
    data = load_sequence_npz(path, split="train")
    assert isinstance(data, SequenceData)
    assert data.X.shape == (40, 12, 4)
    assert data.y.shape == (40, 1)
    assert data.dt.shape == (40, 12) and np.all(data.dt[:, 0] == 0)
    assert data.target_dt.shape == (40,)
    assert data.seq_lengths.shape == (40,)
    assert set(data.fit_kwargs()) == {"dt", "target_dt", "seq_lengths"}


def test_end_to_end_fit_predict_on_sequence_npz(tmp_path):
    """The irregular-Δt consumer path: load a 3-D NPZ and train/predict LMURegressor end-to-end."""
    path = tmp_path / "equities_seq.npz"
    np.savez(path, **_make_equities_seq_arrays())
    train = load_sequence_npz(path, split="train")
    test = load_sequence_npz(path, split="test")

    model = LMURegressor(d=16)  # theta data-driven from the windows' dt
    result = model.fit(train.X, train.y, **train.fit_kwargs())
    assert result.n_epochs >= 1
    preds = model.predict(test.X, **test.fit_kwargs())
    assert preds.shape == (test.X.shape[0], 1)
    assert np.all(np.isfinite(preds))
    assert model.theta is not None and model.theta > 0  # resolved from dt


def test_loader_derives_dt_from_absolute_t():
    """When only absolute t is present, dt is derived (dt[:,0]=0, dt[:,1:]=diff(t))."""
    rng = np.random.default_rng(1)
    n, lookback, n_features = 6, 8, 3
    t = np.cumsum(rng.integers(1, 4, size=(n, lookback)).astype(np.float32), axis=1)
    arrays = {
        "X_train": rng.normal(size=(n, lookback, n_features)).astype(np.float32),
        "y_reg_train": rng.normal(size=(n, 1)).astype(np.float32),
        "t_train": t,
    }
    data = sequence_data_from_arrays(arrays, "train")
    assert np.all(data.dt[:, 0] == 0)
    assert np.allclose(data.dt[:, 1:], np.diff(t, axis=1))


def test_loader_falls_back_to_y_when_no_y_reg():
    rng = np.random.default_rng(2)
    arrays = {
        "X_train": rng.normal(size=(5, 4, 2)).astype(np.float32),
        "y_train": rng.normal(size=(5, 1)).astype(np.float32),  # no y_reg
        "dt_train": np.zeros((5, 4), dtype=np.float32),
    }
    assert sequence_data_from_arrays(arrays, "train").y.shape == (5, 1)


def test_loader_rejects_2d_x():
    arrays = {"X_train": np.zeros((5, 4), dtype=np.float32), "y_reg_train": np.zeros((5, 1), dtype=np.float32), "dt_train": np.zeros((5, 4), dtype=np.float32)}
    with pytest.raises(ValueError):
        sequence_data_from_arrays(arrays, "train")


def test_loader_requires_dt_or_t():
    arrays = {"X_train": np.zeros((5, 4, 2), dtype=np.float32), "y_reg_train": np.zeros((5, 1), dtype=np.float32)}
    with pytest.raises(ValueError):
        sequence_data_from_arrays(arrays, "train")


def test_loader_rejects_bad_dt_first_column():
    arrays = {
        "X_train": np.zeros((3, 4, 2), dtype=np.float32),
        "y_reg_train": np.zeros((3, 1), dtype=np.float32),
        "dt_train": np.ones((3, 4), dtype=np.float32),  # dt[:, 0] != 0
    }
    with pytest.raises(ValueError):
        sequence_data_from_arrays(arrays, "train")


def test_loader_rejects_missing_x():
    with pytest.raises(ValueError):
        sequence_data_from_arrays({"y_reg_train": np.zeros((3, 1))}, "train")


def test_loader_accepts_1d_y_reg():
    rng = np.random.default_rng(3)
    arrays = {
        "X_train": rng.normal(size=(5, 4, 2)).astype(np.float32),
        "y_reg_train": rng.normal(size=5).astype(np.float32),  # 1-D target -> (5, 1)
        "dt_train": np.zeros((5, 4), dtype=np.float32),
    }
    assert sequence_data_from_arrays(arrays, "train").y.shape == (5, 1)


def test_loader_requires_a_target():
    arrays = {"X_train": np.zeros((5, 4, 2), dtype=np.float32), "dt_train": np.zeros((5, 4), dtype=np.float32)}
    with pytest.raises(ValueError):  # neither y_reg nor y
        sequence_data_from_arrays(arrays, "train")


def test_loader_rejects_dt_shape_mismatch():
    arrays = {
        "X_train": np.zeros((3, 4, 2), dtype=np.float32),
        "y_reg_train": np.zeros((3, 1), dtype=np.float32),
        "dt_train": np.zeros((3, 5), dtype=np.float32),  # (3, 5) != (3, 4)
    }
    with pytest.raises(ValueError):
        sequence_data_from_arrays(arrays, "train")


def test_loader_rejects_negative_dt():
    dt = np.zeros((3, 4), dtype=np.float32)
    dt[:, 1] = -1.0
    arrays = {
        "X_train": np.zeros((3, 4, 2), dtype=np.float32),
        "y_reg_train": np.zeros((3, 1), dtype=np.float32),
        "dt_train": dt,
    }
    with pytest.raises(ValueError):
        sequence_data_from_arrays(arrays, "train")
