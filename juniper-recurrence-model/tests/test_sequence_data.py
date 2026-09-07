"""Tests for the 3-D sequence NPZ loader + the end-to-end irregular-Δt consumer path (§9.1c).

Synthesises an ``equities_seq``-shaped 3-D artifact (per-split ``X`` / ``y_reg`` / ``dt`` /
``target_dt`` / ``seq_lengths``), loads it via :func:`load_sequence_npz`, and trains + predicts
:class:`LMURegressor` on it — the juniper-recurrence consumer ingesting the shipped WS-1 contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from juniper_recurrence_model import LMURegressor, SequenceData, load_sequence_npz, sequence_data_from_arrays
from juniper_recurrence_model.data import derive_full_split


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


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_sequence_data_rejects_nonfinite_dt(bad):
    """Non-finite dt must be rejected at ingestion (audit MODEL-01)."""
    arrays = _make_equities_seq_arrays(splits=("train",))
    arrays["dt_train"][0, 1] = bad
    with pytest.raises(ValueError, match="non-finite|finite"):
        sequence_data_from_arrays(arrays, split="train")


def test_sequence_data_rejects_nonfinite_features():
    """Non-finite feature values (X) must be rejected at ingestion."""
    arrays = _make_equities_seq_arrays(splits=("train",))
    arrays["X_train"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite|finite"):
        sequence_data_from_arrays(arrays, split="train")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("key", ["y_reg_train", "y_train"])
def test_sequence_data_rejects_nonfinite_target(bad, key):
    """The TARGET must be rejected too -- X and dt were guarded, y was not.

    That asymmetry is the defect: this guard exists to stop non-finite values
    reaching the model, and a NaN in the target produces a NaN loss on the first
    backward pass -- the very failure the X check prevents, arriving through the
    one door left open.

    Both target keys are covered because the loader prefers ``y_reg_*`` and falls
    back to ``y_*``; guarding only the preferred key would leave the fallback
    path unchecked.
    """
    arrays = _make_equities_seq_arrays(splits=("train",))
    if key == "y_train":
        # Exercise the FALLBACK branch for real rather than skipping it: the
        # loader prefers ``y_reg_*``, so the fallback is only reached once the
        # preferred key is gone. A skip here would have pinned nothing.
        arrays["y_train"] = np.asarray(arrays.pop("y_reg_train"), dtype=float).copy()
    arrays[key] = np.asarray(arrays[key], dtype=float).copy()
    arrays[key][0] = bad
    with pytest.raises(ValueError, match="non-finite|finite"):
        sequence_data_from_arrays(arrays, split="train")


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


# --------------------------------------------------------------------------------------
# Decision 11: juniper-data stopped emitting the *_full family (juniper-data#369), and
# POST /v1/crossval derives its walk-forward folds from it (D-CV-4). The loader now
# rebuilds it. The property that matters is ROW ORDER, not membership: crossval slices by
# row index, so a reconstruction holding the right rows in the wrong order silently changes
# which windows land in which fold.
# --------------------------------------------------------------------------------------


def _assemble_like_juniper_data(per_ticker, keys=("X", "y_reg", "dt", "ticker_code")):
    """Build an artifact exactly the way ``equities_seq._assemble`` does.

    Split arrays are SPLIT-major (every ticker's train, then every ticker's val, ...);
    ``_full`` is ENTITY-major (each ticker's train, val, test in turn). Same rows, different
    permutation -- reproducing that asymmetry is the entire point of this fixture.
    """
    splits = ("train", "val", "test")
    arrays = {}
    for split in splits:
        for key in keys:
            arrays[f"{key}_{split}"] = np.concatenate([t[split][key] for t in per_ticker], axis=0)
    for key in keys:
        blocks = [t[split][key] for t in per_ticker for split in splits]
        arrays[f"{key}_full"] = np.concatenate(blocks, axis=0)
    return arrays


def _ticker_windows(ticker_code, counts, lookback=4, n_features=2, start=0):
    """One ticker's per-split window blocks, with globally unique row values."""
    out, cursor = {}, start
    for split, n in zip(("train", "val", "test"), counts, strict=True):
        ids = np.arange(cursor, cursor + n, dtype=np.float32)
        out[split] = {
            # Every row carries its own globally unique id, so a permutation is visible
            # in the VALUES rather than only in the shapes.
            "X": np.broadcast_to(ids[:, None, None], (n, lookback, n_features)).astype(np.float32).copy(),
            "y_reg": ids[:, None].copy(),
            "dt": np.zeros((n, lookback), dtype=np.float32),
            "ticker_code": np.full(n, ticker_code, dtype=np.int64),
        }
        cursor += n
    return out


def _three_ticker_artifact():
    return _assemble_like_juniper_data(
        [
            _ticker_windows(10, (3, 2, 2), start=0),
            _ticker_windows(20, (4, 1, 2), start=100),
            _ticker_windows(30, (2, 2, 3), start=200),
        ]
    )


class TestDeriveFullSplit:
    """The derived ``*_full`` must equal the array juniper-data used to ship, row for row."""

    def test_multi_ticker_reconstruction_is_exact(self):
        artifact = _three_ticker_artifact()
        expected_X = artifact["X_full"].copy()
        expected_y = artifact["y_reg_full"].copy()

        post_369 = {k: v for k, v in artifact.items() if not k.endswith("_full")}
        derived = derive_full_split(post_369)

        assert np.array_equal(derived["X_full"], expected_X)
        assert np.array_equal(derived["y_reg_full"], expected_y)

    def test_a_plain_concatenation_would_not_have_worked(self):
        """Prove the stable sort is load-bearing rather than incidental.

        If ``concat(train, val, test)`` already equalled ``X_full``, the reordering above
        would be untested ceremony and could be deleted without a failure. It does not.
        """
        artifact = _three_ticker_artifact()
        naive = np.concatenate([artifact["X_train"], artifact["X_val"], artifact["X_test"]], axis=0)

        assert naive.shape == artifact["X_full"].shape
        assert not np.array_equal(naive, artifact["X_full"]), "fixture is not multi-ticker enough to exercise the reordering"
        assert sorted(naive[:, 0, 0].tolist()) == sorted(artifact["X_full"][:, 0, 0].tolist()), "same rows, different order"

    def test_a_producer_full_array_is_never_overwritten(self):
        """A legacy artifact keeps the producer's own arrays, byte for byte."""
        artifact = _three_ticker_artifact()
        sentinel = np.full_like(artifact["X_full"], -7.0)
        artifact["X_full"] = sentinel

        derived = derive_full_split(artifact)
        assert np.array_equal(derived["X_full"], sentinel)

    def test_single_ticker_is_a_plain_concatenation(self):
        artifact = _assemble_like_juniper_data([_ticker_windows(10, (4, 2, 3))])
        post_369 = {k: v for k, v in artifact.items() if not k.endswith("_full")}

        derived = derive_full_split(post_369)
        assert np.array_equal(derived["X_full"], artifact["X_full"])
        assert np.array_equal(derived["X_full"], np.concatenate([artifact["X_train"], artifact["X_val"], artifact["X_test"]], axis=0))

    def test_an_artifact_without_ticker_codes_still_derives(self):
        artifact = _assemble_like_juniper_data([_ticker_windows(10, (3, 2, 2))], keys=("X", "y_reg", "dt"))
        post_369 = {k: v for k, v in artifact.items() if not k.endswith("_full")}

        derived = derive_full_split(post_369)
        assert np.array_equal(derived["X_full"], artifact["X_full"])

    def test_a_legacy_two_way_artifact_derives_without_val(self):
        """No val partition: the whole set is train | test, as the old two-way ``_full`` was."""
        arrays = _make_equities_seq_arrays(splits=("train", "test"))
        derived = derive_full_split(arrays)
        assert derived["X_full"].shape[0] == arrays["X_train"].shape[0] + arrays["X_test"].shape[0]


class TestCrossvalReadSurvivesDecision11:
    """``POST /v1/crossval`` passes ``split="full"``. That read must not fail post-#369."""

    def test_full_split_loads_from_an_artifact_that_has_no_full_family(self):
        arrays = _make_equities_seq_arrays(splits=("train", "val", "test"))
        assert "X_full" not in arrays  # the post-#369 shape

        data = sequence_data_from_arrays(arrays, "full")
        assert data.X.shape[0] == sum(arrays[f"X_{s}"].shape[0] for s in ("train", "val", "test"))
        assert data.dt.shape[0] == data.X.shape[0]
        assert data.y.shape[0] == data.X.shape[0]

    def test_full_split_still_loads_from_a_legacy_artifact(self):
        arrays = _three_ticker_artifact()
        data = sequence_data_from_arrays(arrays, "full")
        assert np.array_equal(data.X, arrays["X_full"])

    def test_a_missing_partition_still_raises_rather_than_deriving_a_partial_set(self):
        """The fallback must not paper over a genuinely broken artifact."""
        with pytest.raises(ValueError, match="X_full"):
            sequence_data_from_arrays({"y_reg_train": np.zeros((2, 1), dtype=np.float32)}, "full")
