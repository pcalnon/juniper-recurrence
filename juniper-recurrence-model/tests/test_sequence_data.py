"""Tests for the 3-D sequence NPZ loader + the end-to-end irregular-Δt consumer path (§9.1c).

Synthesises an ``equities_seq``-shaped 3-D artifact (per-split ``X`` / ``y_reg`` / ``dt`` /
``target_dt`` / ``seq_lengths``), loads it via :func:`load_sequence_npz`, and trains + predicts
:class:`LMURegressor` on it — the juniper-recurrence consumer ingesting the shipped WS-1 contract.
Also pins the explicit ``target=`` selection (W1.3) and the reader's mirror of the validator's
``target_dt`` / ``seq_lengths`` rules (W1.4).
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pytest

from juniper_recurrence_model import LMURegressor, SequenceData, load_sequence_npz, sequence_data_from_arrays
from juniper_recurrence_model.data import DEFAULT_TARGET, TARGET_MODES, derive_full_split


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

    Both target keys are covered because ``target="auto"`` prefers ``y_reg_*`` and
    falls back to ``y_*``; guarding only the preferred key would leave the fallback
    path unchecked. (``target="class"`` reads ``y_*`` too -- pinned separately in
    ``TestTargetSelection``.)
    """
    arrays = _make_equities_seq_arrays(splits=("train",))
    target = "reg"
    if key == "y_train":
        # Exercise the FALLBACK branch for real rather than skipping it: the
        # loader prefers ``y_reg_*``, so the fallback is only reached once the
        # preferred key is gone -- and, since W1.3, only under ``target="auto"``
        # (the default refuses a ``y_*``-only artifact outright). A skip here
        # would have pinned nothing.
        arrays["y_train"] = np.asarray(arrays.pop("y_reg_train"), dtype=float).copy()
        target = "auto"
    arrays[key] = np.asarray(arrays[key], dtype=float).copy()
    arrays[key][0] = bad
    with pytest.raises(ValueError, match="non-finite|finite"):
        sequence_data_from_arrays(arrays, split="train", target=target)


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
    """The fallback survives W1.3 as an explicit opt-in (``target="auto"``); the default refuses it (``TestTargetSelection``)."""
    rng = np.random.default_rng(2)
    arrays = {
        "X_train": rng.normal(size=(5, 4, 2)).astype(np.float32),
        "y_train": rng.normal(size=(5, 1)).astype(np.float32),  # no y_reg
        "dt_train": np.zeros((5, 4), dtype=np.float32),
    }
    assert sequence_data_from_arrays(arrays, "train", target="auto").y.shape == (5, 1)


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


# --------------------------------------------------------------------------------------
# W1.3 (findings F-S2 / F-S8 of juniper-ml
# notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md):
# the target is SELECTED, not silently fallen back to. Before W1.3 an artifact without
# y_reg_{split} had its y_{split} read instead -- on a classification artifact that is the
# one-hot direction label, so a regression run became a two-output direction fit and nothing
# said so. ``reg`` now refuses, ``class`` asks for the label on purpose, and ``auto`` keeps
# the old preference order but logs the fallback.
# --------------------------------------------------------------------------------------

_DATA_LOGGER = "juniper_recurrence_model.data"


def _y_only_artifact(n=6, lookback=4, n_features=2):
    """A three-partition artifact carrying only the one-hot ``y_*`` label -- no ``y_reg_*`` (the F-S2 shape)."""
    rng = np.random.default_rng(7)
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[f"X_{split}"] = rng.normal(size=(n, lookback, n_features)).astype(np.float32)
        arrays[f"y_{split}"] = np.eye(2, dtype=np.float32)[rng.integers(0, 2, size=n)]
        arrays[f"dt_{split}"] = np.zeros((n, lookback), dtype=np.float32)
    return arrays


def _equities_seq_artifact(tickers=(3, 7), per_ticker=(5, 2, 2), lookback=64, n_features=15, seed=11):
    """A full ``equities_seq``-shaped three-partition artifact (the plan's §3.4 key inventory).

    Per split: ``X (W, 64, 15) f32``, one-hot ``y (W, 2) f32``, ``y_reg (W, 1) f32``,
    ``date (W, 64) i32``, ``dt (W, 64) f32`` with ``dt[:, 0] == 0``, ``target_dt (W,) f32``,
    ``window_end_date (W,) i32``, ``ticker_code (W,) i32`` and ``observed_mask (W, 64) u8``, plus
    the unsuffixed ``ticker_vocab (E,)`` string array. No ``seq_lengths`` / ``t`` /
    ``padding_mask`` / ``*_full`` -- the producer emits none of them. Split-major across
    tickers, the way juniper-data lays the partitions down.
    """
    rng = np.random.default_rng(seed)
    arrays = {}
    for split, n_per_ticker in zip(("train", "val", "test"), per_ticker, strict=True):
        n = n_per_ticker * len(tickers)
        dt = np.zeros((n, lookback), dtype=np.float32)
        dt[:, 1:] = rng.integers(1, 4, size=(n, lookback - 1)).astype(np.float32)  # calendar-day gaps
        date = (730_000 + np.cumsum(dt, axis=1)).astype(np.int32)
        arrays[f"X_{split}"] = rng.normal(size=(n, lookback, n_features)).astype(np.float32)
        arrays[f"y_{split}"] = np.eye(2, dtype=np.float32)[rng.integers(0, 2, size=n)]
        arrays[f"y_reg_{split}"] = rng.normal(scale=0.01, size=(n, 1)).astype(np.float32)
        arrays[f"date_{split}"] = date
        arrays[f"dt_{split}"] = dt
        arrays[f"target_dt_{split}"] = rng.integers(1, 4, size=n).astype(np.float32)
        arrays[f"window_end_date_{split}"] = date[:, -1].copy()
        arrays[f"ticker_code_{split}"] = np.repeat(np.asarray(tickers, dtype=np.int32), n_per_ticker)
        arrays[f"observed_mask_{split}"] = np.ones((n, lookback), dtype=np.uint8)
    arrays["ticker_vocab"] = np.array([f"T{code}" for code in tickers], dtype=np.str_)
    return arrays


class TestTargetSelection:
    """``target=`` names the key the regressor fits; nothing falls back silently any more."""

    @pytest.mark.parametrize("split", ["train", "val", "test", "full"])
    def test_reg_on_a_y_only_artifact_raises_the_exact_message(self, split):
        with pytest.raises(ValueError) as excinfo:
            sequence_data_from_arrays(_y_only_artifact(), split, target="reg")
        assert str(excinfo.value) == f"regression target 'y_reg_{split}' missing"

    def test_every_public_entry_defaults_to_the_one_named_constant(self):
        """A re-ruling of R8 is one line (``DEFAULT_TARGET``) only while no entry hard-codes its own default."""
        for entry in (sequence_data_from_arrays, load_sequence_npz):
            parameter = inspect.signature(entry).parameters["target"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, entry.__name__
            assert parameter.default == DEFAULT_TARGET, entry.__name__

    def test_the_default_applies_the_recommended_r8_ruling(self):
        """Pins plan ruling R8's recommendation (``reg``), pending the owner's ruling -- a re-ruling moves this pin with the constant."""
        assert DEFAULT_TARGET == "reg"
        with pytest.raises(ValueError, match=r"^regression target 'y_reg_train' missing$"):
            sequence_data_from_arrays(_y_only_artifact(), "train")

    def test_auto_falls_back_to_y_with_a_warning_naming_the_split_and_both_keys(self, caplog):
        arrays = _y_only_artifact()
        with caplog.at_level(logging.WARNING, logger=_DATA_LOGGER):
            data = sequence_data_from_arrays(arrays, "train", target="auto")

        assert np.array_equal(data.y, arrays["y_train"])
        records = [r for r in caplog.records if r.name == _DATA_LOGGER]
        assert [r.levelno for r in records] == [logging.WARNING]
        message = records[0].getMessage()
        for fragment in ("'train'", "'y_reg_train'", "'y_train'"):
            assert fragment in message, f"{fragment} missing from {message!r}"

    def test_auto_prefers_y_reg_and_stays_silent(self, caplog):
        arrays = _equities_seq_artifact()
        with caplog.at_level(logging.WARNING, logger=_DATA_LOGGER):
            data = sequence_data_from_arrays(arrays, "train", target="auto")

        assert np.array_equal(data.y, arrays["y_reg_train"])
        assert not [r for r in caplog.records if r.name == _DATA_LOGGER]

    def test_auto_with_neither_key_still_raises(self):
        arrays = _y_only_artifact()
        del arrays["y_train"]
        with pytest.raises(ValueError, match=r"neither 'y_reg_train' nor 'y_train' present"):
            sequence_data_from_arrays(arrays, "train", target="auto")

    def test_class_on_a_y_reg_only_artifact_raises(self):
        arrays = _make_equities_seq_arrays(splits=("train",))
        assert "y_train" not in arrays and "y_reg_train" in arrays
        with pytest.raises(ValueError) as excinfo:
            sequence_data_from_arrays(arrays, "train", target="class")
        assert str(excinfo.value) == "classification target 'y_train' missing"

    def test_class_reads_the_one_hot_label_even_when_y_reg_is_present(self):
        arrays = _equities_seq_artifact()
        data = sequence_data_from_arrays(arrays, "train", target="class")
        assert data.y.shape == (arrays["X_train"].shape[0], 2)
        assert np.array_equal(data.y, arrays["y_train"])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_the_class_target_is_finiteness_checked_too(self, bad):
        arrays = _equities_seq_artifact()
        arrays["y_train"] = arrays["y_train"].copy()
        arrays["y_train"][0, 0] = bad
        with pytest.raises(ValueError, match="non-finite"):
            sequence_data_from_arrays(arrays, "train", target="class")

    @pytest.mark.parametrize(("mode", "width"), [("reg", 1), ("class", 2), ("auto", 1)])
    def test_every_mode_loads_an_artifact_carrying_both_keys(self, mode, width):
        assert mode in TARGET_MODES
        data = sequence_data_from_arrays(_equities_seq_artifact(), "train", target=mode)
        assert data.y.shape[1] == width

    @pytest.mark.parametrize("split", ["train", "val", "test", "full"])
    def test_reg_loads_a_full_equities_shaped_three_partition_artifact(self, split, tmp_path):
        """The acceptance shape, read back from disk (``ticker_vocab`` is a ``<U`` array and ``allow_pickle=False``)."""
        arrays = _equities_seq_artifact()
        path = tmp_path / "equities_seq.npz"
        np.savez(path, **arrays)

        data = load_sequence_npz(path, split, target="reg")

        expected_y = derive_full_split(arrays)["y_reg_full"] if split == "full" else arrays[f"y_reg_{split}"]
        n = expected_y.shape[0]
        assert data.X.shape == (n, 64, 15) and data.X.dtype == np.float32
        assert data.y.shape == (n, 1)  # y_reg -- not the (W, 2) one-hot
        assert np.array_equal(data.y, expected_y)
        assert data.dt.shape == (n, 64) and np.all(data.dt[:, 0] == 0)
        assert data.target_dt.shape == (n,)
        assert data.seq_lengths is None  # the producer emits none

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match=r"target must be one of \('reg', 'class', 'auto'\); got 'regression'"):
            sequence_data_from_arrays(_equities_seq_artifact(), "train", target="regression")

    @pytest.mark.parametrize("entry", [sequence_data_from_arrays, load_sequence_npz], ids=lambda entry: entry.__name__)
    def test_target_cannot_be_passed_positionally(self, entry):
        """``Signature.bind`` applies the interpreter's own argument-binding rules without making the call.

        Binding rather than calling: a literal over-long call is what CodeQL's
        ``py/call/wrong-arguments`` reports, even one written to prove Python refuses it.
        """
        with pytest.raises(TypeError, match="too many positional arguments"):
            inspect.signature(entry).bind({}, "train", "auto")

    @pytest.mark.parametrize("split", ["validation", "Train", "", "train\nforged log line"])
    def test_an_unknown_split_is_refused_before_anything_is_read(self, split):
        """The app's ``SplitName`` admits four splits; the reader refuses anything else up front, not as a missing key."""
        with pytest.raises(ValueError, match=r"^split must be one of 'train' / 'val' / 'test' / 'full'; got "):
            sequence_data_from_arrays(_equities_seq_artifact(), split)

    def test_load_sequence_npz_threads_target_to_the_reader(self, tmp_path, caplog):
        path = tmp_path / "y_only.npz"
        np.savez(path, **_y_only_artifact())

        with pytest.raises(ValueError, match=r"^regression target 'y_reg_train' missing$"):
            load_sequence_npz(path, "train", target="reg")
        assert load_sequence_npz(path, "train", target="class").y.shape == (6, 2)
        with caplog.at_level(logging.WARNING, logger=_DATA_LOGGER):
            assert load_sequence_npz(path, "train", target="auto").y.shape == (6, 2)
        assert [r.levelno for r in caplog.records if r.name == _DATA_LOGGER] == [logging.WARNING]


# --------------------------------------------------------------------------------------
# W1.4 (finding F-S3): the reader mirrors juniper-data-client's validate_npz_contract timing
# rules -- finite dt; target_dt (W,), finite, >= 0; seq_lengths (W,), integer, in [1, L].
# One rejection per rule. The dt-key finiteness rule predates W1.4 (MODEL-01) and is pinned
# by test_sequence_data_rejects_nonfinite_dt; the t-derived path is pinned here.
# --------------------------------------------------------------------------------------


class TestTimingMirror:
    """The model-side timing rules a direct ``load_sequence_npz`` caller gets without the app's validator."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_dt_derived_from_a_non_finite_t_is_refused(self, bad):
        t = np.cumsum(np.ones((4, 5)), axis=1)
        t[1, 3] = bad
        arrays = {"X_train": np.zeros((4, 5, 2), dtype=np.float32), "y_reg_train": np.zeros((4, 1), dtype=np.float32), "t_train": t}
        with pytest.raises(ValueError, match=r"dt_train has non-finite gaps"):
            sequence_data_from_arrays(arrays, "train")

    @pytest.mark.parametrize("shape", [(40, 1), (39,), (41,)])
    def test_target_dt_must_be_one_horizon_per_window(self, shape):
        """``(W, 1)`` used to be reshaped into place silently; the contract says ``(W,)``."""
        arrays = _make_equities_seq_arrays(splits=("train",))
        arrays["target_dt_train"] = np.ones(shape, dtype=np.float32)
        with pytest.raises(ValueError, match=r"target_dt_train shape"):
            sequence_data_from_arrays(arrays, "train")

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_target_dt_must_be_finite(self, bad):
        arrays = _make_equities_seq_arrays(splits=("train",))
        arrays["target_dt_train"][2] = bad
        with pytest.raises(ValueError, match=r"target_dt_train has non-finite horizons"):
            sequence_data_from_arrays(arrays, "train")

    def test_target_dt_must_not_be_negative(self):
        arrays = _make_equities_seq_arrays(splits=("train",))
        arrays["target_dt_train"][0] = -1.0
        with pytest.raises(ValueError, match=r"target_dt_train has negative horizons"):
            sequence_data_from_arrays(arrays, "train")

    def test_a_zero_target_dt_is_admitted(self):
        arrays = _make_equities_seq_arrays(splits=("train",))
        arrays["target_dt_train"][:] = 0.0
        assert np.all(sequence_data_from_arrays(arrays, "train").target_dt == 0.0)

    @pytest.mark.parametrize("shape", [(40, 1), (39,)])
    def test_seq_lengths_must_be_one_per_window(self, shape):
        arrays = _make_equities_seq_arrays(splits=("train",))
        arrays["seq_lengths_train"] = np.ones(shape, dtype=np.int64)
        with pytest.raises(ValueError, match=r"seq_lengths_train shape"):
            sequence_data_from_arrays(arrays, "train")

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.bool_])
    def test_seq_lengths_must_have_an_integer_dtype(self, dtype):
        """Every value here is in range; only the dtype is wrong (the model would truncate a float silently)."""
        arrays = _make_equities_seq_arrays(splits=("train",))
        arrays["seq_lengths_train"] = np.ones(40, dtype=dtype)
        with pytest.raises(ValueError, match=r"seq_lengths_train must be an integer dtype"):
            sequence_data_from_arrays(arrays, "train")

    @pytest.mark.parametrize("bad", [0, -1, 13])
    def test_seq_lengths_must_lie_in_one_to_lookback(self, bad):
        arrays = _make_equities_seq_arrays(splits=("train",))  # lookback 12
        arrays["seq_lengths_train"][3] = bad
        with pytest.raises(ValueError, match=r"seq_lengths_train values must be in \[1, 12\]"):
            sequence_data_from_arrays(arrays, "train")

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
    def test_the_seq_lengths_boundaries_and_integer_widths_are_admitted(self, dtype):
        arrays = _make_equities_seq_arrays(splits=("train",))
        arrays["seq_lengths_train"] = np.where(np.arange(40) % 2 == 0, 1, 12).astype(dtype)
        assert np.array_equal(sequence_data_from_arrays(arrays, "train").seq_lengths, arrays["seq_lengths_train"])
