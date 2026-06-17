"""Unit tests for the data adapter ``juniper_recurrence.data`` (plan §8 / §12).

The juniper-data-client is faked (``fake_data`` fixture or inline) so these exercise
the real array mapping (``sequence_data_from_arrays``) and the dataset-id resolution
precedence without a live juniper-data service.
"""

from __future__ import annotations

import pytest

from juniper_recurrence.data import load_sequence_data


def test_maps_arrays_to_sequence(fake_data):
    sequence, descriptor = load_sequence_data(base_url="http://data", dataset_id="ds-1")

    assert sequence.X.shape == (12, 5, 2)
    assert sequence.y.shape == (12, 1)
    assert sequence.dt.shape == (12, 5)
    assert sequence.target_dt is not None and sequence.target_dt.shape == (12,)
    assert sequence.seq_lengths is not None and sequence.seq_lengths.shape == (12,)
    assert set(sequence.fit_kwargs()) == {"dt", "target_dt", "seq_lengths"}

    assert descriptor == {
        "dataset_id": "ds-1",
        "name": None,
        "split": "train",
        "n_windows": 12,
        "lookback": 5,
        "n_features": 2,
        "output_dim": 1,
        "has_target_dt": True,
        "has_seq_lengths": True,
    }


def test_resolve_by_name(fake_data):
    _, descriptor = load_sequence_data(base_url="http://data", name="equities")
    assert descriptor["dataset_id"] == "latest-of-equities"
    assert descriptor["name"] == "equities"


def test_resolve_by_generator(fake_data):
    _, descriptor = load_sequence_data(base_url="http://data", generator="equities_seq")
    assert descriptor["dataset_id"] == "created-1"


def test_requires_a_ref(fake_data):
    with pytest.raises(ValueError, match="requires one of"):
        load_sequence_data(base_url="http://data")


def test_bad_contract_raises(fake_data, monkeypatch):
    def _raise(arrays, **kw):
        raise ValueError("not a 3-D sequence")

    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", _raise)
    with pytest.raises(ValueError, match="sequence"):
        load_sequence_data(base_url="http://data", dataset_id="bad")


def test_resolve_by_name_missing_id(monkeypatch, synthetic_npz_arrays):
    class _Client:
        def __init__(self, **kwargs):
            pass

        def get_latest(self, name):
            return {}  # no dataset_id

        def download_artifact_npz(self, dataset_id):
            return synthetic_npz_arrays

        def close(self):
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _Client)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda a, **kw: "sequence")
    with pytest.raises(ValueError, match="no dataset_id"):
        load_sequence_data(base_url="http://data", name="nope")


def test_resolve_by_generator_missing_id(monkeypatch, synthetic_npz_arrays):
    class _Client:
        def __init__(self, **kwargs):
            pass

        def create_dataset(self, **kwargs):
            return {}  # no dataset_id

        def download_artifact_npz(self, dataset_id):
            return synthetic_npz_arrays

        def close(self):
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _Client)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda a, **kw: "sequence")
    with pytest.raises(ValueError, match="no dataset_id"):
        load_sequence_data(base_url="http://data", generator="g")
