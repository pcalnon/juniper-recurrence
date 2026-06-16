"""Shared pytest fixtures for the juniper-recurrence app test suite.

* ``_clean_juniper_env`` (autouse) clears ambient ``JUNIPER_*`` env vars so every test
  sees the documented defaults regardless of the developer's shell or a stray ``.env``.
* ``synthetic_npz_arrays`` builds a minimal, valid 3-D sequence NPZ array mapping.
* ``fake_data`` patches the data adapter (``JuniperDataClient`` + ``validate_npz_contract``)
  to serve those arrays, so route/CLI tests exercise the real mapping + model + lifecycle
  without a live juniper-data service.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clean_juniper_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("JUNIPER_"):
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def synthetic_npz_arrays() -> dict[str, np.ndarray]:
    """A deterministic, contract-valid 3-D sequence NPZ mapping (train split).

    Shapes: ``X (12, 5, 2)``, ``y_reg (12,)``, ``dt (12, 5)`` with ``dt[:, 0] == 0``,
    ``target_dt (12,)``, ``seq_lengths (12,)``.
    """
    rng = np.random.default_rng(0)
    n_windows, lookback, n_features = 12, 5, 2
    X = rng.standard_normal((n_windows, lookback, n_features)).astype("float32")
    dt = np.zeros((n_windows, lookback), dtype="float32")
    dt[:, 1:] = rng.uniform(0.5, 2.0, size=(n_windows, lookback - 1)).astype("float32")
    y = rng.standard_normal((n_windows,)).astype("float32")
    target_dt = rng.uniform(0.5, 2.0, size=(n_windows,)).astype("float32")
    seq_lengths = np.full((n_windows,), lookback, dtype="int64")
    return {
        "X_train": X,
        "y_reg_train": y,
        "dt_train": dt,
        "target_dt_train": target_dt,
        "seq_lengths_train": seq_lengths,
    }


@pytest.fixture
def fake_data(monkeypatch, synthetic_npz_arrays) -> dict[str, np.ndarray]:
    """Patch the data adapter to serve ``synthetic_npz_arrays`` (no live juniper-data)."""

    class _FakeDataClient:
        def __init__(self, **kwargs) -> None:
            self.closed = False

        def get_latest(self, name):
            return {"dataset_id": f"latest-of-{name}"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "created-1"}

        def download_artifact_npz(self, dataset_id):
            return synthetic_npz_arrays

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _FakeDataClient)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda arrays, **kw: "sequence")
    return synthetic_npz_arrays
