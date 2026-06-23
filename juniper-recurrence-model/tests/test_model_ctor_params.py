"""TST-9: the ``LMURegressor`` ctor params ``time_unit`` and ``random_seed``.

``time_unit`` is carried into the topology meta and must round-trip through the serializer;
``random_seed`` is **inert** for the closed-form linear readout (a deterministic lstsq solve)
but **does** seed the RFF projection (data-independent randomness — DP-3 P2a). Neither param
was exercised non-default before this (audit TST-9).
"""

from __future__ import annotations

import numpy as np

from juniper_recurrence_model import LMURegressor, LMUSerializer, RFFReadoutSpec


def _toy_3d(n: int = 32, n_steps: int = 8, n_features: int = 3, seed: int = 0):
    """A small (n, T, F) batch with strictly-positive irregular integer gaps and a target."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_steps, n_features))
    dt = np.zeros((n, n_steps))
    dt[:, 1:] = rng.integers(1, 4, size=(n, n_steps - 1)).astype(float)
    y = rng.normal(size=(n,))
    return X, dt, y


def test_time_unit_carried_into_topology_meta() -> None:
    X, dt, y = _toy_3d()
    model = LMURegressor(d=8, theta=5.0, time_unit="days")
    model.fit(X, y, dt=dt)
    assert model.describe_topology()["meta"]["time_unit"] == "days"


def test_time_unit_and_random_seed_round_trip_through_serializer(tmp_path) -> None:
    X, dt, y = _toy_3d()
    model = LMURegressor(d=8, theta=5.0, time_unit="calendar_days", random_seed=7)
    model.fit(X, y, dt=dt)
    path = tmp_path / "lmu.npz"
    LMUSerializer().save(model, path)
    loaded = LMUSerializer().load(path)
    assert loaded.time_unit == "calendar_days"
    assert loaded.random_seed == 7


def test_random_seed_is_inert_for_the_linear_readout() -> None:
    # The default linear readout is a closed-form lstsq solve — deterministic regardless of seed.
    X, dt, y = _toy_3d()
    m0 = LMURegressor(d=8, theta=5.0, random_seed=0)
    m0.fit(X, y, dt=dt)
    m1 = LMURegressor(d=8, theta=5.0, random_seed=12345)
    m1.fit(X, y, dt=dt)
    assert np.allclose(m0.predict(X, dt=dt), m1.predict(X, dt=dt))


def test_rff_readout_is_reproducible_for_a_fixed_seed() -> None:
    # Same seed -> identical RFF projection -> identical held-out predictions.
    X, dt, y = _toy_3d(n=64, n_features=2, seed=1)
    X_test, dt_test, _ = _toy_3d(n=16, n_features=2, seed=5)
    preds = []
    for _ in range(2):
        model = LMURegressor(d=8, theta=5.0, readout=RFFReadoutSpec(), random_seed=3)
        model.fit(X, y, dt=dt)
        preds.append(model.predict(X_test, dt=dt_test))
    assert np.allclose(preds[0], preds[1])


def test_rff_readout_differs_across_seeds() -> None:
    # Different seeds -> different RFF projection -> different held-out predictions
    # (random_seed is NOT globally inert; it drives the nonlinear readout's randomness).
    X, dt, y = _toy_3d(n=64, n_features=2, seed=1)
    X_test, dt_test, _ = _toy_3d(n=16, n_features=2, seed=5)
    m0 = LMURegressor(d=8, theta=5.0, readout=RFFReadoutSpec(), random_seed=0)
    m0.fit(X, y, dt=dt)
    m1 = LMURegressor(d=8, theta=5.0, readout=RFFReadoutSpec(), random_seed=999)
    m1.fit(X, y, dt=dt)
    assert not np.allclose(m0.predict(X_test, dt=dt_test), m1.predict(X_test, dt=dt_test))
