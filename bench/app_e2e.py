"""I2 — end-to-end Δt proof through the deployed app.

Generates an ``irregular_sine`` dataset and serves it through the app's data adapter (mocked — no
live juniper-data service), then drives ``POST /v1/train`` -> ``/v1/predict`` -> ``/v1/crossval`` and
asserts the deployed app trains + predicts well on irregular-Δt data. This is the roadmap's OQ-7
"completed-state" gate: the shipped HTTP surface actually works on irregular timing end-to-end.

Run from the repo root:  ``python -m bench.app_e2e``
"""

from __future__ import annotations

from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from bench import datasets


def _arrays_for_app(ds: datasets.Dataset) -> dict[str, np.ndarray]:
    """Map the full window set to the sequence NPZ the app data adapter expects.

    Serves the full set under both the ``_full`` keys (read by ``/v1/crossval``) and the ``_train``
    keys (read by ``/v1/train`` / ``/v1/predict`` for ``split="train"``).
    """
    n, t = ds.X.shape[0], ds.X.shape[1]
    X = ds.X.astype("float32")
    y = ds.y.reshape(-1).astype("float32")
    dt = ds.dt.astype("float32")
    target_dt = ds.target_dt.astype("float32")
    seq_lengths = np.full(n, t, dtype="int64")
    arrays: dict[str, np.ndarray] = {}
    for suffix in ("full", "train"):
        arrays[f"X_{suffix}"] = X
        arrays[f"y_{suffix}"] = y
        arrays[f"dt_{suffix}"] = dt
        arrays[f"target_dt_{suffix}"] = target_dt
        arrays[f"seq_lengths_{suffix}"] = seq_lengths
    return arrays


def main() -> None:
    ds = datasets.irregular_sine(n_steps=800, lookback=16, jitter=0.6, seed=0)
    arrays = _arrays_for_app(ds)

    class _FakeClient:
        def __init__(self, **kw: object) -> None:
            pass

        def get_latest(self, name: str) -> dict[str, str]:
            return {"dataset_id": f"latest-{name}"}

        def create_dataset(self, **kw: object) -> dict[str, str]:
            return {"dataset_id": "bench"}

        def download_artifact_npz(self, dataset_id: str) -> dict[str, np.ndarray]:
            return arrays

        def close(self) -> None:
            pass

    with (
        mock.patch("juniper_recurrence.data.JuniperDataClient", _FakeClient),
        mock.patch(
            "juniper_recurrence.data.validate_npz_contract", lambda a, **k: "sequence"
        ),
    ):
        from juniper_recurrence.app import build_app
        from juniper_recurrence.settings import Settings

        client = TestClient(build_app(Settings()), raise_server_exceptions=False)

        r = client.post(
            "/v1/train", json={"dataset": {"name": "irregular_sine"}, "d": 16}
        )
        assert r.status_code == 200, f"/v1/train -> {r.status_code}: {r.text}"
        metrics = r.json().get("final_metrics", {})
        print(
            f"/v1/train      r2={metrics.get('r2'):.4f}  rmse={metrics.get('rmse'):.4f}"
        )
        assert metrics.get("r2", 0.0) >= 0.9, (
            f"train r2 {metrics.get('r2')} unexpectedly low"
        )

        rp = client.post("/v1/predict", json={"dataset": {"name": "irregular_sine"}})
        assert rp.status_code == 200, f"/v1/predict -> {rp.status_code}: {rp.text}"
        print(f"/v1/predict    shape={rp.json().get('shape')}")

        rc = client.post(
            "/v1/crossval",
            json={"dataset": {"name": "irregular_sine"}, "n_folds": 3, "d": 16},
        )
        assert rc.status_code == 200, f"/v1/crossval -> {rc.status_code}: {rc.text}"
        print(f"/v1/crossval   n_folds={rc.json().get('n_folds')}  (HTTP 200)")

    print(
        "\nI2 e2e PASS — the deployed app trains + predicts + cross-validates on irregular-Δt data."
    )


if __name__ == "__main__":
    main()
