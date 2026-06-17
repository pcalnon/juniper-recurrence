"""Data path: juniper-data-client → 3-D ``equities_seq`` NPZ → model kwargs (plan §8).

Resolves a dataset reference to a ``dataset_id``, downloads the NPZ artifact, runs
juniper-data-client's full-contract validator **when the installed client exposes it**
(see the guarded import below), then maps it to the arrays ``LMURegressor`` consumes via
the model package's ``sequence_data_from_arrays`` (reusing the canonical WS-1 key layout
+ ``dt`` rules instead of re-deriving them — also the validation floor when the validator
is absent).

Framework-light by design: takes primitives (no FastAPI / pydantic / settings import),
so the routers and the headless CLI ``train`` share it. ``JuniperDataClient`` and
``validate_npz_contract`` are imported at module level so tests can monkeypatch them.
"""

from __future__ import annotations

from typing import Any

from juniper_data_client import JuniperDataClient
from juniper_recurrence_model import SequenceData, sequence_data_from_arrays

try:
    # ``validate_npz_contract`` landed in juniper-data-client AFTER the published 0.4.1
    # pin. When the installed client provides it, it runs as the authoritative
    # full-contract gate; otherwise the model-side ``sequence_data_from_arrays`` checks
    # (X is 3-D, the dt rules, a regression target present) are the validation floor.
    # Bump the ``juniper-data-client`` pin once the validator publishes to make it hard.
    from juniper_data_client import validate_npz_contract
except ImportError:  # pragma: no cover - depends on the installed juniper-data-client version
    validate_npz_contract = None

__all__ = ["load_sequence_data"]


def _resolve_dataset_id(
    client: JuniperDataClient,
    *,
    dataset_id: str | None,
    name: str | None,
    generator: str | None,
    params: dict[str, Any] | None,
) -> str:
    """Resolve a dataset reference to a concrete ``dataset_id``.

    Precedence: explicit ``dataset_id`` → latest version of ``name`` → create a new
    dataset from ``generator`` + ``params``.
    """
    if dataset_id:
        return dataset_id
    if name:
        latest = client.get_latest(name)
        resolved = latest.get("dataset_id")
        if not resolved:
            raise ValueError(f"no dataset_id in latest version of {name!r}")
        return resolved
    if generator:
        created = client.create_dataset(generator=generator, params=dict(params or {}), persist=True)
        resolved = created.get("dataset_id")
        if not resolved:
            raise ValueError(f"no dataset_id returned creating {generator!r} dataset")
        return resolved
    raise ValueError("dataset ref requires one of: dataset_id, name, generator")


def load_sequence_data(
    *,
    base_url: str,
    api_key: str | None = None,
    dataset_id: str | None = None,
    name: str | None = None,
    generator: str | None = None,
    params: dict[str, Any] | None = None,
    split: str = "train",
) -> tuple[SequenceData, dict[str, Any]]:
    """Fetch and map one split of a 3-D sequence dataset for the LMU regressor.

    Returns the :class:`SequenceData` (``X`` / ``y`` / ``dt`` / ``target_dt`` /
    ``seq_lengths``) plus a plain descriptor dict for ``DatasetDescriptor``.

    Raises:
        juniper_data_client.JuniperDataClientError: on upstream fetch failures.
        ValueError: when the artifact violates the contract or is not a 3-D sequence.
    """
    client = JuniperDataClient(base_url=base_url, api_key=api_key)
    try:
        resolved_id = _resolve_dataset_id(client, dataset_id=dataset_id, name=name, generator=generator, params=params)
        arrays = client.download_artifact_npz(resolved_id)
        if validate_npz_contract is not None:
            validate_npz_contract(arrays)  # full-contract gate when the client provides it (raises ValueError)
        sequence = sequence_data_from_arrays(arrays, split)
    finally:
        client.close()

    descriptor = {
        "dataset_id": resolved_id,
        "name": name,
        "split": split,
        "n_windows": int(sequence.X.shape[0]),
        "lookback": int(sequence.X.shape[1]),
        "n_features": int(sequence.X.shape[2]),
        "output_dim": int(sequence.y.shape[1]) if sequence.y.ndim > 1 else 1,
        "has_target_dt": sequence.target_dt is not None,
        "has_seq_lengths": sequence.seq_lengths is not None,
    }
    return sequence, descriptor
