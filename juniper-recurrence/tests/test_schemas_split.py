"""Schema tests for the ``split`` field of ``DatasetRef`` (decision 11 straggler sweep).

``DatasetRef.split`` was a bare ``str``, so a misspelled split was accepted at the HTTP edge
and only failed much later — deep inside the model loader, as a ``ValueError`` on a missing
NPZ key. That is a 500-shaped failure for what is simply a malformed request. It is now
``Literal["train", "val", "test", "full"]``, so the rejection happens at the request boundary.

These pin both halves: the four names the NPZ contract actually defines are accepted (decision
11, juniper-ml ``notes/JUNIPER_2026-08-29_JUNIPER-ECOSYSTEM_TRAIN-EVAL-TEST-PARTITION-DESIGN.md``
§9.5), and anything else is a ``ValidationError`` / 422 rather than a deferred crash.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from juniper_recurrence.app import build_app
from juniper_recurrence.schemas import CrossValRequest, DatasetRef
from juniper_recurrence.settings import Settings


def test_split_defaults_to_train():
    """The default is unchanged by the narrowing — a ref that omits ``split`` still means train."""
    assert DatasetRef(dataset_id="x").split == "train"


@pytest.mark.parametrize("split", ["train", "val", "test", "full"])
def test_the_four_contract_splits_validate(split):
    """All four names the contract defines are accepted, not just the two that used to be documented.

    ``val`` is the in-loop partition decision 11 made first-class; ``full`` is the whole dataset,
    served from the artifact's own ``*_full`` arrays when present and otherwise rebuilt by
    ``juniper_recurrence_model.data.derive_full_split``.
    """
    assert DatasetRef(dataset_id="x", split=split).split == split


@pytest.mark.parametrize("split", ["bogus", "eval", "Train", "valid", "", "full "])
def test_a_split_outside_the_contract_is_rejected(split):
    """Rejected at construction — including the near-misses a caller is actually likely to send.

    ``eval`` / ``valid`` are the two wrong names for ``val`` (the ecosystem settled on ``X_val``),
    ``Train`` is the case error, and ``"full "`` the stray-whitespace one. A bare ``str`` accepted
    every one of these.
    """
    with pytest.raises(ValidationError):
        DatasetRef(dataset_id="x", split=split)


def test_the_narrowing_reaches_nested_refs():
    """``CrossValRequest`` embeds a ``DatasetRef``, so it inherits the constraint.

    The crossval router overrides the split with ``"full"`` regardless (D-CV-4), but the field is
    still part of the request body it accepts, and a body is either valid or it is not.
    """
    assert CrossValRequest(dataset={"dataset_id": "x", "split": "val"}, n_folds=2).dataset.split == "val"
    with pytest.raises(ValidationError):
        CrossValRequest(dataset={"dataset_id": "x", "split": "bogus"}, n_folds=2)


def test_a_bad_split_is_a_422_at_the_request_boundary():
    """The point of the change: the HTTP surface rejects it, rather than a later ``ValueError``.

    No data fixture is used deliberately — if this returned anything other than 422 the request
    would have reached the data adapter, which is the behaviour being ruled out.
    """
    client = TestClient(build_app(Settings(api_keys=None)))
    assert client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1", "split": "bogus"}}).status_code == 422
    assert client.post("/v1/predict", json={"dataset": {"dataset_id": "ds-1", "split": "bogus"}}).status_code == 422
    assert client.post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1", "split": "bogus"}, "n_folds": 2}).status_code == 422


def test_the_openapi_schema_advertises_the_four_splits():
    """The narrowing is also a documentation fix: the four names now appear in the published schema."""
    schema = TestClient(build_app(Settings(api_keys=None))).app.openapi()
    assert schema["components"]["schemas"]["DatasetRef"]["properties"]["split"]["enum"] == ["train", "val", "test", "full"]
