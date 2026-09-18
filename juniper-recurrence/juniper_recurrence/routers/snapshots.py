"""Model-snapshot routes: save / list / get / restore an LMU to and from disk.

Design of record: juniper-ml
``notes/JUNIPER_2026-09-16_JUNIPER-RECURRENCE_MODEL-PERSISTENCE-DESIGN.md`` (§7 surface, §11
rulings). This closes Y2 of the canopy selection-reachability arc, where canopy's snapshot
workflow reported success at BOTH ends while never writing or reading model state.

Three things this module deliberately does **not** do:

* **It does not serialise anything itself.** ``LMUSerializer`` already round-trips an
  ``LMURegressor`` losslessly (versioned ``schema: 2``, ``allow_pickle=False``, memory
  eigendecomposition recomputed from ``d``/θ on load). This is HTTP surface over that.
* **It does not prune.** Retention is the ecosystem's, inherited not invented: §6.4 of
  ``JUNIPER_2026-08-16_JUNIPER-ECOSYSTEM_SNAPSHOT-LIFECYCLE-MANAGEMENT-DESIGN.md`` was ratified
  no-deletion (juniper-ml#1296), and says do not build deletion tooling.
* **It does not synthesise a ``TrainResult`` on restore.** A restored model reports the third
  state ``restored`` with no result and no events — see ``AppState.set_restored``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from juniper_recurrence_model import LMURegressor
from juniper_recurrence_model.model import LMUSerializer

from juniper_recurrence.routers._common import get_settings, get_state
from juniper_recurrence.schemas import RestoreResponse, SnapshotListResponse, SnapshotModel, SnapshotRequest
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState

router = APIRouter(tags=["snapshots"])

#: Shape of a snapshot id. Ids are generated here and never supplied by a client, but the get /
#: restore paths take one from the URL, so this rejects input that could not name a snapshot
#: under any circumstances — cheaply, and before touching the filesystem.
#:
#: It is NOT what keeps a caller out of the filesystem: ``_snapshot_path`` never builds a path
#: from the caller's string at all. Kept anyway as a fast reject and because an anchored
#: character class documents the id format in one place.
_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")

#: The serializer appends this when absent; ids are stored WITHOUT it so the two halves agree.
_SUFFIX = ".npz"


def _safe_id(snapshot_id: str) -> str:
    """Return ``snapshot_id`` if it is a legal id, else raise 404.

    404 rather than 422 on purpose: an id that cannot name a file cannot name a snapshot, and
    distinguishing "malformed" from "absent" here would tell a caller which ids exist.
    """
    if not _ID_RE.match(snapshot_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such snapshot: {snapshot_id}")
    return snapshot_id


def _dir(settings: Settings) -> Path:
    """Resolve (and create) the snapshots directory."""
    path = Path(settings.snapshots_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stored_snapshots(settings: Settings) -> dict[str, Path]:
    """Every snapshot the directory actually holds, keyed by id.

    Paths come from :meth:`Path.glob` and are filtered to regular files whose RESOLVED location
    is still inside the directory — so a symlink planted in the directory but pointing out of it
    is not enumerated.
    """
    directory = _dir(settings).resolve()
    found: dict[str, Path] = {}
    for path in directory.glob(f"*{_SUFFIX}"):
        resolved = path.resolve()
        if resolved.is_file() and resolved.is_relative_to(directory):
            found[path.name[: -len(_SUFFIX)]] = resolved
    return found


def _snapshot_path(settings: Settings, snapshot_id: str) -> Path:
    """The on-disk path for ``snapshot_id``, or 404.

    **No path is ever built from the caller's string.** The directory is enumerated by the
    server, and the caller's id only ever participates in an equality lookup against ids the
    server itself derived from real filenames. So there is no path expression for a traversal to
    live in — the class of bug is absent rather than guarded against.

    That is a stronger property than the guard it replaces (an allowlist plus a containment
    check), and it is why the structure is this way round. It also happens to be what a static
    analyser can see, but the reason to prefer it is that "cannot construct a bad path" beats
    "constructs a path, then checks it".

    :func:`_safe_id` still runs first as a cheap rejection of input that could not name a
    snapshot under any circumstances; it is no longer the thing standing between a caller and
    the filesystem.
    """
    stored = _stored_snapshots(settings)
    path = stored.get(_safe_id(snapshot_id))
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such snapshot: {snapshot_id}")
    return path


def _read_meta(path: Path) -> dict[str, Any]:
    """Return the serializer's own ``meta`` descriptor from a snapshot, or ``{}``.

    Read from the file rather than recomputed, so the listing cannot disagree with what was
    actually saved. A file that is unreadable or not one of ours yields ``{}`` rather than
    failing the whole listing — one bad file must not hide every good one.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            return dict(json.loads(str(data["meta"])))
    except Exception:  # noqa: BLE001 — any unreadable file is simply un-described
        return {}


def _describe(path: Path) -> SnapshotModel:
    stat = path.stat()
    meta = _read_meta(path)
    return SnapshotModel(
        id=path.name[: -len(_SUFFIX)] if path.name.endswith(_SUFFIX) else path.name,
        created=datetime.fromtimestamp(stat.st_mtime, tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        size_bytes=stat.st_size,
        description=str(meta.get("description", "")),
        meta=meta,
    )


@router.post("/v1/model/snapshots", response_model=SnapshotModel, status_code=status.HTTP_201_CREATED)
def save_snapshot(
    body: SnapshotRequest,
    state: Annotated[AppState, Depends(get_state)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SnapshotModel:
    """Serialise the current model to ``snapshots_dir``. ``409`` when there is none.

    The 409 is checked HERE rather than left to the serializer: ``LMUSerializer.save`` raises
    ``RuntimeError`` on an unfitted model, which would surface as a 500 for what is a
    precondition failure. Mirrors ``GET /v1/model``'s existing refusal.
    """
    model = state.model
    if model is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "no trained model; call POST /v1/train first")

    directory = _dir(settings)
    snapshot_id = f"lmu-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S%f')}"
    path = directory / f"{snapshot_id}{_SUFFIX}"

    LMUSerializer().save(model, str(path))

    # The description is the CALLER's, not the model's, so the serializer does not carry it.
    # Round-trip it through the meta blob so a listing can show it without a sidecar file.
    if body.description:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files if key != "meta"}
            meta = json.loads(str(data["meta"]))
        meta["description"] = body.description
        np.savez(str(path), meta=json.dumps(meta), **arrays)

    return _describe(path)


@router.get("/v1/model/snapshots", response_model=SnapshotListResponse)
def list_snapshots(settings: Annotated[Settings, Depends(get_settings)]) -> SnapshotListResponse:
    """Every stored snapshot, newest first."""
    found = sorted(_stored_snapshots(settings).values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return SnapshotListResponse(snapshots=[_describe(path) for path in found])


@router.get("/v1/model/snapshots/{snapshot_id}", response_model=SnapshotModel)
def get_snapshot(snapshot_id: str, settings: Annotated[Settings, Depends(get_settings)]) -> SnapshotModel:
    """One snapshot's metadata. ``404`` when absent."""
    return _describe(_snapshot_path(settings, snapshot_id))


@router.post("/v1/model/snapshots/{snapshot_id}/restore", response_model=RestoreResponse)
def restore_snapshot(
    snapshot_id: str,
    state: Annotated[AppState, Depends(get_state)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RestoreResponse:
    """Load a snapshot into the app state. ``404`` when absent, ``409`` while a fit is running.

    The resulting state is ``restored``, never ``trained``: this process did not fit the model,
    and no ``TrainResult`` is invented to make the status shape uniform.
    """
    path = _snapshot_path(settings, snapshot_id)

    # A restore that lands mid-fit would have the training thread publish over it moments later,
    # so the caller would be told the restore succeeded and then silently get the fitted model.
    # Non-blocking, like POST /v1/train's own guard.
    if not state.train_lock.acquire(blocking=False):
        raise HTTPException(status.HTTP_409_CONFLICT, "a training run is in progress; retry when it completes")
    try:
        try:
            model: LMURegressor = LMUSerializer().load(str(path))
        except Exception as exc:  # noqa: BLE001 — a corrupt/foreign file is a 422, not a 500
            raise HTTPException(422, f"snapshot could not be loaded: {exc}") from exc
        state.set_restored(model, snapshot_id)
    finally:
        state.train_lock.release()

    return RestoreResponse(
        state="restored",
        restored_from=snapshot_id,
        topology=dict(model.describe_topology()),
        metrics=model.metrics(),
    )
