"""In-process application state for the juniper-recurrence service (plan §4).

A single in-memory holder for the current trained model, its last
:class:`~juniper_model_core.TrainResult`, the training event buffer, and a
descriptor of the dataset it was trained on. One instance lives per app (stored on
``app.state.app_state`` by :func:`juniper_recurrence.app.build_app`) — so each
``build_app`` gets isolated state, which keeps tests hermetic while remaining the
single in-process holder the plan calls for (uvicorn ``workers=1``; persistence and
scale-out are deferred to WS-8).

Concurrency: ``train_lock`` serialises training (a second concurrent ``/v1/train``
gets ``409`` via a non-blocking acquire). Readers of the model alone (``model`` /
``dataset``) take no lock — :meth:`set_trained` publishes the model reference **last**, so a
reader that sees a non-``None`` model also sees a fully populated result / events /
descriptor (publish-the-pointer-last). The status and the two routes that check an
``expect_operation_id`` (``predict``, snapshot save) read through the small record lock
described below, so the model and its operation id arrive as a pair.

Operation identity (W1.5; F-S6 / F-CON1 / F-CON2 of juniper-ml
``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``):
every request that takes ``train_lock`` — a fit, or a snapshot restore — is an
:class:`Operation` with its own ``operation_id`` (uuid4 hex), minted at the moment the lock
is taken. The state records three things about operations, all guarded by a small private
lock so a status reader always sees one consistent picture:

* the operation **holding the lock right now**, so a refused caller and a status poll can
  both name it (who is busy, and since when);
* the operation that **produced the in-memory model**, so ``/v1/predict`` and a snapshot save
  can check a caller's ``expect_operation_id`` against it;
* the **last operation that failed**, so a fit that raises leaves a terminal record instead
  of silently leaving the status describing an earlier run.

That private lock is only ever held for a few assignments. ``train_lock`` is only ever
*tried* (``blocking=False``) while holding it, and released while holding it, so taking the
lock and recording the holder are one atomic step, and so are clearing the holder and
releasing the lock — a caller that wins the lock the instant it is released can never have
its own record cleared by the previous holder.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from juniper_model_core import TrainResult
    from juniper_model_core.crossval import CrossValResult
    from juniper_recurrence_model import LMURegressor

    from juniper_recurrence.events import EventSink
    from juniper_recurrence.schemas import DatasetDescriptor

__all__ = ["AppState", "FailureRecord", "Operation", "OperationKind", "StatusSnapshot", "new_operation_id", "utc_now_iso"]

#: The two kinds of request that take ``train_lock`` and so replace (or try to replace) the model.
OperationKind = Literal["train", "restore"]


def new_operation_id() -> str:
    """A fresh operation id: a uuid4 as 32 lowercase hex characters."""
    return uuid.uuid4().hex


def utc_now_iso() -> str:
    """The current time as ISO-8601 UTC with millisecond precision and a ``Z`` suffix.

    Milliseconds, not seconds: two requests a few ms apart (the F-S6 reproduction was 5 ms)
    must not read as simultaneous.
    """
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Operation:
    """One request that took ``train_lock``: a fit (``train``) or a snapshot restore (``restore``).

    Frozen, so a status reader can hold one without it changing underneath it; the state swaps
    in an updated copy instead (see :meth:`AppState.note_dataset`).
    """

    operation_id: str
    kind: OperationKind
    #: When the operation took ``train_lock`` (ISO-8601 UTC). Served as ``busy_since`` while it runs.
    started_at: str
    #: The request's ``X-Request-ID`` header, or ``None`` when it sent none.
    requested_by: str | None = None
    #: The dataset the operation is for: the requested id until a ``name`` / ``generator``
    #: reference resolves, then the resolved one. ``None`` for a restore.
    dataset_id: str | None = None


@dataclass(frozen=True)
class FailureRecord:
    """The terminal record of an operation that did not complete."""

    operation: Operation
    #: The error detail the failing request returned (for an unexpected exception, its type and message).
    detail: str
    #: The HTTP status the failing request returned.
    status_code: int


@dataclass(frozen=True)
class StatusSnapshot:
    """One consistent read of the state for ``GET /v1/training/status``.

    ``state`` is one of ``idle`` / ``training`` / ``restoring`` / ``trained`` / ``restored`` /
    ``failed``; ``operation`` is the operation that state describes (``None`` under ``idle``).
    """

    state: str
    operation: Operation | None = None
    #: Set only while an operation holds ``train_lock`` (``training`` / ``restoring``).
    busy_since: str | None = None
    #: Set only under ``failed``.
    failure: FailureRecord | None = None
    #: The operation that produced the in-memory model, whatever ``state`` says; ``None`` without a model.
    model_operation_id: str | None = None
    result: TrainResult | None = None
    events: list = field(default_factory=list)
    #: The snapshot id, only under ``restored`` (the field's shipped meaning).
    restored_from: str | None = None


class AppState:
    """Single in-process holder for the trained model + last run artifacts."""

    def __init__(self) -> None:
        self.train_lock = threading.Lock()
        self._model: LMURegressor | None = None
        # Snapshot id this model was RESTORED from, or None when it came from a fit in this
        # process. Drives the ``restored`` status value -- see set_restored.
        self._restored_from: str | None = None
        self._result: TrainResult | None = None
        self._events: EventSink | None = None
        self._dataset: DatasetDescriptor | None = None
        # Operation records (W1.5) -- see the module docstring. Guarded by _record_lock.
        self._record_lock = threading.Lock()
        self._active: Operation | None = None
        self._model_operation: Operation | None = None
        self._failure: FailureRecord | None = None
        # Cross-validation runs are independent of training; a separate lock + last-result holder.
        self.crossval_lock = threading.Lock()
        self._crossval_result: CrossValResult | None = None
        self._crossval_dataset: DatasetDescriptor | None = None

    # --- operations (W1.5) ------------------------------------------------------------------

    def try_begin(self, kind: OperationKind, *, requested_by: str | None = None, dataset_id: str | None = None) -> tuple[Operation | None, Operation | None]:
        """Take ``train_lock`` without blocking and register a new operation, as one step.

        Returns ``(operation, None)`` when the lock was free, and ``(None, holder)`` when it was
        not -- ``holder`` being the operation that holds it, so the refusal can name it. The
        holder is ``None`` only when the lock was taken outside this method (tests do that).
        The caller that gets an operation MUST hand it to :meth:`end` exactly once.
        """
        with self._record_lock:
            if not self.train_lock.acquire(blocking=False):
                return None, self._active
            operation = Operation(operation_id=new_operation_id(), kind=kind, started_at=utc_now_iso(), requested_by=requested_by, dataset_id=dataset_id)
            self._active = operation
            return operation, None

    def note_dataset(self, operation: Operation, dataset_id: str | None) -> None:
        """Record the resolved dataset id on an in-flight operation.

        A ``name`` or ``generator`` reference has no id until the fetch resolves it, which is
        the slow part of a fit, so the id is filled in afterwards rather than left unknown.
        """
        with self._record_lock:
            if self._active is not None and self._active.operation_id == operation.operation_id:
                self._active = replace(self._active, dataset_id=dataset_id)

    def fail(self, operation: Operation, *, detail: str, status_code: int) -> None:
        """Record that ``operation`` failed. The model, if any, is untouched -- it is an earlier one."""
        with self._record_lock:
            self._failure = FailureRecord(operation=self._latest(operation), detail=detail, status_code=status_code)

    def end(self, operation: Operation) -> None:
        """Clear the in-flight record and release ``train_lock``, as one step."""
        with self._record_lock:
            if self._active is not None and self._active.operation_id == operation.operation_id:
                self._active = None
            self.train_lock.release()

    def _latest(self, operation: Operation | None) -> Operation | None:
        """The newest copy of ``operation`` (the in-flight one carries any resolved dataset id). Call under ``_record_lock``."""
        if operation is not None and self._active is not None and self._active.operation_id == operation.operation_id:
            return self._active
        return operation

    # --- publishing a model -----------------------------------------------------------------

    def set_trained(
        self,
        model: LMURegressor,
        result: TrainResult,
        events: EventSink,
        dataset: DatasetDescriptor,
        *,
        operation: Operation | None = None,
    ) -> None:
        """Publish a completed training run. Sets ``_model`` last (see module docstring).

        ``operation`` is the fit that produced the model; it becomes the model's operation id.
        A success supersedes any earlier failure record.
        """
        with self._record_lock:
            self._result = result
            self._events = events
            self._dataset = dataset
            # A fit SUPERSEDES a restore: this model came from a run in this process, so the
            # restored-from marker must not survive and report the new model as loaded from disk.
            self._restored_from = None
            self._model_operation = self._latest(operation)
            self._failure = None
            self._model = model  # published last

    def set_restored(self, model: LMURegressor, snapshot_id: str, *, operation: Operation | None = None) -> None:
        """Publish a model LOADED FROM DISK, with no training run behind it.

        Deliberately does **not** synthesise a :class:`TrainResult`. This process did not fit
        this model, and inventing epoch/timing fields to make the status shape uniform would
        report a run that never happened -- the defect class this whole feature exists to close
        (design §6/§11.3 of juniper-ml
        ``notes/JUNIPER_2026-09-16_JUNIPER-RECURRENCE_MODEL-PERSISTENCE-DESIGN.md``).

        ``_result`` / ``_events`` / ``_dataset`` are cleared rather than left stale: a previous
        fit's result beside a restored model would be the same misattribution in a subtler form.
        The model's own metrics survive on the model and are served by ``GET /v1/model``.

        ``operation`` is the restore itself, so a restored model carries the id minted when the
        restore took the lock -- not the id of the fit that produced the snapshot, which this
        process may never have seen.

        Sets ``_model`` last, like :meth:`set_trained` (publish-the-pointer-last).
        """
        with self._record_lock:
            self._result = None
            self._events = None
            self._dataset = None
            self._restored_from = snapshot_id
            self._model_operation = self._latest(operation)
            self._failure = None
            self._model = model  # published last

    # --- readers ----------------------------------------------------------------------------

    @property
    def model(self) -> LMURegressor | None:
        return self._model

    @property
    def dataset(self) -> DatasetDescriptor | None:
        return self._dataset

    def model_with_operation(self) -> tuple[LMURegressor | None, str | None]:
        """The in-memory model and the id of the operation that produced it, read together.

        Read as a pair so a fit publishing in between cannot pair one model with another's id.
        """
        with self._record_lock:
            operation = self._model_operation
            return self._model, operation.operation_id if operation is not None else None

    def status_snapshot(self) -> StatusSnapshot:
        """Everything ``GET /v1/training/status`` reports, read in one consistent step.

        Precedence: an operation holding the lock (``training`` / ``restoring``) → the last
        operation having failed (``failed``) → no model (``idle``) → a restored model
        (``restored``) → a fitted one (``trained``).

        ``restored`` is a separate state, not a flavour of ``trained``: a model loaded from a
        snapshot is present and predictable, but **this process never fitted it**, so there is
        no result and no event stream to report. ``failed`` leaves any earlier model loaded and
        predictable; ``model_operation_id`` names it.
        """
        with self._record_lock:
            active, failure, model_operation = self._active, self._failure, self._model_operation
            model, restored_from, result, events = self._model, self._restored_from, self._result, self._events
        model_operation_id = model_operation.operation_id if model_operation is not None else None
        if active is not None:
            return StatusSnapshot(state="training" if active.kind == "train" else "restoring", operation=active, busy_since=active.started_at, model_operation_id=model_operation_id)
        if failure is not None:
            return StatusSnapshot(state="failed", operation=failure.operation, failure=failure, model_operation_id=model_operation_id)
        if model is None:
            return StatusSnapshot(state="idle")
        if restored_from is not None:
            return StatusSnapshot(state="restored", operation=model_operation, model_operation_id=model_operation_id, restored_from=restored_from)
        return StatusSnapshot(state="trained", operation=model_operation, model_operation_id=model_operation_id, result=result, events=events.snapshot() if events is not None else [])

    def status(self) -> tuple[str, TrainResult | None, list]:
        """``(state, last_result, ordered_events)`` -- the pre-W1.5 view of :meth:`status_snapshot`."""
        snapshot = self.status_snapshot()
        return (snapshot.state, snapshot.result, snapshot.events)

    @property
    def restored_from(self) -> str | None:
        """Snapshot id the current model was restored from, or ``None`` if it was fitted here."""
        return self._restored_from

    def set_crossval(self, result: CrossValResult, dataset: DatasetDescriptor) -> None:
        """Publish a completed cross-validation run. Sets ``_crossval_result`` last (publish-the-pointer-last)."""
        self._crossval_dataset = dataset
        self._crossval_result = result  # published last

    def crossval_status(self) -> tuple[str, CrossValResult | None, DatasetDescriptor | None]:
        """``("idle"|"done", last_result, dataset)`` for ``GET /v1/crossval/status``."""
        if self._crossval_result is None:
            return ("idle", None, None)
        return ("done", self._crossval_result, self._crossval_dataset)
