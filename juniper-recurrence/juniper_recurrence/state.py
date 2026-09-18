"""In-process application state for the juniper-recurrence service (plan §4).

A single in-memory holder for the current trained model, its last
:class:`~juniper_model_core.TrainResult`, the training event buffer, and a
descriptor of the dataset it was trained on. One instance lives per app (stored on
``app.state.app_state`` by :func:`juniper_recurrence.app.build_app`) — so each
``build_app`` gets isolated state, which keeps tests hermetic while remaining the
single in-process holder the plan calls for (uvicorn ``workers=1``; persistence and
scale-out are deferred to WS-8).

Concurrency: ``train_lock`` serialises training (a second concurrent ``/v1/train``
gets ``409`` via a non-blocking acquire). Readers (``predict`` / ``status`` /
``model`` / ``dataset``) take no lock — :meth:`set_trained` publishes the model
reference **last**, so a reader that sees a non-``None`` model also sees a fully
populated result / events / descriptor (publish-the-pointer-last).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from juniper_model_core import TrainResult
    from juniper_model_core.crossval import CrossValResult
    from juniper_recurrence_model import LMURegressor

    from juniper_recurrence.events import EventSink
    from juniper_recurrence.schemas import DatasetDescriptor

__all__ = ["AppState"]


class AppState:
    """Single in-process holder for the trained model + last run artifacts."""

    def __init__(self) -> None:
        self.train_lock = threading.Lock()
        self._model: LMURegressor | None = None
        # Snapshot id this model was RESTORED from, or None when it came from a fit in this
        # process. Drives the third ``status()`` value -- see set_restored.
        self._restored_from: str | None = None
        self._result: TrainResult | None = None
        self._events: EventSink | None = None
        self._dataset: DatasetDescriptor | None = None
        # Cross-validation runs are independent of training; a separate lock + last-result holder.
        self.crossval_lock = threading.Lock()
        self._crossval_result: CrossValResult | None = None
        self._crossval_dataset: DatasetDescriptor | None = None

    def set_trained(
        self,
        model: LMURegressor,
        result: TrainResult,
        events: EventSink,
        dataset: DatasetDescriptor,
    ) -> None:
        """Publish a completed training run. Sets ``_model`` last (see module docstring)."""
        self._result = result
        self._events = events
        self._dataset = dataset
        # A fit SUPERSEDES a restore: this model came from a run in this process, so the
        # restored-from marker must not survive and report the new model as loaded from disk.
        self._restored_from = None
        self._model = model  # published last

    def set_restored(self, model: LMURegressor, snapshot_id: str) -> None:
        """Publish a model LOADED FROM DISK, with no training run behind it.

        Deliberately does **not** synthesise a :class:`TrainResult`. This process did not fit
        this model, and inventing epoch/timing fields to make the status shape uniform would
        report a run that never happened -- the defect class this whole feature exists to close
        (design §6/§11.3 of juniper-ml
        ``notes/JUNIPER_2026-09-16_JUNIPER-RECURRENCE_MODEL-PERSISTENCE-DESIGN.md``).

        ``_result`` / ``_events`` / ``_dataset`` are cleared rather than left stale: a previous
        fit's result beside a restored model would be the same misattribution in a subtler form.
        The model's own metrics survive on the model and are served by ``GET /v1/model``.

        Sets ``_model`` last, like :meth:`set_trained` (publish-the-pointer-last).
        """
        self._result = None
        self._events = None
        self._dataset = None
        self._restored_from = snapshot_id
        self._model = model  # published last

    @property
    def model(self) -> LMURegressor | None:
        return self._model

    @property
    def dataset(self) -> DatasetDescriptor | None:
        return self._dataset

    def status(self) -> tuple[str, TrainResult | None, list]:
        """``("idle"|"trained"|"restored", last_result, ordered_events)`` for ``/v1/training/status``.

        ``restored`` is a THIRD state, not a flavour of ``trained``: a model loaded from a
        snapshot is present and predictable, but **this process never fitted it**, so there is no
        result and no event stream to report. Collapsing it into ``trained`` would lose exactly
        that distinction. The snapshot id is available from :attr:`restored_from` so a consumer
        can say *which* model, not merely that one was loaded.
        """
        if self._model is None:
            return ("idle", None, [])
        if self._restored_from is not None:
            return ("restored", None, [])
        events = self._events.snapshot() if self._events is not None else []
        return ("trained", self._result, events)

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
