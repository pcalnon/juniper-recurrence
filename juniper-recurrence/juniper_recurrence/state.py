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
    from juniper_recurrence_model import LMURegressor

    from juniper_recurrence.events import EventSink
    from juniper_recurrence.schemas import DatasetDescriptor

__all__ = ["AppState"]


class AppState:
    """Single in-process holder for the trained model + last run artifacts."""

    def __init__(self) -> None:
        self.train_lock = threading.Lock()
        self._model: LMURegressor | None = None
        self._result: TrainResult | None = None
        self._events: EventSink | None = None
        self._dataset: DatasetDescriptor | None = None

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
        self._model = model  # published last

    @property
    def model(self) -> LMURegressor | None:
        return self._model

    @property
    def dataset(self) -> DatasetDescriptor | None:
        return self._dataset

    def status(self) -> tuple[str, TrainResult | None, list]:
        """``("idle"|"trained", last_result, ordered_events)`` for ``/v1/training/status``."""
        if self._model is None:
            return ("idle", None, [])
        events = self._events.snapshot() if self._events is not None else []
        return ("trained", self._result, events)
