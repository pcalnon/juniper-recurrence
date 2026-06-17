"""In-memory training-event sink for the juniper-recurrence app (plan §9).

A bounded, ordered ring buffer that is itself the ``on_event`` callable passed to
``juniper_service_core.TrainingLifecycle``. The lifecycle stamps a monotonic ``seq``
on each event before it arrives here, so a snapshot is already legally ordered. The
buffer feeds ``GET /v1/training/status``; older events past ``maxlen`` are dropped
(a single synchronous run emits only ``training_start`` → ``epoch_end`` →
``training_end``, so the default cap is never reached in practice).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from juniper_model_core import TrainingEvent

__all__ = ["EventSink"]


class EventSink:
    """A bounded, ordered sink for :class:`~juniper_model_core.TrainingEvent`\\ s."""

    def __init__(self, maxlen: int = 256) -> None:
        self._events: deque[TrainingEvent] = deque(maxlen=maxlen)

    def __call__(self, event: TrainingEvent) -> None:
        """Append an emitted event (the ``on_event`` lifecycle hook)."""
        self._events.append(event)

    def snapshot(self) -> list[TrainingEvent]:
        """An ordered copy of the buffered events (oldest first)."""
        return list(self._events)
