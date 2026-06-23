"""TST-8: ``EventSink`` ring-buffer eviction (``events.py`` ``maxlen``).

``EventSink`` is the ``on_event`` sink wired into ``TrainingLifecycle``; it is a bounded,
ordered ``deque``. Existing route tests only ever push the three events of a single
synchronous run (start -> epoch_end -> end), so the overflow / eviction branch — the
reason the buffer is bounded at all — was untested. These tests drive past ``maxlen``.
"""

from __future__ import annotations

from juniper_model_core import TrainingEvent, TrainingEventType

from juniper_recurrence.events import EventSink

_EVENT_TYPE = next(iter(TrainingEventType))


def _event(seq: int) -> TrainingEvent:
    """A distinct, ordered TrainingEvent identified by its ``seq``."""
    return TrainingEvent(type=_EVENT_TYPE, payload={"seq": seq}, seq=seq)


def test_sink_evicts_oldest_past_maxlen() -> None:
    sink = EventSink(maxlen=2)
    for i in range(5):
        sink(_event(i))  # __call__ is the on_event hook
    assert [e.seq for e in sink.snapshot()] == [3, 4]  # 0,1,2 evicted; order kept


def test_sink_under_capacity_keeps_all_in_order() -> None:
    sink = EventSink(maxlen=8)
    for i in range(3):
        sink(_event(i))
    assert [e.seq for e in sink.snapshot()] == [0, 1, 2]


def test_default_maxlen_holds_a_typical_run() -> None:
    # The default cap (256) is never reached by a normal start/epoch/end run.
    sink = EventSink()
    for i in range(10):
        sink(_event(i))
    assert len(sink.snapshot()) == 10


def test_snapshot_is_an_independent_copy() -> None:
    sink = EventSink(maxlen=4)
    sink(_event(0))
    snap = sink.snapshot()
    sink(_event(1))
    assert [e.seq for e in snap] == [0]  # the earlier snapshot is not mutated
