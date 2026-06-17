"""Training routes: ``POST /v1/train`` (synchronous) + ``GET /v1/training/status``.

D-WS4b-2 — training runs **inline**: load the 3-D NPZ, construct ``LMURegressor``,
drive ``TrainingLifecycle.run`` to completion on the request thread, store the model +
result + event buffer, and return the ``TrainResult`` in the response. No background
task, no WebSocket stream (deferred to WS-8). Correct for the µs one-shot ``lstsq``.

A non-blocking ``train_lock`` serialises runs — a second concurrent ``/v1/train`` gets
``409`` rather than torn state. The data fetch happens inside the lock so the whole run
is serialised (the fetch dominates wall-clock, the solve is negligible).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from juniper_data_client import JuniperDataClientError
from juniper_recurrence_model import LMURegressor
from juniper_service_core import TrainingLifecycle

from juniper_recurrence.data import load_sequence_data
from juniper_recurrence.events import EventSink
from juniper_recurrence.routers._common import get_settings, get_state, map_data_error
from juniper_recurrence.schemas import DatasetDescriptor, EventModel, StatusResponse, TrainRequest, TrainResponse
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState

router = APIRouter(tags=["training"])


@router.post("/v1/train", response_model=TrainResponse)
def train(
    req: TrainRequest,
    state: Annotated[AppState, Depends(get_state)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TrainResponse:
    """Synchronously train the LMU on a dataset split and return the ``TrainResult``."""
    if not state.train_lock.acquire(blocking=False):
        raise HTTPException(status.HTTP_409_CONFLICT, "a training run is already in progress")
    try:
        try:
            sequence, descriptor = load_sequence_data(
                base_url=settings.juniper_data_url,
                api_key=settings.juniper_data_api_key,
                dataset_id=req.dataset.dataset_id,
                name=req.dataset.name,
                generator=req.dataset.generator,
                params=req.dataset.params,
                split=req.dataset.split,
            )
        except (JuniperDataClientError, ValueError) as exc:
            raise map_data_error(exc) from exc

        d = req.d if req.d is not None else settings.default_d
        theta = req.theta if req.theta is not None else settings.default_theta
        ridge = req.ridge if req.ridge is not None else settings.default_ridge

        sink = EventSink()
        model = LMURegressor(d=d, theta=theta, ridge=ridge)
        lifecycle = TrainingLifecycle(model, on_event=sink)
        result = lifecycle.run(sequence.X, sequence.y, **sequence.fit_kwargs())

        dataset = DatasetDescriptor(**descriptor)
        state.set_trained(model, result, sink, dataset)
        return TrainResponse(
            final_metrics=result.final_metrics,
            n_epochs=result.n_epochs,
            stopped_reason=result.stopped_reason,
            dataset=dataset,
        )
    finally:
        state.train_lock.release()


@router.get("/v1/training/status", response_model=StatusResponse)
def training_status(state: Annotated[AppState, Depends(get_state)]) -> StatusResponse:
    """Last training status + ordered events from the in-memory sink (instant — sync)."""
    state_name, result, events = state.status()
    return StatusResponse(
        state=state_name,
        final_metrics=result.final_metrics if result is not None else None,
        stopped_reason=result.stopped_reason if result is not None else None,
        events=[EventModel(type=event.type, seq=event.seq, payload=event.payload) for event in events],
    )
