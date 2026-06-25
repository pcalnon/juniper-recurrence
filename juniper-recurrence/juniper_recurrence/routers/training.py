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

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from juniper_data_client import JuniperDataClientError
from juniper_service_core import TrainingLifecycle

from juniper_recurrence._readout import build_lmu_regressor
from juniper_recurrence.data import load_sequence_data
from juniper_recurrence.events import EventSink
from juniper_recurrence.routers._common import get_settings, get_state, map_data_error
from juniper_recurrence.schemas import DatasetDescriptor, EventModel, StatusResponse, TrainRequest, TrainResponse
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState

router = APIRouter(tags=["training"])

logger = logging.getLogger(__name__)


@router.post("/v1/train", response_model=TrainResponse)
def train(
    req: TrainRequest,
    state: Annotated[AppState, Depends(get_state)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TrainResponse:
    """Synchronously train the LMU on a dataset split and return the ``TrainResult``."""
    if not state.train_lock.acquire(blocking=False):
        logger.warning("POST /v1/train rejected: a training run is already in progress")
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
            logger.warning("training aborted: dataset fetch failed (dataset=%s): %s", req.dataset.dataset_id or req.dataset.name or req.dataset.generator, exc)
            raise map_data_error(exc) from exc

        d = req.d if req.d is not None else settings.default_d
        theta = req.theta if req.theta is not None else settings.default_theta

        sink = EventSink()
        try:
            model = build_lmu_regressor(
                d=d,
                theta=theta,
                readout=req.readout,
                ridge=req.ridge,
                rff_features=req.rff_features,
                rff_gamma=req.rff_gamma,
                mlp_hidden=req.mlp_hidden,
                mlp_weight_decay=req.mlp_weight_decay,
                mlp_lr=req.mlp_lr,
                mlp_max_epochs=req.mlp_max_epochs,
                mlp_patience=req.mlp_patience,
                default_ridge=settings.default_ridge,
            )
        except ValueError as exc:
            # Schema validation already rejects bad-knob / ridge-with-mlp combinations (422). The only
            # ValueError reachable here is the readout='mlp' torch-capability gap — a deployment without
            # the [torch] extra — which is a service-unavailability, not a client error.
            logger.warning("training unavailable: readout=%r requires the [torch] extra: %s", req.readout, exc)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        logger.info("training start: dataset=%s split=%s windows=%s d=%s theta=%s readout=%s", descriptor["dataset_id"], descriptor.get("split"), descriptor.get("n_windows"), d, theta, req.readout or "linear")
        start = time.perf_counter()
        lifecycle = TrainingLifecycle(model, on_event=sink)
        result = lifecycle.run(sequence.X, sequence.y, **sequence.fit_kwargs())

        dataset = DatasetDescriptor(**descriptor)
        state.set_trained(model, result, sink, dataset)
        logger.info("training complete: dataset=%s epochs=%s duration=%.3fs metrics=%s", descriptor["dataset_id"], result.n_epochs, time.perf_counter() - start, result.final_metrics)
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
