"""Training routes: ``POST /v1/train`` (synchronous) + ``GET /v1/training/status``.

D-WS4b-2 — training runs **inline**: load the 3-D NPZ, construct ``LMURegressor``,
drive ``TrainingLifecycle.run`` to completion on the request thread, store the model +
result + event buffer, and return the ``TrainResult`` in the response. No background
task, no WebSocket stream (deferred to WS-8). Correct for the µs one-shot ``lstsq``.

A non-blocking ``train_lock`` serialises runs — a second concurrent ``/v1/train`` gets
``409`` rather than torn state. The data fetch happens inside the lock so the whole run
is serialised (the fetch dominates wall-clock, the solve is negligible).

W1.5 (F-S6 / F-CON1 / F-CON2 of juniper-ml
``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``): the run
is an *operation* whose ``operation_id`` is minted when the route takes the lock. The id is in the
response, in the ``409`` a refused caller gets (with ``busy_since``, so it can tell whose run it hit
and since when), and in ``GET /v1/training/status`` -- which now reports a run in flight
(``training``) and a run that failed (``failed``) instead of describing an earlier one. The fit is
still **not cancellable**: a caller that gives up client-side leaves it running under the lock.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from juniper_data_client import JuniperDataClientError
from juniper_service_core import TrainingLifecycle

from juniper_recurrence import metrics
from juniper_recurrence._readout import build_lmu_regressor
from juniper_recurrence.data import load_sequence_data
from juniper_recurrence.events import EventSink
from juniper_recurrence.routers._common import RecordedOperation, RequestIdHeader, get_settings, get_state, map_data_error
from juniper_recurrence.schemas import BusyDetail, BusyResponse, DatasetDescriptor, EventModel, OperationFailure, StatusResponse, TrainRequest, TrainResponse
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState, Operation

router = APIRouter(tags=["training"])

logger = logging.getLogger(__name__)


def _busy_detail(holder: Operation | None) -> BusyDetail:
    """The ``409`` detail naming the operation that holds the lock (W1.5)."""
    message = "a snapshot restore is in progress" if holder is not None and holder.kind == "restore" else "a training run is already in progress"
    if holder is None:
        return BusyDetail(message=message)
    return BusyDetail(message=message, operation_id=holder.operation_id, operation=holder.kind, busy_since=holder.started_at, requested_by=holder.requested_by, dataset_id=holder.dataset_id)


@router.post(
    "/v1/train",
    response_model=TrainResponse,
    responses={status.HTTP_409_CONFLICT: {"model": BusyResponse, "description": "Another operation holds the service; the detail names it (operation_id, busy_since)."}},
)
def train(
    req: TrainRequest,
    state: Annotated[AppState, Depends(get_state)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_request_id: RequestIdHeader = None,
) -> TrainResponse:
    """Synchronously train the LMU on a dataset split and return the ``TrainResult``."""
    operation, holder = state.try_begin("train", requested_by=x_request_id or None, dataset_id=req.dataset.dataset_id)
    if operation is None:
        busy = _busy_detail(holder)
        logger.warning("POST /v1/train rejected: %s (holder operation_id=%s busy_since=%s)", busy.message, busy.operation_id, busy.busy_since)
        raise HTTPException(status.HTTP_409_CONFLICT, detail=busy.model_dump())
    with RecordedOperation(state, operation):
        try:
            sequence, descriptor = load_sequence_data(
                base_url=settings.juniper_data_url,
                api_key=settings.juniper_data_api_key,
                dataset_id=req.dataset.dataset_id,
                name=req.dataset.name,
                generator=req.dataset.generator,
                params=req.dataset.params,
                split=req.dataset.split,
                timeout=settings.juniper_data_timeout_seconds,
            )
        except (JuniperDataClientError, ValueError) as exc:
            logger.warning("training aborted: dataset fetch failed (operation_id=%s dataset=%s): %s", operation.operation_id, req.dataset.dataset_id or req.dataset.name or req.dataset.generator, exc)
            raise map_data_error(exc) from exc
        state.note_dataset(operation, descriptor["dataset_id"])

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

        logger.info("training start: operation_id=%s dataset=%s split=%s windows=%s d=%s theta=%s readout=%s", operation.operation_id, descriptor["dataset_id"], descriptor.get("split"), descriptor.get("n_windows"), d, theta, req.readout or "linear")
        start = time.perf_counter()
        lifecycle = TrainingLifecycle(model, on_event=sink)
        result = lifecycle.run(sequence.X, sequence.y, **sequence.fit_kwargs())

        dataset = DatasetDescriptor(**descriptor)
        duration = time.perf_counter() - start
        metrics.record_train(duration, result.final_metrics)
        response = TrainResponse(
            final_metrics=result.final_metrics,
            # W0.7: LMURegressor.fit scores final_metrics on the arrays it was fitted on (no X_val is
            # passed here), so they are in-sample. Set explicitly rather than left to the schema
            # default, so the claim sits where the metrics are produced.
            metrics_scope="in_sample",
            n_epochs=result.n_epochs,
            stopped_reason=result.stopped_reason,
            dataset=dataset,
            operation_id=operation.operation_id,
        )
        # Published LAST, once nothing left can fail: the status then says "trained" only for a run
        # whose caller is about to receive its result, never for one that went on to raise.
        state.set_trained(model, result, sink, dataset, operation=operation)
        logger.info("training complete: operation_id=%s dataset=%s epochs=%s duration=%.3fs metrics=%s", operation.operation_id, descriptor["dataset_id"], result.n_epochs, duration, result.final_metrics)
        return response


@router.get("/v1/training/status", response_model=StatusResponse)
def training_status(state: Annotated[AppState, Depends(get_state)]) -> StatusResponse:
    """The current or last operation, its outcome, and ordered events (instant — sync)."""
    snapshot = state.status_snapshot()
    operation = snapshot.operation
    result = snapshot.result
    failure = snapshot.failure
    return StatusResponse(
        state=snapshot.state,
        final_metrics=result.final_metrics if result is not None else None,
        stopped_reason=result.stopped_reason if result is not None else None,
        events=[EventModel(type=event.type, seq=event.seq, payload=event.payload) for event in snapshot.events],
        # Names WHICH snapshot under state="restored"; None otherwise. "Loaded from disk" without
        # an id is precise about the wrong thing (design §11.3).
        restored_from=snapshot.restored_from,
        operation_id=operation.operation_id if operation is not None else None,
        operation=operation.kind if operation is not None else None,
        busy_since=snapshot.busy_since,
        dataset_id=operation.dataset_id if operation is not None else None,
        requested_by=operation.requested_by if operation is not None else None,
        model_operation_id=snapshot.model_operation_id,
        failure=OperationFailure(detail=failure.detail, status_code=failure.status_code) if failure is not None else None,
    )
