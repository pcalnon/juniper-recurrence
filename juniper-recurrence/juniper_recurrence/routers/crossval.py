"""Cross-validation route: ``POST /v1/crossval`` (synchronous) + ``GET /v1/crossval/status``.

The *indirect* evaluation route (crossval-layer design §0): a dataset selection → the service
runs walk-forward folds over the ``_full`` arrays → an aggregated result. It drives
``juniper_model_core.crossval`` (model-core 0.2.0): per fold a fresh :class:`LMURegressor` is fit
on the training slice and scored held-out on the eval slice (the contract has no ``reset()`` /
``score()``, so the executor takes a model factory and scores externally), then per-fold metrics
are aggregated to mean / std.

Folds are derived from the ``_full`` split, which the WS-1 data contract emits in chronological
order — so index-order walk-forward folds are leakage-safe without a separate ordering key. Like
``/v1/train`` this runs **inline** (D-WS4b-2): folds of a closed-form ``lstsq`` solve are fast. A
non-blocking ``crossval_lock`` serialises runs (a second concurrent call gets ``409``); the most
recent result is persisted on the app state for ``GET /v1/crossval/status``.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from juniper_data_client import JuniperDataClientError
from juniper_model_core.crossval import CrossValResult, cross_validate, walk_forward_folds

from juniper_recurrence._readout import build_lmu_regressor
from juniper_recurrence.data import load_sequence_data
from juniper_recurrence.routers._common import get_settings, get_state, map_data_error
from juniper_recurrence.schemas import CrossValFoldModel, CrossValRequest, CrossValResponse, CrossValStatusResponse, DatasetDescriptor
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState

router = APIRouter(tags=["crossval"])

logger = logging.getLogger(__name__)


def _to_response(result: CrossValResult, dataset: DatasetDescriptor) -> CrossValResponse:
    """Render a :class:`~juniper_model_core.crossval.CrossValResult` as the API response model."""
    return CrossValResponse(
        task_type=result.task_type,
        n_folds=len(result.folds),
        folds=[CrossValFoldModel(fold=fold.fold, train_metrics=fold.train_metrics, eval_metrics=fold.eval_metrics, n_epochs=fold.n_epochs) for fold in result.folds],
        eval_aggregate=result.eval_aggregate,
        eval_std=result.eval_std,
        dataset=dataset,
    )


@router.post("/v1/crossval", response_model=CrossValResponse)
def crossval(
    req: CrossValRequest,
    state: Annotated[AppState, Depends(get_state)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CrossValResponse:
    """Synchronously cross-validate the LMU over walk-forward folds of the dataset's ``_full`` split."""
    if not state.crossval_lock.acquire(blocking=False):
        logger.warning("POST /v1/crossval rejected: a cross-validation run is already in progress")
        raise HTTPException(status.HTTP_409_CONFLICT, "a cross-validation run is already in progress")
    try:
        try:
            sequence, descriptor = load_sequence_data(
                base_url=settings.juniper_data_url,
                api_key=settings.juniper_data_api_key,
                dataset_id=req.dataset.dataset_id,
                name=req.dataset.name,
                generator=req.dataset.generator,
                params=req.dataset.params,
                split="full",  # CV always derives folds from the full chronological set (D-CV-4)
            )
        except (JuniperDataClientError, ValueError) as exc:
            logger.warning("cross-validation aborted: dataset fetch failed (dataset=%s): %s", req.dataset.dataset_id or req.dataset.name or req.dataset.generator, exc)
            raise map_data_error(exc) from exc

        d = req.d if req.d is not None else settings.default_d
        theta = req.theta if req.theta is not None else settings.default_theta

        try:
            folds = walk_forward_folds(
                sequence.X.shape[0],
                n_folds=req.n_folds,
                scheme=req.scheme,
                embargo=req.embargo,
                min_train=req.min_train,
            )
        except ValueError as exc:
            raise HTTPException(422, f"invalid cross-validation configuration: {exc}") from exc

        # The per-fold model factory, built once up front to validate the readout config (torch
        # availability for readout='mlp', knob combinations) and fail fast with a clean error rather
        # than surfacing identically on every fold. Schema validation already covers the client-side
        # knob errors (422); the case reachable here is the readout='mlp' torch-capability gap (503).
        make_model = partial(
            build_lmu_regressor,
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
        try:
            make_model()
        except ValueError as exc:
            logger.warning("cross-validation unavailable: readout=%r requires the [torch] extra: %s", req.readout, exc)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        logger.info("cross-validation start: dataset=%s folds=%s scheme=%s d=%s theta=%s readout=%s", descriptor["dataset_id"], len(folds), req.scheme, d, theta, req.readout or "linear")
        start = time.perf_counter()
        result = cross_validate(
            lambda _fold: make_model(),
            sequence.X,
            sequence.y,
            folds,
            aux=sequence.fit_kwargs(),
        )
        dataset = DatasetDescriptor(**descriptor)
        state.set_crossval(result, dataset)
        logger.info("cross-validation complete: dataset=%s folds=%s duration=%.3fs eval_aggregate=%s", descriptor["dataset_id"], len(result.folds), time.perf_counter() - start, result.eval_aggregate)
        return _to_response(result, dataset)
    finally:
        state.crossval_lock.release()


@router.get("/v1/crossval/status", response_model=CrossValStatusResponse)
def crossval_status(state: Annotated[AppState, Depends(get_state)]) -> CrossValStatusResponse:
    """The most recent cross-validation result (persisted in-process), or ``idle`` if none has run."""
    state_name, result, dataset = state.crossval_status()
    return CrossValStatusResponse(
        state=state_name,
        result=_to_response(result, dataset) if result is not None and dataset is not None else None,
    )
