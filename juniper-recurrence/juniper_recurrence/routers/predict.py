"""Predict route: ``POST /v1/predict`` — continuous ``ŷ`` over the trained LMU.

Accepts inline arrays (``X`` + optional ``dt`` / ``target_dt`` / ``seq_lengths``) or a
dataset ref. Passes ``dt`` explicitly to engage the Δt path. Returns continuous
predictions — never an ``argmax`` collapse to labels (RK-6). ``409`` before any train.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from juniper_data_client import JuniperDataClientError

from juniper_recurrence.data import load_sequence_data
from juniper_recurrence.routers._common import get_settings, get_state, map_data_error
from juniper_recurrence.schemas import PredictRequest, PredictResponse
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState

router = APIRouter(tags=["predict"])

logger = logging.getLogger(__name__)


@router.post("/v1/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    state: Annotated[AppState, Depends(get_state)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PredictResponse:
    """Predict continuous targets for inline ``X`` or a dataset split."""
    model = state.model
    if model is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "no trained model; call POST /v1/train first")

    if req.X is not None:
        features = np.asarray(req.X, dtype=float)
        kwargs: dict[str, Any] = {}
        if req.dt is not None:
            kwargs["dt"] = np.asarray(req.dt, dtype=float)
        if req.target_dt is not None:
            kwargs["target_dt"] = np.asarray(req.target_dt, dtype=float)
        if req.seq_lengths is not None:
            kwargs["seq_lengths"] = np.asarray(req.seq_lengths)
    else:
        try:
            sequence, _ = load_sequence_data(
                base_url=settings.juniper_data_url,
                api_key=settings.juniper_data_api_key,
                dataset_id=req.dataset.dataset_id,
                name=req.dataset.name,
                generator=req.dataset.generator,
                params=req.dataset.params,
                split=req.dataset.split,
            )
        except (JuniperDataClientError, ValueError) as exc:
            logger.warning("predict aborted: dataset fetch failed (dataset=%s): %s", req.dataset.dataset_id or req.dataset.name or req.dataset.generator, exc)
            raise map_data_error(exc) from exc
        features = sequence.X
        kwargs = sequence.fit_kwargs()

    try:
        predictions = model.predict(features, **kwargs)
    except (ValueError, RuntimeError) as exc:
        logger.warning("prediction failed: %s", exc)
        # 422 literal: avoids Starlette's deprecated HTTP_422_UNPROCESSABLE_ENTITY constant.
        raise HTTPException(422, f"prediction failed: {exc}") from exc

    return PredictResponse(predictions=predictions.tolist(), shape=list(predictions.shape))
