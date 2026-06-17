"""Model route: ``GET /v1/model`` — current topology + metrics. ``409`` if none."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from juniper_recurrence.routers._common import get_state
from juniper_recurrence.schemas import ModelResponse
from juniper_recurrence.state import AppState

router = APIRouter(tags=["model"])


@router.get("/v1/model", response_model=ModelResponse)
def get_model(state: Annotated[AppState, Depends(get_state)]) -> ModelResponse:
    """Topology (``describe_topology``) + regression metrics of the current model."""
    model = state.model
    if model is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "no trained model; call POST /v1/train first")
    return ModelResponse(topology=dict(model.describe_topology()), metrics=model.metrics())
