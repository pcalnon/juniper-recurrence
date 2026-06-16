"""Dataset route: ``GET /v1/dataset`` — descriptor of the last-loaded split (thin v1)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from juniper_recurrence.routers._common import get_state
from juniper_recurrence.schemas import DatasetDescriptor
from juniper_recurrence.state import AppState

router = APIRouter(tags=["dataset"])


@router.get("/v1/dataset", response_model=DatasetDescriptor)
def get_dataset(state: Annotated[AppState, Depends(get_state)]) -> DatasetDescriptor:
    """Descriptor (name / split / shapes) of the dataset the current model trained on."""
    descriptor = state.dataset
    if descriptor is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "no dataset loaded; call POST /v1/train first")
    return descriptor
