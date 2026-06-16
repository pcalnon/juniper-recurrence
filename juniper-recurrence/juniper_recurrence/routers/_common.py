"""Shared router dependencies and error mapping for the juniper-recurrence API.

The app-state and settings dependencies read the per-app instances stashed on
``app.state`` by :func:`juniper_recurrence.app.build_app`. :func:`map_data_error`
translates juniper-data-client failures into the appropriate HTTP status so the
train / predict data path returns ``404`` / ``422`` / ``502`` rather than a bare 500.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from juniper_data_client import (
    JuniperDataConfigurationError,
    JuniperDataConnectionError,
    JuniperDataNotFoundError,
    JuniperDataTimeoutError,
    JuniperDataValidationError,
)

from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState

__all__ = ["get_state", "get_settings", "map_data_error"]


def get_state(request: Request) -> AppState:
    """FastAPI dependency: the per-app :class:`AppState` (uvicorn ``workers=1``)."""
    return request.app.state.app_state


def get_settings(request: Request) -> Settings:
    """FastAPI dependency: the per-app :class:`Settings`."""
    return request.app.state.settings


def map_data_error(exc: Exception) -> HTTPException:
    """Translate a data-fetch failure into an :class:`HTTPException`.

    * not-found → ``404``
    * connection / timeout → ``502`` (upstream juniper-data unreachable)
    * validation / contract (``ValueError``) → ``422``
    * misconfiguration → ``500``
    * anything else → ``502``
    """
    if isinstance(exc, JuniperDataNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, f"dataset not found: {exc}")
    if isinstance(exc, (JuniperDataConnectionError, JuniperDataTimeoutError)):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, f"juniper-data unreachable: {exc}")
    if isinstance(exc, (JuniperDataValidationError, ValueError)):
        # 422 as an int literal: Starlette deprecated HTTP_422_UNPROCESSABLE_ENTITY and the
        # renamed constant is absent on older fastapi>=0.110 resolutions; the literal is safe.
        return HTTPException(422, f"invalid dataset: {exc}")
    if isinstance(exc, JuniperDataConfigurationError):
        return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"data-client misconfigured: {exc}")
    return HTTPException(status.HTTP_502_BAD_GATEWAY, f"data fetch failed: {exc}")
