"""Shared router dependencies and error mapping for the juniper-recurrence API.

The app-state and settings dependencies read the per-app instances stashed on
``app.state`` by :func:`juniper_recurrence.app.build_app`. :func:`map_data_error`
translates juniper-data-client failures into the appropriate HTTP status so the
train / predict data path returns ``404`` / ``422`` / ``502`` rather than a bare 500.

:class:`RecordedOperation` and :func:`require_expected_operation` are the route half of
operation identity (W1.5; see :mod:`juniper_recurrence.state`).
"""

from __future__ import annotations

from types import TracebackType
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from juniper_data_client import (
    JuniperDataConfigurationError,
    JuniperDataConnectionError,
    JuniperDataNotFoundError,
    JuniperDataTimeoutError,
    JuniperDataValidationError,
)

from juniper_recurrence.schemas import OperationMismatchDetail
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState, Operation

__all__ = ["get_state", "get_settings", "map_data_error", "RecordedOperation", "RequestIdHeader", "require_expected_operation"]

#: The ``X-Request-ID`` header of a lock-taking request (train, restore), recorded verbatim as its
#: operation's ``requested_by`` (W1.5). A caller whose request times out client-side has no
#: ``operation_id`` -- the response never arrived -- so it finds its own operation in
#: ``GET /v1/training/status`` by the id it sent.
RequestIdHeader = Annotated[str | None, Header(description="Optional caller correlation id. Recorded verbatim as the operation's requested_by in GET /v1/training/status, so a caller whose request timed out can find its own operation there.")]


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


class RecordedOperation:
    """Run one lock-holding operation so that it always leaves a truthful terminal record.

    Before W1.5 a fit that raised left nothing behind: the route's ``finally`` released the
    lock, and ``GET /v1/training/status`` went on describing whatever ran before. Used as
    ``with RecordedOperation(state, operation):`` around the whole operation, this records how
    it ended and releases ``train_lock`` (:meth:`AppState.end`), and **never** swallows or
    remaps the exception -- the route's own error mapping is untouched:

    * an :class:`HTTPException` is recorded with the detail and status it is about to return;
    * anything else is recorded as ``"<Type>: <message>"`` with ``500``, which is what the
      unhandled exception becomes on the wire (no exception handler is installed), and is
      re-raised as-is;
    * success records nothing here -- the route publishes the model, and with it the outcome,
      through ``set_trained`` / ``set_restored``.
    """

    def __init__(self, state: AppState, operation: Operation) -> None:
        self._state = state
        self.operation = operation

    def __enter__(self) -> Operation:
        return self.operation

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        try:
            if isinstance(exc, HTTPException):
                self._state.fail(self.operation, detail=str(exc.detail), status_code=exc.status_code)
            elif exc is not None:
                self._state.fail(self.operation, detail=f"{type(exc).__name__}: {exc}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            self._state.end(self.operation)
        return False  # never suppress: the exception, if any, propagates exactly as before


def require_expected_operation(expected: str | None, actual: str | None) -> None:
    """Refuse with ``409`` when a caller's ``expect_operation_id`` does not name the model's operation.

    ``expected`` is the request's ``expect_operation_id`` (``None`` = the caller asked for no
    check, the pre-W1.5 behaviour); ``actual`` is the operation that produced the in-memory model,
    read together with the model (:meth:`AppState.model_with_operation`). The ``409`` detail names
    both ids, so the refused caller learns whose model it would have acted on (F-CON1 / F-CON2).
    """
    if expected is None or expected == actual:
        return
    raise HTTPException(
        status.HTTP_409_CONFLICT,
        detail=OperationMismatchDetail(
            message=f"expect_operation_id {expected} does not match the operation that produced the in-memory model ({actual}); another caller may have trained or restored since",
            expected_operation_id=expected,
            model_operation_id=actual,
        ).model_dump(),
    )
