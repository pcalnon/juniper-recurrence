"""Custom exceptions for the juniper-recurrence client library.

Mirrors juniper-data-client's flat hierarchy (one base + typed leaves), adding a
``JuniperRecurrenceConflictError`` for the recurrence app's ``409`` responses (a training /
cross-validation run already in progress, or an operation that needs a trained model that does
not yet exist) — a status the data-client surface never returns.

That mirroring is the whole contract. The three Juniper clients are separately
released packages with no shared code, so nothing mechanical keeps them
aligned: no drift check can span them, and the alignment is a convention
carried by each package's tests and AGENTS.md. juniper-data-client#158 is the
reference implementation.
"""

from __future__ import annotations

from typing import Any


class JuniperRecurrenceClientError(Exception):
    """Base exception for all juniper-recurrence client errors.

    Carries the machine-readable context a caller needs to *act* on the error
    rather than re-parse its message (defect-register ``APD-RCLIENT-001``).
    Without ``status_code`` a 400 and a 422 raise the same type with the same
    text, so the only way to tell "you sent bad input" from "the service could
    not process it" was substring-matching the message.

    Every attribute is optional and keyword-only: locally raised errors
    (configuration, connection, timeout) have no HTTP response behind them, and
    existing call sites that pass only a message keep working unchanged.

    Attributes:
        message: The human-readable summary, also passed to ``Exception``.
        status_code: HTTP status of the originating response, when there was
            one. ``None`` for errors raised before or without a response.
        detail: The server's ``detail`` payload **exactly as decoded** -- a
            ``str`` for most handlers, and a ``list[dict]`` for FastAPI's 422
            validation errors. Deliberately not stringified: the structure is
            the point, and rendering it into the message is lossy.
        response: The originating ``requests.Response``, when available, for
            callers that need headers or the raw body.
    """

    def __init__(  # noqa: B042 — kwargs survive pickle via the default __reduce__; see below
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        detail: Any = None,
        response: Any = None,
    ) -> None:
        # B042 asks that an exception's ``__init__`` forward every argument to
        # ``super().__init__()`` and take no kwargs, so pickle and copy
        # round-trip. The concern is real but already answered by CPython:
        # ``BaseException.__reduce__`` returns ``(cls, args, self.__dict__)``
        # whenever the instance dict is non-empty, so the keyword-only context
        # is restored automatically -- as long as ``cls(*args)`` stays
        # constructible, which is why the ``super()`` call below forwards the
        # message and nothing else. B042's own remedy is not available here:
        # "take no kwargs" is precisely the defect this class closed
        # (``APD-RCLIENT-001``), and forwarding the extras to ``super()``
        # would put them in ``args``, making ``str(exc)`` a tuple repr and
        # the pickle rebuild a ``TypeError``
        # (``test_context_survives_pickle_and_copy`` pins the latter).
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.response = response


class JuniperRecurrenceConnectionError(JuniperRecurrenceClientError):
    """Raised when the connection to the juniper-recurrence service fails."""


class JuniperRecurrenceTimeoutError(JuniperRecurrenceClientError):
    """Raised when a request to the juniper-recurrence service times out."""


class JuniperRecurrenceNotFoundError(JuniperRecurrenceClientError):
    """Raised when a requested resource is not found (404)."""


class JuniperRecurrenceValidationError(JuniperRecurrenceClientError):
    """Raised when request parameters fail validation (400 / 422)."""


class JuniperRecurrenceConflictError(JuniperRecurrenceClientError):
    """Raised on a 409 Conflict — a training/cross-validation run is already in progress, or
    the operation requires a trained model/dataset that does not yet exist."""


class JuniperRecurrenceConfigurationError(JuniperRecurrenceClientError):
    """Raised when juniper-recurrence client configuration is missing or invalid."""
