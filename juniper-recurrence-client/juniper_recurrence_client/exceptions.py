"""Custom exceptions for the juniper-recurrence client library.

Mirrors juniper-data-client's flat hierarchy (one base + typed leaves), adding a
``JuniperRecurrenceConflictError`` for the recurrence app's ``409`` responses (a training /
cross-validation run already in progress, or an operation that needs a trained model that does
not yet exist) — a status the data-client surface never returns.
"""


class JuniperRecurrenceClientError(Exception):
    """Base exception for all juniper-recurrence client errors."""


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
