"""juniper-recurrence-client — HTTP client for the juniper-recurrence service.

A lean ``requests``-based client wrapping the juniper-recurrence FastAPI app's REST surface
(train / predict / cross-validate / inspect / health). Mirrors juniper-data-client and
juniper-cascor-client so consumers (notably juniper-canopy's recurrence backend adapter) drive
every Juniper backend the same way.
"""

from __future__ import annotations

from juniper_recurrence_client._version import __version__
from juniper_recurrence_client.client import JuniperRecurrenceClient, RequestHook
from juniper_recurrence_client.exceptions import (
    JuniperRecurrenceClientError,
    JuniperRecurrenceConfigurationError,
    JuniperRecurrenceConflictError,
    JuniperRecurrenceConnectionError,
    JuniperRecurrenceNotFoundError,
    JuniperRecurrenceTimeoutError,
    JuniperRecurrenceValidationError,
)

__all__ = [
    "__version__",
    "JuniperRecurrenceClient",
    "RequestHook",
    "JuniperRecurrenceClientError",
    "JuniperRecurrenceConnectionError",
    "JuniperRecurrenceTimeoutError",
    "JuniperRecurrenceNotFoundError",
    "JuniperRecurrenceConflictError",
    "JuniperRecurrenceValidationError",
    "JuniperRecurrenceConfigurationError",
]
