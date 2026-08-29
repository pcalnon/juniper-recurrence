"""Consumer-shaped type probe for ``juniper_recurrence_client`` (defect register ``APD-ECO-006``).

Project:     Juniper
Sub-Project: juniper-recurrence-client
Application: published type-surface probe
Author:      Paul Calnon
License:     MIT License

WHY THIS FILE EXISTS, AND WHY IT LIVES **OUTSIDE** THE PACKAGE
-------------------------------------------------------------
This repo's mypy hook (``.pre-commit-config.yaml``, at the MONOREPO ROOT rather than beside the
package) was scoped ``^juniper-recurrence-client/juniper_recurrence_client/.*\\.py$`` --
library-internal source only. Nothing type-checked a file that *imports* the package the way a
consumer does, so the published type surface was never verified as usable from outside. That is
``APD-ECO-006``: "no client type-checks a consumer-shaped probe."

The register originally recorded this client as configuring "no mypy at all"; that was wrong, and the
correction is instructive -- ``juniper-recurrence-client`` is a sub-package of the ``juniper-recurrence``
monorepo, so its config lives two levels up at the repo root and a per-repo sweep concluded it was
absent (juniper-ml#1449). It is in fact checked **strictly**, which is why this probe must satisfy
strict mode: every definition fully annotated, no implicit ``Any``.

WHAT THIS CATCHES
-----------------
* a public name in ``__all__`` that does not resolve for an importer;
* a public method whose annotation is missing, wrong, or silently ``Any`` at the boundary;
* an error class that stops deriving from the package base, silently breaking every consumer's
  ``except JuniperRecurrenceClientError``.

WHAT THIS DOES **NOT** CATCH -- stated so it is not mistaken for a guarantee
---------------------------------------------------------------------------
mypy resolves ``juniper_recurrence_client`` from the **source tree** here, not from an installed
distribution, so ``py.typed`` is bypassed entirely. A wheel that shipped without ``py.typed`` would
give a real consumer an untyped package while this probe still passed -- the ``APD-SVCCORE-008`` /
``APD-OBS-002`` class. Catching that needs a check against the built artifact, and is a separate
concern from this row.

This file is never imported at runtime. The *type check* is the test, so it must stay import-clean.
"""

from __future__ import annotations

from typing import Any

from juniper_recurrence_client import (
    JuniperRecurrenceClient,
    JuniperRecurrenceClientError,
    JuniperRecurrenceConfigurationError,
    JuniperRecurrenceConflictError,
    JuniperRecurrenceConnectionError,
    JuniperRecurrenceNotFoundError,
    JuniperRecurrenceTimeoutError,
    JuniperRecurrenceValidationError,
)


def probe_client_surface(client: JuniperRecurrenceClient) -> None:
    """Exercise the public methods a consumer actually calls, checking each declared return type."""
    health: dict[str, Any] = client.health_check()
    status: dict[str, Any] = client.training_status()
    prediction: dict[str, Any] = client.predict(dataset_id="some-dataset", split="test")

    # Read a value back off each result, so a return silently widened away from ``dict`` does not
    # pass unnoticed the way a bare call would.
    _status_field: Any = health.get("status")
    _state: Any = status.get("state")
    _values: Any = prediction.get("predictions")


def probe_per_call_timeout_override(client: JuniperRecurrenceClient) -> dict[str, Any]:
    """``predict`` must remain keyword-only and accept the documented consumer arguments.

    ``APD-RCLIENT-002`` shipped a per-call timeout override on this client, and ``APD-ECO-003``
    tracks the fact that its siblings still lack one. Pinning the keyword-only call shape here means
    a refactor that changes the published signature is caught against a consumer rather than only
    against the library's own callers.
    """
    return client.predict(dataset_id="d", name="n", split="train")


def probe_exception_hierarchy() -> None:
    """Every published error must be catchable through the package's base error.

    A consumer writes ``except JuniperRecurrenceClientError``. If a subclass ever stops deriving from
    it, that consumer silently stops catching it -- a failure with no runtime signal until the
    exception escapes in production.
    """
    for derived in (
        JuniperRecurrenceConfigurationError,
        JuniperRecurrenceConflictError,
        JuniperRecurrenceConnectionError,
        JuniperRecurrenceNotFoundError,
        JuniperRecurrenceTimeoutError,
        JuniperRecurrenceValidationError,
    ):
        _derived: type[JuniperRecurrenceClientError] = derived
    _base_is_exception: type[Exception] = JuniperRecurrenceClientError
