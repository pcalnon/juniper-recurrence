"""juniper-recurrence — FastAPI + CLI service for the Δt-native LMU recurrence model.

The application layer (WS-4b) that wraps the already-shipped ``LMURegressor``
(``juniper-recurrence-model``, WS-4a) on the ``juniper-service-core`` framework
(WS-2), fed 3-D windowed sequences (``equities_seq``) via ``juniper-data-client``
(WS-1). It is the first real consumer of service-core's ``create_app`` +
``TrainingLifecycle`` (the 2nd-implementer proof for the *service* contract).

Only :data:`__version__` is exposed at the top level, kept dependency-free so the
package version is importable without pulling fastapi / pydantic-settings. The
FastAPI app and its factory live in :mod:`juniper_recurrence.app`
(``from juniper_recurrence.app import app, build_app``); the CLI entrypoint is
:func:`juniper_recurrence.main.main`.

Design of record: ``notes/JUNIPER_RECURRENCE_WS4B_APP_BUILD_PLAN_2026-06-15.md`` and
``notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`` (juniper-ml).
"""

from juniper_recurrence._version import __version__

__all__ = ["__version__"]
