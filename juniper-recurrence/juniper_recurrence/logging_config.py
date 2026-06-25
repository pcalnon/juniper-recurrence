"""Application logging configuration for the juniper-recurrence service.

Closes the audit's H1 finding: the service previously never configured logging, so
``settings.log_level`` was read by nothing and the app emitted no operational log lines.

:func:`init_logging` is the single entrypoint the CLI ``serve`` command calls at startup.
It prefers the shared :func:`juniper_observability.configure_logging` (structured-JSON with
``request_id`` correlation, the ecosystem norm) when the optional ``[observability]`` extra is
installed, and otherwise falls back to a stdlib :func:`logging.basicConfig` so logging still
works without the extra (the app and CLI run without ``juniper-observability`` installed).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from juniper_recurrence.settings import Settings

__all__ = ["init_logging"]

# Plain-text fallback format; mirrors juniper_observability.DEFAULT_LOG_FORMAT_PLAIN so output
# is consistent whether or not the [observability] extra is installed.
_PLAIN_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def init_logging(settings: Settings) -> None:
    """Configure root logging from ``settings`` (replaces existing handlers; call once at startup).

    Uses the shared structured-JSON logger when ``juniper-observability`` is installed, else a
    stdlib plain-text handler. The level comes from ``settings.log_level`` (unknown values fall
    back to ``INFO``); the format from ``settings.log_format`` (``"json"`` vs plain text).
    """
    try:
        from juniper_observability import configure_logging
    except ImportError:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, format=_PLAIN_FORMAT, force=True)
    else:
        configure_logging(settings.log_level, settings.log_format, settings.service_name)
