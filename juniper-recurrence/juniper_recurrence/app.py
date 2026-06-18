"""FastAPI application assembly for the juniper-recurrence service.

Builds on the *as-built* ``juniper-service-core`` app factory: ``create_app``
mounts only the generic health router, so the owning service owns its
security / middleware stack (the as-built reconciliation, plan §2). This module
attaches that stack and exposes the module-level ``app`` that ``uvicorn`` (and the
CLI ``serve`` subcommand) import.

Middleware order mirrors the canonical cascor / canopy / data assembly —
``RequestBodyLimitMiddleware`` → ``SecurityHeadersMiddleware`` →
``SecurityMiddleware`` — added in that sequence so that, under Starlette's LIFO
execution, ``SecurityMiddleware`` (API-key auth + rate limiting) runs outermost.

The train / predict / model / dataset routers are threaded through
``create_app(routers=...)``; a fresh :class:`AppState` (the in-process model / result /
event holder) is created per ``build_app`` and stashed on ``app.state`` for the routers.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from juniper_service_core import (
    RequestBodyLimitMiddleware,
    SecurityHeadersMiddleware,
    SecurityMiddleware,
    build_api_key_auth,
    build_rate_limiter,
    create_app,
)

from juniper_recurrence._version import __version__
from juniper_recurrence.routers import crossval_router, dataset_router, model_router, predict_router, training_router
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState

__all__ = ["build_app", "app"]

logger = logging.getLogger(__name__)


def build_app(settings: Settings | None = None) -> FastAPI:
    """Assemble the juniper-recurrence FastAPI app.

    Args:
        settings: Pre-built settings (tests inject these to exercise auth /
            rate-limit configurations). Defaults to ``Settings()``, which reads
            the ``JUNIPER_RECURRENCE_`` environment namespace.

    Returns:
        A configured :class:`~fastapi.FastAPI` instance with the generic health
        router (from ``create_app``) plus the security / middleware stack.
    """
    settings = settings or Settings()

    application = create_app(
        title="Juniper Recurrence",
        version=__version__,
        routers=(training_router, predict_router, model_router, dataset_router, crossval_router),
    )

    # Middleware (Starlette LIFO: last added runs first on the request path). This
    # exact order is the de-cascored canonical assembly shared by cascor / canopy /
    # data: body-limit innermost, security-headers next, API-key auth + rate-limit
    # outermost.
    application.add_middleware(RequestBodyLimitMiddleware)
    application.add_middleware(SecurityHeadersMiddleware)

    # Observability: Prometheus /metrics (IP-allowlist gated) + build-info + HTTP
    # request metrics. Optional — guarded so the app still runs without the
    # [observability] extra. Added BEFORE SecurityMiddleware so Security stays the
    # outermost handler (canonical order): unauthenticated / rate-limited requests are
    # rejected before PrometheusMiddleware runs, so they add no metric cardinality.
    # /metrics is exempt from SecurityMiddleware's API-key check via service-core's
    # EXEMPT_PATHS (SEC-16); MetricsAuthMiddleware IP-gates it instead.
    if settings.metrics_enabled:
        try:
            from juniper_observability import (
                MetricsAuthMiddleware,
                PrometheusMiddleware,
                get_prometheus_app,
                set_build_info,
            )
        except ImportError:
            logger.warning(
                "metrics_enabled is true but juniper-observability is not installed; "
                "/metrics will not be mounted (install the [observability] extra)."
            )
        else:
            application.add_middleware(PrometheusMiddleware, service_name=settings.service_name)
            # Prometheus metric names are underscore-only; the build-info namespace is the
            # underscored service name -> the metric is ``juniper_recurrence_build_info``.
            set_build_info(settings.service_name.replace("-", "_"), __version__)
            application.mount(
                "/metrics",
                MetricsAuthMiddleware(get_prometheus_app(), trusted_ips=settings.metrics_trusted_ips),
            )

    api_key_auth = build_api_key_auth(settings.resolve_api_keys())
    rate_limiter = build_rate_limiter(
        requests_per_minute=settings.rate_limit_requests_per_minute,
        enabled=settings.rate_limit_enabled,
    )
    application.add_middleware(SecurityMiddleware, api_key_auth=api_key_auth, rate_limiter=rate_limiter)

    # Stash per-app instances for routers / tests (mirrors cascor's app.state usage).
    # AppState is created fresh per build_app, so each app — and each test — gets
    # isolated in-process model / result / event state.
    application.state.settings = settings
    application.state.api_key_auth = api_key_auth
    application.state.app_state = AppState()

    return application


# Module-level ASGI app for ``uvicorn juniper_recurrence.app:app`` (CLI ``serve``).
app = build_app()
