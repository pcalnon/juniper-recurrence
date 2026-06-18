"""Tests for the Prometheus ``/metrics`` endpoint.

Design: ``notes/JUNIPER_RECURRENCE_METRICS_ENDPOINT_DESIGN_2026-06-18.md`` (juniper-ml).

``/metrics`` is mounted only when ``metrics_enabled`` and ``juniper-observability`` is
installed; it is IP-allowlist gated by ``MetricsAuthMiddleware`` (reads ``scope["client"]``)
and exempt from the API-key ``SecurityMiddleware`` via service-core's ``EXEMPT_PATHS`` (SEC-16).

The whole module ``importorskip``s ``juniper_observability`` so it is a no-op for contributors
without the optional ``[observability]`` extra; CI installs ``.[test,observability]`` so it runs.
``MetricsAuthMiddleware`` matches on the ASGI peer address, so the trusted-path tests must spoof a
real client IP via ``TestClient(app, client=("127.0.0.1", ...))`` (Starlette's default
``"testclient"`` host is unparseable and always denied). Collector re-registration across the
repeated ``build_app`` calls is handled by observability's idempotent ``register_or_reuse`` helpers.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings

pytest.importorskip("juniper_observability")


def _client(settings: Settings, *, host: str = "127.0.0.1") -> TestClient:
    """A TestClient whose ASGI peer is ``host`` (so MetricsAuthMiddleware can gate it)."""
    return TestClient(build_app(settings), client=(host, 12345), raise_server_exceptions=False)


def test_metrics_trusted_ip_scrape_ok():
    resp = _client(Settings()).get("/metrics")  # default allowlist = loopback; peer 127.0.0.1
    assert resp.status_code == 200, resp.text
    assert "text/plain" in resp.headers["content-type"]
    assert "juniper_recurrence_build_info" in resp.text


def test_metrics_untrusted_ip_forbidden():
    # 127.0.0.1 is not in 10.0.0.0/8 -> MetricsAuthMiddleware denies.
    resp = _client(Settings(metrics_trusted_ips=["10.0.0.0/8"])).get("/metrics")
    assert resp.status_code == 403


def test_metrics_exempt_from_api_key_auth():
    # API-key auth is on, but /metrics is in service-core EXEMPT_PATHS -> no key required.
    resp = _client(Settings(api_keys=["secret-key"])).get("/metrics")  # no X-API-Key sent
    assert resp.status_code != 401
    assert resp.status_code == 200


def test_metrics_disabled_returns_404():
    resp = _client(Settings(metrics_enabled=False)).get("/metrics")
    assert resp.status_code == 404


def test_http_requests_total_recorded():
    client = _client(Settings())
    assert client.get("/v1/health").status_code == 200  # a request for the middleware to record
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "juniper_http_requests_total" in resp.text


def test_invalid_trusted_ip_rejected_at_settings():
    # The fail-loud field validator mirrors MetricsAuthMiddleware's parsing.
    with pytest.raises(ValueError, match="metrics_trusted_ips"):
        Settings(metrics_trusted_ips=["172.18.0.0/164"])


def test_metrics_graceful_when_observability_absent(monkeypatch):
    # Simulate the [observability] extra being absent: the guarded import raises,
    # build_app must still succeed, and /metrics is simply not mounted.
    monkeypatch.setitem(sys.modules, "juniper_observability", None)
    app = build_app(Settings(metrics_enabled=True))  # must not raise
    client = TestClient(app, client=("127.0.0.1", 12345), raise_server_exceptions=False)
    assert client.get("/metrics").status_code == 404
    assert client.get("/v1/health").status_code == 200  # the rest of the app still serves
