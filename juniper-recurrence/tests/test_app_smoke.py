"""Smoke tests for the juniper-recurrence app assembly (WS-4b PR-1, plan §12).

Covers the load-bearing PR-1 properties: ``build_app`` returns a FastAPI app, the
generic health router is mounted and exempt from auth, the canonical middleware
stack is attached (and active), and security headers are injected. The protected
train / predict / model / dataset routes arrive in PR-2; here we exercise the
security middleware against an (unmounted) protected path.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from juniper_recurrence import __version__
from juniper_recurrence.app import app as module_app
from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings


def test_build_app_returns_fastapi():
    app = build_app()
    assert isinstance(app, FastAPI)
    assert app.title == "Juniper Recurrence"
    assert app.version == __version__


def test_module_level_app_is_fastapi():
    # The object uvicorn imports via ``juniper_recurrence.app:app``.
    assert isinstance(module_app, FastAPI)


def test_health_endpoints_ok_and_exempt():
    client = TestClient(build_app(Settings(api_keys=None)))

    live = client.get("/v1/health")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}

    ready = client.get("/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_security_headers_attached():
    client = TestClient(build_app())
    response = client.get("/v1/health")
    # SecurityHeadersMiddleware injects these on every response.
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "content-security-policy" in response.headers


def test_middleware_stack_attached_in_canonical_order():
    app = build_app()
    middleware_classes = [m.cls.__name__ for m in app.user_middleware]
    # All three service-core middlewares are present.
    assert "RequestBodyLimitMiddleware" in middleware_classes
    assert "SecurityHeadersMiddleware" in middleware_classes
    assert "SecurityMiddleware" in middleware_classes
    # Starlette stores user_middleware in reverse add-order (last added first), so
    # SecurityMiddleware (added last) is the outermost handler.
    assert middleware_classes[0] == "SecurityMiddleware"


def test_auth_enabled_protects_unmounted_paths_but_not_health():
    # With API keys configured, SecurityMiddleware is active. Health stays exempt;
    # any non-exempt path requires a valid X-API-Key (PR-2 mounts real protected
    # routes — here a 404 path still exercises the auth gate).
    client = TestClient(build_app(Settings(api_keys=["s3cret"])))

    assert client.get("/v1/health").status_code == 200  # exempt

    missing_key = client.get("/v1/does-not-exist")
    assert missing_key.status_code == 401  # auth runs before routing

    with_key = client.get("/v1/does-not-exist", headers={"X-API-Key": "s3cret"})
    assert with_key.status_code == 404  # passed auth, no such route

    bad_key = client.get("/v1/does-not-exist", headers={"X-API-Key": "wrong"})
    assert bad_key.status_code == 401


def test_auth_disabled_by_default_open_access():
    # No api_keys -> APIKeyAuth disabled -> non-exempt path is reachable (404, not 401).
    client = TestClient(build_app(Settings(api_keys=None)))
    assert client.get("/v1/does-not-exist").status_code == 404


def test_docs_reachable_and_exempt():
    client = TestClient(build_app(Settings(api_keys=["s3cret"])))
    # /docs and /openapi.json are in the service-core EXEMPT_PATHS set.
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
