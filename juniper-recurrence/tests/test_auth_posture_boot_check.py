"""Regression tests for the SEC-F01 boot-time auth-posture self-check (HO-2 class).

The app lifespan calls juniper-service-core's ``enforce_auth_posture(...,
require_auth=False, service_name="juniper-recurrence")`` at startup — before
serving — so an empty/blank ``JUNIPER_RECURRENCE_API_KEYS`` secret (which
silently disables ``APIKeyAuth`` and serves the API open behind a healthy
health check) is at least LOUD at boot. ``require_auth`` stays ``False`` until
the owner-approved ``JUNIPER_RECURRENCE_REQUIRE_AUTH`` follow-up flips the
posture to fail-closed.

The wiring test monkeypatches the module attribute and drives the real lifespan
via ``TestClient`` (entering the client context runs startup); the behavioural
tests exercise the helper directly, independent of ``init_logging``.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from juniper_recurrence import app as app_module
from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings


def test_lifespan_calls_posture_check_with_resolved_keys(monkeypatch):
    """The lifespan must invoke enforce_auth_posture once, with the settings'
    resolved keys, require_auth=False, and the service's own name."""
    calls: list[tuple[list[str], bool, str]] = []

    def _recorder(api_keys, *, require_auth, service_name, logger=None, **_kwargs):
        calls.append((list(api_keys or []), require_auth, service_name))

    monkeypatch.setattr(app_module, "enforce_auth_posture", _recorder)
    settings = Settings(api_keys=["k1", "k2"])
    with TestClient(build_app(settings)):
        pass
    assert calls == [(settings.resolve_api_keys(), False, "juniper-recurrence")]


def test_no_keys_and_not_required_warns_open(caplog):
    """The HO-2 class made loud: no real key + require_auth=False logs a WARNING
    naming the service, and does NOT raise (the service still starts)."""
    from juniper_service_core import enforce_auth_posture

    with caplog.at_level(logging.WARNING):
        enforce_auth_posture([], require_auth=False, service_name="juniper-recurrence")
    assert any("running OPEN" in rec.getMessage() and "juniper-recurrence" in rec.getMessage() for rec in caplog.records)


def test_blank_key_counts_as_unset(caplog):
    """An empty/whitespace key — exactly what an empty secret file resolves to —
    is NOT real auth: the check must report an open posture, not a secured one."""
    from juniper_service_core import auth_is_configured, enforce_auth_posture

    assert not auth_is_configured([""])
    assert not auth_is_configured(["   "])
    with caplog.at_level(logging.WARNING):
        enforce_auth_posture(["   "], require_auth=False, service_name="juniper-recurrence")
    assert any("running OPEN" in rec.getMessage() for rec in caplog.records)


def test_required_with_no_key_raises():
    """The fail-closed posture the follow-up flag will enable: require_auth=True
    with no real key raises AuthPostureError (failing uvicorn's startup)."""
    from juniper_service_core import AuthPostureError, enforce_auth_posture

    with pytest.raises(AuthPostureError):
        enforce_auth_posture([], require_auth=True, service_name="juniper-recurrence")


def test_escape_hatch_bypasses_the_check(monkeypatch):
    """JUNIPER_SKIP_AUTH_POSTURE_CHECK=1 bypasses the check even when it would raise."""
    from juniper_service_core import enforce_auth_posture

    monkeypatch.setenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", "1")
    enforce_auth_posture([], require_auth=True, service_name="juniper-recurrence")
