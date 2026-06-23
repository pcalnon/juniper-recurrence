"""TST-7: best-effort ``X-Request-ID`` propagation in ``JuniperRecurrenceClient._request``.

The client attaches an ``X-Request-ID`` header from juniper-observability's
``request_id_var`` ContextVar when one is set, but only as a best-effort no-op: a missing
juniper-observability (ImportError) or an unset/empty contextvar must leave the request
untouched, and a caller-supplied header always wins. These four branches
(``client.py`` _request) had no test referencing ``request_id_var``.
"""

from __future__ import annotations

import sys

import pytest
import responses

from juniper_recurrence_client import JuniperRecurrenceClient
from juniper_recurrence_client.constants import ENDPOINT_HEALTH

BASE_URL = "http://recurrence.test:8211"
_HEADER = "X-Request-ID"


def _client(**kwargs: object) -> JuniperRecurrenceClient:
    kwargs.setdefault("retries", 0)
    return JuniperRecurrenceClient(base_url=BASE_URL, **kwargs)


def _add_health() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/health", json={"status": "ok"}, status=200)


@responses.activate
def test_no_request_id_header_when_observability_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None entry in sys.modules makes ``import juniper_observability`` raise ImportError,
    # which _request must swallow (graceful no-op, no header attached).
    monkeypatch.setitem(sys.modules, "juniper_observability", None)
    _add_health()
    _client().health_check()
    assert _HEADER not in responses.calls[0].request.headers


@responses.activate
def test_request_id_attached_from_contextvar(monkeypatch: pytest.MonkeyPatch) -> None:
    obs = pytest.importorskip("juniper_observability")
    token = obs.request_id_var.set("rid-abc-123")
    try:
        _add_health()
        _client().health_check()
    finally:
        obs.request_id_var.reset(token)
    assert responses.calls[0].request.headers[_HEADER] == "rid-abc-123"


@responses.activate
def test_no_request_id_header_when_contextvar_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = pytest.importorskip("juniper_observability")
    token = obs.request_id_var.set("")  # falsy -> the ``if rid:`` guard skips
    try:
        _add_health()
        _client().health_check()
    finally:
        obs.request_id_var.reset(token)
    assert _HEADER not in responses.calls[0].request.headers


@responses.activate
def test_caller_supplied_request_id_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    obs = pytest.importorskip("juniper_observability")
    token = obs.request_id_var.set("rid-from-contextvar")
    try:
        _add_health()
        # The public verbs don't expose headers, so exercise the caller-wins branch
        # (``if "X-Request-ID" not in headers``) through _request directly.
        client = _client()
        client._request("GET", ENDPOINT_HEALTH, headers={_HEADER: "caller-explicit"})
    finally:
        obs.request_id_var.reset(token)
    assert responses.calls[0].request.headers[_HEADER] == "caller-explicit"
