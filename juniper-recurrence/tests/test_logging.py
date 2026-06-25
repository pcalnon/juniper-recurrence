"""Logging-configuration tests for the juniper-recurrence app (audit H1).

The service previously never configured logging. These assert that ``init_logging``
configures the root logger from settings (level + handler), prefers the shared
structured-JSON formatter when ``juniper-observability`` is installed and falls back to
stdlib logging when it is not, that the app's lifespan configures logging on startup
(via ``create_app(lifespan=)``), and that the training router emits operational log lines.
"""

from __future__ import annotations

import importlib.util
import logging
import sys

import pytest
from fastapi.testclient import TestClient

from juniper_recurrence.app import build_app
from juniper_recurrence.logging_config import init_logging
from juniper_recurrence.settings import Settings

_HAS_OBSERVABILITY = importlib.util.find_spec("juniper_observability") is not None


@pytest.fixture
def _restore_root_logging():
    """Snapshot/restore the root logger so these tests don't leak global logging state."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_init_logging_sets_level_and_installs_handler(_restore_root_logging):
    init_logging(Settings(log_level="WARNING"))
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert root.handlers, "init_logging must install at least one handler"


@pytest.mark.skipif(not _HAS_OBSERVABILITY, reason="structured-JSON formatter needs the [observability] extra")
def test_init_logging_json_uses_shared_formatter(_restore_root_logging):
    from juniper_observability import JuniperJsonFormatter

    init_logging(Settings(log_level="INFO", log_format="json"))
    root = logging.getLogger()
    assert any(isinstance(handler.formatter, JuniperJsonFormatter) for handler in root.handlers)


def test_init_logging_falls_back_without_observability(monkeypatch, _restore_root_logging):
    # Blocking the import (sys.modules[name] = None) makes ``import juniper_observability`` raise
    # ImportError — the [observability]-extra-absent path — so init_logging uses stdlib logging.
    monkeypatch.setitem(sys.modules, "juniper_observability", None)
    init_logging(Settings(log_level="DEBUG"))
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert root.handlers


def test_app_lifespan_configures_logging(monkeypatch, _restore_root_logging):
    # Logging is configured by the app's FastAPI lifespan (create_app(lifespan=)) at startup —
    # not in the `serve` CLI. Entering the TestClient context runs the lifespan startup.
    calls: list[str] = []
    monkeypatch.setattr("juniper_recurrence.app.init_logging", lambda settings: calls.append("init"))
    with TestClient(build_app(Settings(api_keys=None))):
        pass
    assert calls == ["init"], "the app lifespan must configure logging on startup"


def test_train_route_logs_run_lifecycle(fake_data, caplog):
    with caplog.at_level(logging.INFO, logger="juniper_recurrence.routers.training"):
        resp = TestClient(build_app(Settings(api_keys=None))).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    assert resp.status_code == 200, resp.text
    messages = [record.getMessage() for record in caplog.records]
    assert any("training start" in message for message in messages)
    assert any("training complete" in message for message in messages)


def test_train_409_logs_warning(fake_data, caplog):
    app = build_app(Settings(api_keys=None))
    app.state.app_state.train_lock.acquire()  # simulate an in-progress run
    try:
        with caplog.at_level(logging.WARNING, logger="juniper_recurrence.routers.training"):
            resp = TestClient(app).post("/v1/train", json={"dataset": {"dataset_id": "x"}})
        assert resp.status_code == 409
        assert any("already in progress" in record.getMessage() for record in caplog.records)
    finally:
        app.state.app_state.train_lock.release()
