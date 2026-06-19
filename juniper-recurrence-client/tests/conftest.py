"""Shared fixtures for the juniper-recurrence-client test suite."""

from __future__ import annotations

import pytest

from juniper_recurrence_client.constants import API_KEY_ENV_VAR, API_KEY_FILE_ENV_VAR


@pytest.fixture(autouse=True)
def _clean_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the client API-key env vars so auth tests start from a known state."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    monkeypatch.delenv(API_KEY_FILE_ENV_VAR, raising=False)
