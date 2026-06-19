"""API-key resolution tests (explicit arg, env var, ``_FILE`` Docker-secret, precedence)."""

from __future__ import annotations

import pytest

from juniper_recurrence_client import JuniperRecurrenceClient
from juniper_recurrence_client.constants import API_KEY_ENV_VAR, API_KEY_FILE_ENV_VAR, API_KEY_HEADER_NAME

BASE_URL = "http://x:8211"


def test_explicit_api_key_sets_header() -> None:
    client = JuniperRecurrenceClient(base_url=BASE_URL, api_key="secret-key")
    assert client.session.headers[API_KEY_HEADER_NAME] == "secret-key"


def test_no_api_key_leaves_header_unset() -> None:
    client = JuniperRecurrenceClient(base_url=BASE_URL)
    assert API_KEY_HEADER_NAME not in client.session.headers


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    client = JuniperRecurrenceClient(base_url=BASE_URL)
    assert client.session.headers[API_KEY_HEADER_NAME] == "env-key"


def test_file_indirection_beats_plain_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("  file-key\n", encoding="utf-8")
    monkeypatch.setenv(API_KEY_FILE_ENV_VAR, str(key_file))
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    client = JuniperRecurrenceClient(base_url=BASE_URL)
    assert client.session.headers[API_KEY_HEADER_NAME] == "file-key"


def test_explicit_api_key_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    client = JuniperRecurrenceClient(base_url=BASE_URL, api_key="explicit")
    assert client.session.headers[API_KEY_HEADER_NAME] == "explicit"


def test_empty_file_falls_back_to_plain_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    monkeypatch.setenv(API_KEY_FILE_ENV_VAR, str(empty))
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    client = JuniperRecurrenceClient(base_url=BASE_URL)
    assert client.session.headers[API_KEY_HEADER_NAME] == "env-key"
