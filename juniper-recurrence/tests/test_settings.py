"""Settings tests for the juniper-recurrence app (WS-4b PR-1, plan §7 / §12).

Asserts the env-prefix binding, the Docker ``_FILE`` secret indirection, the
``api_keys`` CSV / JSON / list normalisation (including the ``NoDecode`` env path),
the shared ``JUNIPER_DATA_URL`` alias, and — the recorded incident — that no local
``.env`` leaks into the settings (no ``env_file=`` is configured).
"""

from __future__ import annotations

import pytest

from juniper_recurrence.settings import Settings

# Ambient ``JUNIPER_*`` env vars are cleared by the autouse fixture in conftest.py,
# so every test below sees the documented defaults.


def test_defaults():
    settings = Settings()
    assert settings.service_name == "juniper-recurrence"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8210
    assert settings.api_keys is None
    assert settings.resolve_api_keys() == []
    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_requests_per_minute == 60
    assert settings.juniper_data_url == "http://localhost:8100"
    assert settings.juniper_data_api_key is None
    assert settings.default_d == 16
    assert settings.default_theta is None
    assert settings.default_ridge == 0.0
    assert settings.log_level == "INFO"  # inherited from SettingsBase
    assert settings.log_format == "text"


def test_default_ridge_accepts_gcv():
    """default_ridge widens to float | Literal['gcv'] (DP-3 P1); the 0.0 default is unchanged."""
    assert Settings(default_ridge="gcv").default_ridge == "gcv"
    assert Settings().default_ridge == 0.0


def test_env_prefix_honored(monkeypatch):
    monkeypatch.setenv("JUNIPER_RECURRENCE_PORT", "8888")
    monkeypatch.setenv("JUNIPER_RECURRENCE_HOST", "127.0.0.1")
    monkeypatch.setenv("JUNIPER_RECURRENCE_RATE_LIMIT_REQUESTS_PER_MINUTE", "5")
    settings = Settings()
    assert settings.port == 8888
    assert settings.host == "127.0.0.1"
    assert settings.rate_limit_requests_per_minute == 5


def test_api_keys_programmatic_forms():
    assert Settings(api_keys="k1,k2").resolve_api_keys() == ["k1", "k2"]
    assert Settings(api_keys=" a , b ,c ").resolve_api_keys() == ["a", "b", "c"]
    assert Settings(api_keys='["x","y"]').resolve_api_keys() == ["x", "y"]
    assert Settings(api_keys=["p", "q"]).resolve_api_keys() == ["p", "q"]
    assert Settings(api_keys=None).resolve_api_keys() == []
    assert Settings(api_keys="   ").resolve_api_keys() == []
    assert Settings(api_keys=[]).resolve_api_keys() == []


def test_api_keys_bracketed_but_invalid_json_falls_back_to_csv():
    # A '[...]'-wrapped but non-JSON api_keys value degrades to a CSV split instead of raising
    # (settings.py:122-123): json.loads fails -> parsed=None -> the raw text is CSV-split. This is
    # the secret-file hardening (a malformed JSON-looking payload never raises a ValidationError).
    assert Settings(api_keys="[k1, k2]").resolve_api_keys() == ["[k1", "k2]"]


def test_api_keys_non_str_non_list_passthrough_rejected():
    # A non-str / non-list api_keys value passes through the field validator unchanged
    # (settings.py:130) and is then rejected by the field's list[str] | None type check.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(api_keys=123)


def test_api_keys_csv_env_var(monkeypatch):
    # NoDecode keeps pydantic-settings from JSON-decoding the env value, so a plain
    # CSV payload reaches the field validator instead of raising ValidationError.
    monkeypatch.setenv("JUNIPER_RECURRENCE_API_KEYS", "e1, e2 ,e3")
    assert Settings().resolve_api_keys() == ["e1", "e2", "e3"]


def test_api_keys_json_env_var(monkeypatch):
    monkeypatch.setenv("JUNIPER_RECURRENCE_API_KEYS", '["j1","j2"]')
    assert Settings().resolve_api_keys() == ["j1", "j2"]


def test_api_keys_file_indirection(tmp_path, monkeypatch):
    secret = tmp_path / "api_keys.txt"
    secret.write_text("file-key-1,file-key-2\n")
    monkeypatch.setenv("JUNIPER_RECURRENCE_API_KEYS_FILE", str(secret))
    settings = Settings()
    assert settings.resolve_api_keys() == ["file-key-1", "file-key-2"]


def test_juniper_data_api_key_file_indirection(tmp_path, monkeypatch):
    secret = tmp_path / "data_key.txt"
    secret.write_text("  outbound-key  \n")  # get_secret strips surrounding whitespace
    monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(secret))
    settings = Settings()
    assert settings.juniper_data_api_key == "outbound-key"


def test_juniper_data_api_key_shared_env_var(monkeypatch):
    monkeypatch.setenv("JUNIPER_DATA_API_KEY", "shared-outbound")
    assert Settings().juniper_data_api_key == "shared-outbound"


def test_juniper_data_url_shared_alias(monkeypatch):
    monkeypatch.setenv("JUNIPER_DATA_URL", "http://shared:8100")
    assert Settings().juniper_data_url == "http://shared:8100"


def test_juniper_data_url_prefixed_alias(monkeypatch):
    monkeypatch.setenv("JUNIPER_RECURRENCE_JUNIPER_DATA_URL", "http://prefixed:8100")
    assert Settings().juniper_data_url == "http://prefixed:8100"


def test_no_dotenv_leak(tmp_path, monkeypatch):
    # The recorded incident: a local .env must NOT leak into settings. We set no
    # env_file=, so a .env in the CWD is ignored and the default port stands.
    (tmp_path / ".env").write_text("JUNIPER_RECURRENCE_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().port == 8210
