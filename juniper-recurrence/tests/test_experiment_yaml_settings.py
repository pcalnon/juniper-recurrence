"""Tests for the experiment YAML config layer (Wave 3.3 -- CLI experimentation plan SS5.1/SS5.2/SS5.6).

``ExperimentYamlSettingsSource`` projects ONLY the experiment YAML's ``service:`` block
into ``Settings`` (YAML > env in the SS5.1 precedence), validating fail-loud: unknown
top-level blocks, ``schema_version``, unknown ``service:`` keys (the model is
``extra="ignore"``, so silent dropping must be impossible), and the launcher-owned
infra keys ``host``/``port``/``juniper_data_url`` (SS5.6 rule 6). The layer is inert
when ``JUNIPER_RECURRENCE_CONFIG_FILE`` is unset, init-kwargs still beat the YAML, and
the ``--config`` CLI flag threads the env var before settings load.
"""

from __future__ import annotations

import pytest

from juniper_recurrence.settings import ExperimentConfigError, Settings

_BASE_YAML = """
schema_version: 1
experiment:
  name: layer-test
  seed: 1
service:
  log_level: DEBUG
  log_format: json
  rate_limit_enabled: false
  default_d: 32
dataset:
  generator: irregular_sine
  split: train
  params:
    seed: 1
train:
  d: 8
  readout: linear
crossval:
  enabled: false
predict:
  enabled: false
outputs:
  plots: []
"""


def _write_yaml(tmp_path, body):
    path = tmp_path / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestExperimentYamlLayer:
    """SS5.1 precedence + inertness."""

    def test_inert_without_env_var(self, monkeypatch):
        monkeypatch.delenv("JUNIPER_RECURRENCE_CONFIG_FILE", raising=False)
        monkeypatch.setenv("JUNIPER_RECURRENCE_LOG_LEVEL", "WARNING")
        assert Settings().log_level == "WARNING"

    def test_yaml_beats_env(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_RECURRENCE_CONFIG_FILE", str(path))
        monkeypatch.setenv("JUNIPER_RECURRENCE_LOG_LEVEL", "ERROR")
        monkeypatch.setenv("JUNIPER_RECURRENCE_RATE_LIMIT_ENABLED", "true")
        settings = Settings()
        assert settings.log_level == "DEBUG"
        assert settings.log_format == "json"
        assert settings.rate_limit_enabled is False
        assert settings.default_d == 32

    def test_init_kwargs_beat_yaml(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_RECURRENCE_CONFIG_FILE", str(path))
        assert Settings(log_level="WARNING").log_level == "WARNING"

    def test_env_still_wins_for_unprojected_fields(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_RECURRENCE_CONFIG_FILE", str(path))
        monkeypatch.setenv("JUNIPER_RECURRENCE_METRICS_ENABLED", "false")
        settings = Settings()
        assert settings.metrics_enabled is False
        assert settings.log_level == "DEBUG"

    def test_non_service_blocks_are_ignored_by_settings(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_RECURRENCE_CONFIG_FILE", str(path))
        settings = Settings()
        assert settings.port == 8210
        assert settings.default_theta is None


class TestExperimentYamlValidation:
    """SS5.6 rules 1/2/6 -- fail loud before boot."""

    def _expect_error(self, tmp_path, monkeypatch, body, match):
        path = _write_yaml(tmp_path, body)
        monkeypatch.setenv("JUNIPER_RECURRENCE_CONFIG_FILE", str(path))
        with pytest.raises(ExperimentConfigError, match=match):
            Settings()

    @pytest.mark.parametrize("key", ["host", "port", "juniper_data_url"])
    def test_launcher_owned_service_keys_rejected(self, tmp_path, monkeypatch, key):
        body = f"schema_version: 1\nservice:\n  {key}: anything\n"
        self._expect_error(tmp_path, monkeypatch, body, match="rule 6")

    def test_unknown_service_key_rejected(self, tmp_path, monkeypatch):
        body = "schema_version: 1\nservice:\n  default_dd: 8\n"
        self._expect_error(tmp_path, monkeypatch, body, match="default_dd")

    def test_unknown_top_level_block_rejected(self, tmp_path, monkeypatch):
        body = "schema_version: 1\nsurprise: {}\nservice:\n  log_level: DEBUG\n"
        self._expect_error(tmp_path, monkeypatch, body, match="surprise")

    def test_schema_version_required(self, tmp_path, monkeypatch):
        self._expect_error(tmp_path, monkeypatch, "service:\n  log_level: DEBUG\n", match="schema_version")

    def test_future_schema_version_rejected(self, tmp_path, monkeypatch):
        self._expect_error(tmp_path, monkeypatch, "schema_version: 99\nservice: {}\n", match="schema_version")

    def test_missing_file_fails_loud(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JUNIPER_RECURRENCE_CONFIG_FILE", str(tmp_path / "nope.yaml"))
        with pytest.raises(ExperimentConfigError, match="unreadable"):
            Settings()

    def test_non_mapping_yaml_rejected(self, tmp_path, monkeypatch):
        self._expect_error(tmp_path, monkeypatch, "- a\n- list\n", match="mapping")


class TestConfigCliFlag:
    """The ``--config`` flag threads the env var before either handler builds Settings()."""

    def test_serve_config_flag_sets_env_var(self, tmp_path, monkeypatch):
        from unittest import mock

        from juniper_recurrence import main as main_mod

        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.delenv("JUNIPER_RECURRENCE_CONFIG_FILE", raising=False)
        captured = {}

        def fake_serve(args):
            import os

            captured["env"] = os.environ.get("JUNIPER_RECURRENCE_CONFIG_FILE")
            return 0

        with mock.patch.object(main_mod, "_serve", side_effect=fake_serve):
            rc = main_mod.main(["serve", "--config", str(path)])
        assert rc == 0
        assert captured["env"] == str(path)

    def test_train_config_flag_sets_env_var(self, tmp_path, monkeypatch):
        from unittest import mock

        from juniper_recurrence import main as main_mod

        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.delenv("JUNIPER_RECURRENCE_CONFIG_FILE", raising=False)
        captured = {}

        def fake_train(args):
            import os

            captured["env"] = os.environ.get("JUNIPER_RECURRENCE_CONFIG_FILE")
            return 0

        with mock.patch.object(main_mod, "_train", side_effect=fake_train):
            rc = main_mod.main(["train", "--generator", "irregular_sine", "--config", str(path)])
        assert rc == 0
        assert captured["env"] == str(path)
