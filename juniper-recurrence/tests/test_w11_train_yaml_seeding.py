"""Tests for the W-11 ``train:`` YAML seeding of the ``train`` subcommand (plan SS11 / Wave 3.6).

An explicitly-passed CLI flag wins; an unset flag (None) falls back to the YAML
``train:`` value; keys the CLI does not model are warned about, never applied.
"""

from __future__ import annotations

import argparse

from juniper_recurrence.main import _apply_train_overrides, _experiment_train_overrides


def _namespace(**kwargs) -> argparse.Namespace:
    base = {key: None for key in ("d", "theta", "ridge", "readout", "rff_features", "rff_gamma", "mlp_hidden", "mlp_weight_decay", "mlp_lr", "mlp_max_epochs", "mlp_patience")}
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestExperimentTrainOverrides:
    def test_unset_env_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("JUNIPER_RECURRENCE_CONFIG_FILE", raising=False)
        assert _experiment_train_overrides() == {}

    def test_train_block_extracted_and_unknown_keys_warned(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "experiment.yaml"
        path.write_text(
            "schema_version: 1\nexperiment: {name: t, seed: 1}\ntrain: {d: 32, readout: rff, rff_features: 128, epochs: 9}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("JUNIPER_RECURRENCE_CONFIG_FILE", str(path))
        overrides = _experiment_train_overrides()
        assert overrides == {"d": 32, "readout": "rff", "rff_features": 128}
        assert "epochs" in capsys.readouterr().err


class TestApplyTrainOverrides:
    def test_yaml_seeds_unset_flags(self):
        args = _apply_train_overrides(_namespace(), {"d": 32, "readout": "rff", "rff_gamma": "median"})
        assert args.d == 32
        assert args.readout == "rff"
        assert args.rff_gamma == "median"

    def test_explicit_cli_beats_yaml(self):
        args = _apply_train_overrides(_namespace(d=8, readout="linear"), {"d": 32, "readout": "rff"})
        assert args.d == 8
        assert args.readout == "linear"

    def test_yaml_null_theta_stays_data_driven(self):
        args = _apply_train_overrides(_namespace(), {"theta": None})
        assert args.theta is None

    def test_no_overrides_is_inert(self):
        args = _apply_train_overrides(_namespace(d=8), {})
        assert args.d == 8
        assert args.ridge is None
