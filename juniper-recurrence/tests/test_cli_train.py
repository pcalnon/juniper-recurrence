"""CLI ``train`` subcommand tests (plan §11 / §12).

The data adapter is mocked so ``train`` runs end-to-end on the synthetic fixture:
fit the LMU, print metrics, and persist via ``LMUSerializer`` — no live juniper-data.
"""

from __future__ import annotations

from juniper_recurrence_model import sequence_data_from_arrays

from juniper_recurrence import main as cli


def test_cli_train_end_to_end(monkeypatch, synthetic_npz_arrays, tmp_path, capsys):
    sequence = sequence_data_from_arrays(synthetic_npz_arrays, "train")
    descriptor = {
        "dataset_id": "ds-1",
        "name": None,
        "split": "train",
        "n_windows": 12,
        "lookback": 5,
        "n_features": 2,
        "output_dim": 1,
        "has_target_dt": True,
        "has_seq_lengths": True,
    }
    monkeypatch.setattr("juniper_recurrence.data.load_sequence_data", lambda **kwargs: (sequence, descriptor))

    out = tmp_path / "model.npz"
    rc = cli.main(["train", "--dataset", "ds-1", "--d", "4", "--out", str(out)])

    assert rc == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "Trained LMURegressor" in printed
    assert "Metrics:" in printed
    assert "r2" in printed
    assert f"Saved model to {out}" in printed


def test_cli_train_without_out_prints_metrics(monkeypatch, synthetic_npz_arrays, capsys):
    sequence = sequence_data_from_arrays(synthetic_npz_arrays, "train")
    descriptor = {
        "dataset_id": "ds-2",
        "name": None,
        "split": "train",
        "n_windows": 12,
        "lookback": 5,
        "n_features": 2,
        "output_dim": 1,
        "has_target_dt": True,
        "has_seq_lengths": True,
    }
    monkeypatch.setattr("juniper_recurrence.data.load_sequence_data", lambda **kwargs: (sequence, descriptor))

    rc = cli.main(["train", "--dataset", "ds-2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Metrics:" in out
    assert "Saved model to" not in out


def test_cli_train_requires_ref(capsys):
    rc = cli.main(["train"])
    assert rc == 2
    assert "requires one of" in capsys.readouterr().err


def test_cli_ridge_arg_parses_gcv_and_float():
    """--ridge accepts the literal 'gcv' or a float (DP-3 P1)."""
    assert cli._ridge_arg("gcv") == "gcv"
    assert cli._ridge_arg("0.25") == 0.25


def test_cli_train_with_gcv_ridge(monkeypatch, synthetic_npz_arrays, capsys):
    """`train --ridge gcv` fits with a GCV-selected ridge end-to-end (DP-3 P1)."""
    sequence = sequence_data_from_arrays(synthetic_npz_arrays, "train")
    descriptor = {
        "dataset_id": "ds-1",
        "name": None,
        "split": "train",
        "n_windows": 12,
        "lookback": 5,
        "n_features": 2,
        "output_dim": 1,
        "has_target_dt": True,
        "has_seq_lengths": True,
    }
    monkeypatch.setattr("juniper_recurrence.data.load_sequence_data", lambda **kwargs: (sequence, descriptor))
    rc = cli.main(["train", "--dataset", "ds-1", "--d", "4", "--ridge", "gcv"])
    assert rc == 0
    assert "Metrics:" in capsys.readouterr().out


def test_cli_gamma_arg_parses_median_and_float():
    """--rff-gamma accepts the literal 'median' or a float (DP-3 P2c)."""
    assert cli._gamma_arg("median") == "median"
    assert cli._gamma_arg("0.5") == 0.5


def test_cli_train_with_rff_readout(monkeypatch, synthetic_npz_arrays, capsys):
    """`train --readout rff` fits with the RFF nonlinear readout end-to-end (DP-3 P2c)."""
    sequence = sequence_data_from_arrays(synthetic_npz_arrays, "train")
    descriptor = {
        "dataset_id": "ds-1",
        "name": None,
        "split": "train",
        "n_windows": 12,
        "lookback": 5,
        "n_features": 2,
        "output_dim": 1,
        "has_target_dt": True,
        "has_seq_lengths": True,
    }
    monkeypatch.setattr("juniper_recurrence.data.load_sequence_data", lambda **kwargs: (sequence, descriptor))
    rc = cli.main(["train", "--dataset", "ds-1", "--d", "4", "--readout", "rff", "--rff-features", "32", "--rff-gamma", "median"])
    assert rc == 0
    assert "Metrics:" in capsys.readouterr().out


def test_cli_train_rejects_rff_params_without_rff_readout(monkeypatch, synthetic_npz_arrays, capsys):
    """`train --rff-features` without `--readout rff` exits 2 with an error (not silently ignored).

    Mirrors the HTTP edge's 422 — the consistency fix routes both surfaces through build_lmu_regressor.
    """
    sequence = sequence_data_from_arrays(synthetic_npz_arrays, "train")
    descriptor = {
        "dataset_id": "ds-1",
        "name": None,
        "split": "train",
        "n_windows": 12,
        "lookback": 5,
        "n_features": 2,
        "output_dim": 1,
        "has_target_dt": True,
        "has_seq_lengths": True,
    }
    monkeypatch.setattr("juniper_recurrence.data.load_sequence_data", lambda **kwargs: (sequence, descriptor))
    rc = cli.main(["train", "--dataset", "ds-1", "--rff-features", "64"])
    assert rc == 2
    assert "rff_features / rff_gamma are only valid when readout='rff'" in capsys.readouterr().err
