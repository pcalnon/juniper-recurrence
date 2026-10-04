"""CLI ``train`` subcommand tests (plan §11 / §12).

The data adapter is mocked so ``train`` runs end-to-end on the synthetic fixture:
fit the LMU, print metrics, and persist via ``LMUSerializer`` — no live juniper-data.

The ``--params`` / ``--params-file`` tests (W0.6) fake only ``JuniperDataClient``, one
layer lower, so the params can be seen arriving at ``create_dataset`` and the real
``validate_npz_contract`` + ``sequence_data_from_arrays`` still run on the artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
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


# --- W0.6: --params / --params-file ----------------------------------------------------------
#
# juniper-ml notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md,
# F-S1: `train --generator` exposed no way to pass generator params, so create_dataset always
# received {} and an equities_seq dataset was built from the generator's bare defaults.

# The --help example's bundle (plan W0.6 Details): explicit equities_seq params, with
# fundamentals_fill="drop" and the stationary log_return target.
_EQUITIES_PARAMS = {
    "symbols": ["AAPL"],
    "start_date": "2015-01-01",
    "end_date": "2022-01-01",
    "lookback": 64,
    "fundamentals_fill": "drop",
    "regression_target": "log_return",
}


def _install_recording_client(monkeypatch, artifact) -> list[tuple[str, object]]:
    """Replace the adapter's ``JuniperDataClient`` with an offline fake that records every call.

    Only the client is faked. ``validate_npz_contract`` and ``sequence_data_from_arrays`` are the
    real ones, so the artifact is checked and mapped exactly as a live fetch would be. An empty
    record therefore means no client was even constructed -- nothing reached the network.
    """
    calls: list[tuple[str, object]] = []

    class _RecordingDataClient:
        def __init__(self, **kwargs) -> None:
            calls.append(("init", kwargs))

        def create_dataset(self, *, generator, params, persist):
            calls.append(("create_dataset", {"generator": generator, "params": params, "persist": persist}))
            return {"dataset_id": "ds-eq"}

        def download_artifact_npz(self, dataset_id):
            calls.append(("download_artifact_npz", dataset_id))
            return artifact

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _RecordingDataClient)
    return calls


def _three_partition_artifact(train_arrays):
    """A finite ``train | val | test`` 3-D artifact -- decision 11's three partitions, no ``*_full``.

    ``val`` / ``test`` reuse windows of the train fixture. Their values do not matter here; what
    matters is that the real ``validate_npz_contract`` checks all three partitions, as it does
    for a live ``equities_seq`` artifact.
    """
    arrays = dict(train_arrays)
    for split, rows in (("val", slice(0, 4)), ("test", slice(4, 8))):
        for key, value in train_arrays.items():
            arrays[f"{key.removesuffix('_train')}_{split}"] = value[rows].copy()
    return arrays


def _params_args(text: str, *, via_file: bool, tmp_path) -> list[str]:
    """``--params <text>``, or ``--params-file <path>`` of a file holding ``text``."""
    if not via_file:
        return ["--params", text]
    path = tmp_path / "params.json"
    path.write_text(text, encoding="utf-8")
    return ["--params-file", str(path)]


def test_cli_train_params_reach_create_dataset(monkeypatch, synthetic_npz_arrays, capsys):
    """`train --generator equities_seq --params <json>` creates the dataset WITH those params, then fits.

    This is the plan's bounded offline E2E for W0.6: explicit params -> a finite three-partition
    artifact -> the real contract check and array mapping -> a completed fit that prints its
    final metrics. Before W0.6 the CLI had no --params, so create_dataset always got {}.
    """
    artifact = _three_partition_artifact(synthetic_npz_arrays)
    assert {key.rsplit("_", 1)[1] for key in artifact} == {"train", "val", "test"}
    assert all(np.isfinite(value).all() for value in artifact.values())
    calls = _install_recording_client(monkeypatch, artifact)

    rc = cli.main(["train", "--generator", "equities_seq", "--params", json.dumps(_EQUITIES_PARAMS)])

    assert rc == 0
    assert [name for name, _ in calls] == ["init", "create_dataset", "download_artifact_npz", "close"]
    assert calls[1][1] == {"generator": "equities_seq", "params": _EQUITIES_PARAMS, "persist": True}
    assert calls[2][1] == "ds-eq"
    out = capsys.readouterr().out
    assert "Trained LMURegressor on dataset ds-eq (split=train, windows=12, F=2)." in out
    assert "Metrics:" in out
    assert "  r2: " in out


def test_cli_train_params_file_reaches_create_dataset(monkeypatch, synthetic_npz_arrays, tmp_path, capsys):
    """`--params-file <path>` is the same channel: the file's JSON object reaches create_dataset."""
    calls = _install_recording_client(monkeypatch, synthetic_npz_arrays)
    params_file = tmp_path / "equities_seq.json"
    params_file.write_text(json.dumps(_EQUITIES_PARAMS, indent=2), encoding="utf-8")

    rc = cli.main(["train", "--generator", "equities_seq", "--params-file", str(params_file)])

    assert rc == 0
    assert calls[1] == ("create_dataset", {"generator": "equities_seq", "params": _EQUITIES_PARAMS, "persist": True})
    assert "Metrics:" in capsys.readouterr().out


def test_cli_train_generator_without_params_still_sends_empty_params(monkeypatch, synthetic_npz_arrays):
    """Neither flag: behaviour is unchanged -- the adapter's own default, an empty params object."""
    calls = _install_recording_client(monkeypatch, synthetic_npz_arrays)

    assert cli.main(["train", "--generator", "equities_seq"]) == 0
    assert calls[1] == ("create_dataset", {"generator": "equities_seq", "params": {}, "persist": True})


def test_cli_train_params_and_params_file_are_mutually_exclusive(monkeypatch, synthetic_npz_arrays, tmp_path, capsys):
    """Both flags at once is a usage error: argparse exits 2, as ``main()`` does for every parse error."""
    calls = _install_recording_client(monkeypatch, synthetic_npz_arrays)
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["train", "--generator", "equities_seq", "--params", "{}", "--params-file", str(params_file)])

    assert excinfo.value.code == 2
    assert "argument --params-file: not allowed with argument --params" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("via_file", [False, True], ids=["params", "params-file"])
def test_cli_train_malformed_params_json_exits_2(monkeypatch, synthetic_npz_arrays, tmp_path, capsys, via_file):
    """Text that is not JSON is refused before any dataset is requested (exit 2, message on stderr)."""
    calls = _install_recording_client(monkeypatch, synthetic_npz_arrays)

    rc = cli.main(["train", "--generator", "equities_seq", *_params_args('{"symbols": ["AAPL"', via_file=via_file, tmp_path=tmp_path)])

    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error: --params")
    assert "is not valid JSON" in err
    assert calls == []


@pytest.mark.parametrize(("text", "kind"), [('["AAPL"]', "array"), ('"AAPL"', "string"), ("64", "number"), ("true", "boolean"), ("null", "null")])
def test_cli_train_params_must_be_a_json_object(monkeypatch, synthetic_npz_arrays, capsys, text, kind):
    """Valid JSON that is not an object is refused: a list or a scalar is not a params mapping."""
    calls = _install_recording_client(monkeypatch, synthetic_npz_arrays)

    rc = cli.main(["train", "--generator", "equities_seq", "--params", text])

    assert rc == 2
    assert f"error: --params must be a JSON object of generator params, got a JSON {kind}" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("ref", [["--dataset", "ds-1"], ["--name", "equities_seq_v1"]], ids=["dataset", "name"])
@pytest.mark.parametrize("via_file", [False, True], ids=["params", "params-file"])
def test_cli_train_params_require_generator(monkeypatch, synthetic_npz_arrays, tmp_path, capsys, ref, via_file):
    """With only --dataset / --name the params would be silently dropped -- so refuse (exit 2)."""
    calls = _install_recording_client(monkeypatch, synthetic_npz_arrays)

    rc = cli.main(["train", *ref, *_params_args(json.dumps(_EQUITIES_PARAMS), via_file=via_file, tmp_path=tmp_path)])

    assert rc == 2
    assert "error: --params/--params-file require --generator" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("content", [None, b"\xff{}"], ids=["missing", "not-utf8"])
def test_cli_train_unreadable_params_file_exits_2(monkeypatch, synthetic_npz_arrays, tmp_path, capsys, content):
    """A --params-file that cannot be read as UTF-8 text is a clean exit 2, not a traceback."""
    calls = _install_recording_client(monkeypatch, synthetic_npz_arrays)
    params_file = tmp_path / "params.json"
    if content is not None:
        params_file.write_bytes(content)

    rc = cli.main(["train", "--generator", "equities_seq", "--params-file", str(params_file)])

    assert rc == 2
    assert f"error: cannot read --params-file {params_file}" in capsys.readouterr().err
    assert calls == []


def test_cli_train_help_shows_an_equities_seq_params_example(capsys):
    """`train --help` carries an equities_seq example whose --params value is itself a valid params object."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["train", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    example = next(line for line in out.splitlines() if "--generator equities_seq --params '" in line)
    assert json.loads(example.split("--params '", 1)[1].rsplit("'", 1)[0]) == _EQUITIES_PARAMS
    assert "--generator equities_seq --params-file " in out
