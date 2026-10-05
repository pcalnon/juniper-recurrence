"""``juniper_data_timeout_seconds``: the service's inner juniper-data timeout is configurable (W1.5, F-S9).

``load_sequence_data`` built ``JuniperDataClient(base_url=…, api_key=…)`` with no ``timeout``, so the
client's 30 s default governed dataset creation -- where a cold ``equities_seq`` fetch happens --
and no setting could raise it, while every outer budget stayed green (F-S9 / F-D5 of juniper-ml
``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``).

These replace the client constructor in ``juniper_recurrence.data`` with a recorder and assert the
``timeout`` keyword it receives, through every path that fetches: the train, predict and
cross-validation routes and the headless CLI ``train``. Fixtures live here (no ``conftest.py`` edits).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from juniper_recurrence import data
from juniper_recurrence import main as cli
from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings

_ENV = "JUNIPER_RECURRENCE_JUNIPER_DATA_TIMEOUT_SECONDS"


@pytest.fixture
def client_kwargs(monkeypatch, synthetic_npz_arrays) -> list[dict]:
    """Record the keyword arguments of every ``JuniperDataClient`` the data adapter constructs."""
    # Serve the windows under the ``full`` split too, so the cross-validation route can fetch them.
    arrays = dict(synthetic_npz_arrays)
    arrays.update({key.replace("_train", "_full"): value for key, value in synthetic_npz_arrays.items()})
    recorded: list[dict] = []

    class _RecordingClient:
        def __init__(self, **kwargs) -> None:
            recorded.append(kwargs)

        def get_latest(self, name):
            return {"dataset_id": f"latest-of-{name}"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "created-1"}

        def download_artifact_npz(self, dataset_id):
            return arrays

        def close(self) -> None:
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _RecordingClient)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda arrays, **kw: "sequence")
    return recorded


# --- the setting --------------------------------------------------------------------------------


def test_default_is_at_least_120_seconds_and_matches_the_adapter_default():
    """The plan's floor is 120 s; the setting and the adapter's own default must not drift apart."""
    timeout = Settings().juniper_data_timeout_seconds
    assert timeout >= 120
    assert timeout == data.DEFAULT_JUNIPER_DATA_TIMEOUT_SECONDS


def test_env_override(monkeypatch):
    monkeypatch.setenv(_ENV, "300")
    assert Settings().juniper_data_timeout_seconds == 300.0


def test_no_unprefixed_alias(monkeypatch):
    """Only the service's own prefixed variable binds -- another service's timeout cannot leak in."""
    monkeypatch.setenv("JUNIPER_DATA_TIMEOUT_SECONDS", "5")
    assert Settings().juniper_data_timeout_seconds == data.DEFAULT_JUNIPER_DATA_TIMEOUT_SECONDS


@pytest.mark.parametrize("bad", [0, -1.0])
def test_a_non_positive_timeout_is_rejected(bad):
    with pytest.raises(ValidationError):
        Settings(juniper_data_timeout_seconds=bad)


# --- the setting reaches JuniperDataClient(timeout=…) ----------------------------------------------


def test_train_passes_the_setting_to_the_client(client_kwargs):
    resp = TestClient(build_app(Settings(api_keys=None, juniper_data_timeout_seconds=222.0))).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    assert resp.status_code == 200, resp.text
    assert [kwargs["timeout"] for kwargs in client_kwargs] == [222.0]


def test_train_uses_the_default_when_nothing_is_configured(client_kwargs):
    resp = TestClient(build_app(Settings(api_keys=None))).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    assert resp.status_code == 200, resp.text
    assert client_kwargs[0]["timeout"] == data.DEFAULT_JUNIPER_DATA_TIMEOUT_SECONDS


def test_the_env_override_reaches_the_client(client_kwargs, monkeypatch):
    monkeypatch.setenv(_ENV, "300")
    resp = TestClient(build_app(Settings(api_keys=None))).post("/v1/train", json={"dataset": {"generator": "equities_seq"}})
    assert resp.status_code == 200, resp.text
    assert client_kwargs[0]["timeout"] == 300.0


def test_predict_passes_the_setting_to_the_client(client_kwargs):
    client = TestClient(build_app(Settings(api_keys=None, juniper_data_timeout_seconds=150.0)))
    assert client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}}).status_code == 200
    resp = client.post("/v1/predict", json={"dataset": {"dataset_id": "ds-1"}})
    assert resp.status_code == 200, resp.text
    assert [kwargs["timeout"] for kwargs in client_kwargs] == [150.0, 150.0]


def test_crossval_passes_the_setting_to_the_client(client_kwargs):
    resp = TestClient(build_app(Settings(api_keys=None, juniper_data_timeout_seconds=180.0))).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 2, "d": 4})
    assert resp.status_code == 200, resp.text
    assert [kwargs["timeout"] for kwargs in client_kwargs] == [180.0]


def test_cli_train_passes_the_setting_to_the_client(client_kwargs, monkeypatch, capsys):
    monkeypatch.setenv(_ENV, "250")
    assert cli.main(["train", "--dataset", "ds-1"]) == 0
    assert [kwargs["timeout"] for kwargs in client_kwargs] == [250.0]
    assert "Trained LMURegressor on dataset ds-1" in capsys.readouterr().out


def test_the_adapter_never_falls_back_to_the_clients_30_second_default(client_kwargs):
    """A direct caller that names no timeout still gets the adapter's default, not the client's 30 s."""
    data.load_sequence_data(base_url="http://data", dataset_id="ds-1")
    assert client_kwargs[0]["timeout"] == data.DEFAULT_JUNIPER_DATA_TIMEOUT_SECONDS
    assert client_kwargs[0]["timeout"] > 30
