"""W5.10 -- the resolved readout hyperparameters in the ``/v1/train`` and ``/v1/crossval`` responses.

Plan item W5.10 (F-S7's response half) of juniper-ml
``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``:
the request names *modes* (``theta=None``, ``ridge="gcv"``, ``rff_gamma="median"``) and until now
the response said nothing about what the fit resolved them to. These pin the ``diagnostics`` block:
the helper that reads it off a fitted model, the two routes and the status re-serve that carry it,
the published OpenAPI schema, and the ``train`` CLI's print of it.

The data path is the suite's fake juniper-data client (a synthetic 3-D artifact), so the real fold
executor and ``LMURegressor`` run end to end; the E-H equities artifact itself is not needed to pin
the contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from juniper_recurrence_model import readouts as model_readouts
from juniper_recurrence_model.data import sequence_data_from_arrays

from juniper_recurrence import main as cli
from juniper_recurrence._readout import build_lmu_regressor, readout_diagnostics
from juniper_recurrence.app import build_app
from juniper_recurrence.schemas import ReadoutDiagnostics
from juniper_recurrence.settings import Settings

GCV_FLOOR = float(model_readouts._GCV_GRID[0])
GCV_CEILING = float(model_readouts._GCV_GRID[-1])

DIAGNOSTIC_KEYS = {"readout", "theta", "ridge_mode", "ridge_selected", "gamma", "gcv_at_edge", "null_model"}


def _client(**settings_kwargs) -> TestClient:
    return TestClient(build_app(Settings(**settings_kwargs)))


def _fit_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    n, lookback, n_features = 24, 4, 2
    X = rng.standard_normal((n, lookback, n_features)).astype("float32")
    dt = np.zeros((n, lookback), dtype="float32")
    dt[:, 1:] = rng.uniform(0.5, 2.0, size=(n, lookback - 1)).astype("float32")
    y = rng.standard_normal((n,)).astype("float32")
    target_dt = np.ones((n,), dtype="float32")
    return X, y, dt, target_dt


def _fitted(**kwargs):
    defaults = dict(d=4, theta=None, readout=None, ridge=None, rff_features=None, rff_gamma=None, default_ridge=0.0)
    defaults.update(kwargs)
    model = build_lmu_regressor(**defaults)
    X, y, dt, target_dt = _fit_data()
    model.fit(X, y, dt=dt, target_dt=target_dt)
    return model


@pytest.fixture
def full_npz_arrays() -> dict[str, np.ndarray]:
    """A contract-valid ``full`` split (20 windows) whose per-window elapsed time varies, so a
    data-driven theta differs between folds of different sizes."""
    rng = np.random.default_rng(0)
    n_windows, lookback, n_features = 40, 5, 2  # 40 against a look-back of 5: three folds survive the W5.2 default embargo
    X = rng.standard_normal((n_windows, lookback, n_features)).astype("float32")
    dt = np.zeros((n_windows, lookback), dtype="float32")
    dt[:, 1:] = rng.uniform(0.5, 2.0, size=(n_windows, lookback - 1)).astype("float32")
    y = rng.standard_normal((n_windows,)).astype("float32")
    target_dt = rng.uniform(0.5, 2.0, size=(n_windows,)).astype("float32")
    seq_lengths = np.full((n_windows,), lookback, dtype="int64")
    return {"X_train": X, "y_reg_train": y, "dt_train": dt, "target_dt_train": target_dt, "seq_lengths_train": seq_lengths}


@pytest.fixture
def fake_full_data(monkeypatch, full_npz_arrays) -> dict[str, np.ndarray]:
    class _FakeDataClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_latest(self, name):
            return {"dataset_id": f"latest-of-{name}"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "created-1"}

        def download_artifact_npz(self, dataset_id):
            return full_npz_arrays

        def close(self) -> None:
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _FakeDataClient)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda arrays, **kw: "sequence")
    return full_npz_arrays


# --- readout_diagnostics(model): the one place the fitted readout is read -----------------


def test_diagnostics_refuse_an_unfitted_model() -> None:
    model = build_lmu_regressor(d=4, theta=2.0, readout=None, ridge=None, rff_features=None, rff_gamma=None, default_ridge=0.0)
    with pytest.raises(RuntimeError, match="fitted"):
        readout_diagnostics(model)


def test_diagnostics_linear_fixed_ridge_reports_the_value_and_no_gcv_flags() -> None:
    diag = readout_diagnostics(_fitted(theta=2.0, ridge=0.5))
    assert diag.readout == "linear"
    assert diag.theta == 2.0  # a pinned theta is reported as pinned
    assert diag.ridge_mode == "fixed"
    assert diag.ridge_selected == 0.5
    assert diag.gamma is None
    assert diag.gcv_at_edge is None  # not GCV-selected: not applicable, not False
    assert diag.null_model is None  # not determined until the model reports it (W5.9)


def test_diagnostics_linear_service_default_ridge_is_gcv_and_an_explicit_zero_is_fixed() -> None:
    # A bare request: ridge=None falls back to Settings().default_ridge -- "gcv" since 0.7.0 (W5.8 (B)) --
    # and the block says so; a deployment that sets default_ridge back to 0.0 is reported as fixed 0.0.
    diag = readout_diagnostics(_fitted(theta=2.0, ridge=None, default_ridge=Settings().default_ridge))
    assert diag.ridge_mode == "gcv"
    assert GCV_FLOOR <= diag.ridge_selected <= GCV_CEILING
    assert isinstance(diag.gcv_at_edge, bool) and isinstance(diag.null_model, bool)  # the model reports both (W5.9)
    diag = readout_diagnostics(_fitted(theta=2.0, ridge=None, default_ridge=0.0))
    assert diag.ridge_mode == "fixed"
    assert diag.ridge_selected == 0.0


def test_diagnostics_data_driven_theta_is_the_resolved_median() -> None:
    # theta=None: the model resolves median(sum(dt)) at fit; the block reports that number, not None.
    _, _, dt, _ = _fit_data()
    diag = readout_diagnostics(_fitted(theta=None))
    assert diag.theta == pytest.approx(float(np.median(np.sum(dt, axis=1))))
    assert diag.theta > 0


def test_diagnostics_gcv_reports_the_selected_lambda_and_the_edge_flag() -> None:
    diag = readout_diagnostics(_fitted(theta=2.0, ridge="gcv"))
    assert diag.ridge_mode == "gcv"
    assert diag.ridge_selected is not None
    assert GCV_FLOOR <= diag.ridge_selected <= GCV_CEILING
    assert isinstance(diag.gcv_at_edge, bool)
    assert diag.gcv_at_edge == (diag.ridge_selected in (GCV_FLOOR, GCV_CEILING))
    assert isinstance(diag.null_model, bool)  # model 0.5.0 (W5.9) sets null_model_ on every GCV fit


def test_diagnostics_take_the_models_w5_9_flags_verbatim() -> None:
    # juniper-recurrence-model 0.5.0 sets gcv_at_edge_ / null_model_ on the fitted readout after a GCV
    # fit; the block carries exactly those, not a re-derivation.
    model = _fitted(theta=2.0, ridge="gcv")
    assert isinstance(model._readout.gcv_at_edge_, bool) and isinstance(model._readout.null_model_, bool)
    diag = readout_diagnostics(model)
    assert diag.gcv_at_edge is model._readout.gcv_at_edge_
    assert diag.null_model is model._readout.null_model_


@pytest.mark.parametrize("selected, expected", [(GCV_CEILING, True), (GCV_FLOOR, True), (1.0, False)])
def test_diagnostics_gcv_at_edge_is_derived_from_the_grid_bounds(selected: float, expected: bool) -> None:
    # The flag is a function of the selected lambda and the grid's two ends; pin it at both ends and
    # in the middle by overriding what the fitted readout reports as its selection. The model's own
    # flag (set since model 0.5.0) wins when present, so clear it to exercise the fallback derivation.
    model = _fitted(theta=2.0, ridge="gcv")
    model._readout.ridge = selected
    model._readout.gcv_at_edge_ = None
    assert readout_diagnostics(model).gcv_at_edge is expected


def test_diagnostics_gcv_at_edge_is_none_when_the_model_hides_its_grid(monkeypatch) -> None:
    model = _fitted(theta=2.0, ridge="gcv")  # the fit itself needs the grid; hide it only from the read
    model._readout.gcv_at_edge_ = None  # and pretend the readout did not report its own flag (pre-0.5.0 model)
    monkeypatch.delattr(model_readouts, "_GCV_GRID")
    diag = readout_diagnostics(model)
    assert diag.ridge_mode == "gcv"
    assert diag.gcv_at_edge is None


def test_diagnostics_prefer_the_readouts_own_flags_when_it_reports_them() -> None:
    # The W5.9 contract (juniper-recurrence-model 0.5.0): a readout that evaluates the GCV curve
    # beyond the ceiling sets ``gcv_at_edge_`` / ``null_model_`` on itself, and those win over the
    # derivation here. Simulated by setting the attributes on the fitted readout.
    model = _fitted(theta=2.0, ridge="gcv")
    model._readout.ridge = 1.0  # mid-grid: the derivation alone would say False
    model._readout.gcv_at_edge_ = True
    model._readout.null_model_ = True
    diag = readout_diagnostics(model)
    assert diag.gcv_at_edge is True
    assert diag.null_model is True


def test_diagnostics_rff_reports_the_resolved_gamma_and_its_gcv_lambda() -> None:
    diag = readout_diagnostics(_fitted(theta=2.0, readout="rff", rff_features=16, rff_gamma="median"))
    assert diag.readout == "rff"
    assert diag.gamma is not None and diag.gamma > 0  # the median heuristic resolved to a number
    assert diag.ridge_mode == "gcv"  # the RFF rung's ridge default
    assert diag.ridge_selected is not None
    assert isinstance(diag.gcv_at_edge, bool)


def test_diagnostics_rff_explicit_gamma_and_ridge_are_reported_as_given() -> None:
    diag = readout_diagnostics(_fitted(theta=2.0, readout="rff", rff_features=16, rff_gamma=0.25, ridge=2.0))
    assert diag.gamma == 0.25
    assert diag.ridge_mode == "fixed"
    assert diag.ridge_selected == 2.0
    assert diag.gcv_at_edge is None


def test_diagnostics_block_is_json_safe_floats_and_bools() -> None:
    # The response is JSON; nothing numpy-typed may leak through the block.
    dumped = readout_diagnostics(_fitted(theta=None, ridge="gcv")).model_dump()
    assert set(dumped) == DIAGNOSTIC_KEYS
    assert type(dumped["theta"]) is float
    assert type(dumped["ridge_selected"]) is float
    assert type(dumped["gcv_at_edge"]) is bool


# --- /v1/train carries the block ---------------------------------------------------------------


def test_train_response_carries_the_fits_diagnostics(fake_data) -> None:
    body = _client(api_keys=None).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "d": 4, "theta": 3.0, "ridge": 0.0}).json()
    assert set(body["diagnostics"]) == DIAGNOSTIC_KEYS
    assert body["diagnostics"] == {"readout": "linear", "theta": 3.0, "ridge_mode": "fixed", "ridge_selected": 0.0, "gamma": None, "gcv_at_edge": None, "null_model": None}


def test_bare_train_request_gets_the_gcv_default_and_the_response_says_so(fake_data) -> None:
    # W5.8 (B): a request that omits ``ridge`` is GCV-selected since 0.7.0, and the block announces the
    # lambda it chose and whether the fit abstained -- the default must not be silent.
    body = _client(api_keys=None).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "d": 4, "theta": 3.0}).json()
    diag = body["diagnostics"]
    assert diag["readout"] == "linear" and diag["ridge_mode"] == "gcv"
    assert GCV_FLOOR <= diag["ridge_selected"] <= GCV_CEILING
    assert isinstance(diag["gcv_at_edge"], bool) and isinstance(diag["null_model"], bool)


def test_bare_train_request_honours_a_deployment_that_opts_back_to_zero(fake_data) -> None:
    body = _client(api_keys=None, default_ridge=0.0).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "d": 4, "theta": 3.0}).json()
    assert body["diagnostics"]["ridge_mode"] == "fixed" and body["diagnostics"]["ridge_selected"] == 0.0


def test_train_response_reports_the_resolved_theta_when_the_request_omitted_it(fake_data) -> None:
    # F-S7: a theta=None request used to leave the resolved value unknowable; now it is in the response.
    body = _client(api_keys=None).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "d": 4}).json()
    assert body["diagnostics"]["theta"] == pytest.approx(float(np.median(np.sum(fake_data["dt_train"], axis=1))))


def test_train_response_reports_the_gcv_lambda(fake_data) -> None:
    body = _client(api_keys=None).post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}, "d": 4, "ridge": "gcv"}).json()
    diag = body["diagnostics"]
    assert diag["ridge_mode"] == "gcv"
    assert GCV_FLOOR <= diag["ridge_selected"] <= GCV_CEILING
    assert isinstance(diag["gcv_at_edge"], bool)
    assert isinstance(diag["null_model"], bool)  # reported by model 0.5.0 (W5.9)


# --- /v1/crossval carries one block per fold, and the status re-serves it -----------------------


@pytest.mark.parametrize("request_ridge", [{"ridge": "gcv"}, {}], ids=["explicit-gcv", "bare-request"])
def test_crossval_response_carries_per_fold_diagnostics(fake_full_data, request_ridge) -> None:
    # Explicit ``ridge="gcv"`` and the bare request (the W5.8 (B) default) are the same fit, and every
    # fold's block reports the lambda it selected -- the response half of W5.8's acceptance.
    body = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4, **request_ridge}).json()
    assert len(body["folds"]) == 3
    for fold in body["folds"]:
        diag = fold["diagnostics"]
        assert set(diag) == DIAGNOSTIC_KEYS
        assert diag["readout"] == "linear"
        assert diag["theta"] > 0  # each fold resolved its own data-driven theta
        assert diag["ridge_mode"] == "gcv"
        assert GCV_FLOOR <= diag["ridge_selected"] <= GCV_CEILING
        assert isinstance(diag["gcv_at_edge"], bool)
        assert diag["gamma"] is None
        assert isinstance(diag["null_model"], bool)


def test_crossval_per_fold_theta_is_each_folds_own_median(fake_full_data) -> None:
    # Expanding folds train on growing prefixes, so the median elapsed time differs per fold; the
    # block reports what each fold's model actually used -- the number the W5.1 note had to read
    # in-process.
    from juniper_model_core.crossval import walk_forward_folds

    body = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4, "embargo": 0}).json()
    elapsed = np.sum(fake_full_data["dt_train"], axis=1)
    folds = walk_forward_folds(elapsed.shape[0], n_folds=3, scheme="expanding", embargo=0, min_train=None)
    expected = [float(np.median(elapsed[np.asarray(fold.train_idx)])) for fold in folds]
    assert [f["diagnostics"]["theta"] for f in body["folds"]] == pytest.approx(expected)


def test_crossval_rff_folds_report_their_own_gamma(fake_full_data) -> None:
    body = _client(api_keys=None).post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4, "readout": "rff", "rff_features": 16}).json()
    for fold in body["folds"]:
        diag = fold["diagnostics"]
        assert diag["readout"] == "rff"
        assert diag["gamma"] > 0
        assert diag["ridge_mode"] == "gcv"


def test_crossval_status_reserves_the_per_fold_diagnostics(fake_full_data) -> None:
    client = _client(api_keys=None)
    posted = client.post("/v1/crossval", json={"dataset": {"dataset_id": "ds-1"}, "n_folds": 3, "d": 4, "ridge": "gcv"}).json()
    status = client.get("/v1/crossval/status").json()
    assert status["state"] == "done"
    assert [f["diagnostics"] for f in status["result"]["folds"]] == [f["diagnostics"] for f in posted["folds"]]


# --- the published contract ----------------------------------------------------------------------


def test_openapi_publishes_the_diagnostics_block_on_both_responses() -> None:
    schema = _client(api_keys=None).app.openapi()
    components = schema["components"]["schemas"]
    block = components["ReadoutDiagnostics"]
    assert set(block["properties"]) == DIAGNOSTIC_KEYS
    assert set(block["required"]) == {"readout", "theta"}
    assert components["TrainResponse"]["properties"]["diagnostics"]["$ref"].endswith("/ReadoutDiagnostics")
    assert "diagnostics" in components["TrainResponse"]["required"]
    assert components["CrossValFoldModel"]["properties"]["diagnostics"]["$ref"].endswith("/ReadoutDiagnostics")
    assert "diagnostics" in components["CrossValFoldModel"]["required"]


def test_schema_model_validates_its_fields() -> None:
    from pydantic import ValidationError

    ReadoutDiagnostics(readout="linear", theta=1.0)
    with pytest.raises(ValidationError):
        ReadoutDiagnostics(readout="linear", theta=0.0)  # theta must be positive
    with pytest.raises(ValidationError):
        ReadoutDiagnostics(readout="linear", theta=1.0, ridge_mode="auto")


# --- the train CLI prints the same block --------------------------------------------------------


def test_cli_train_prints_the_readout_block(monkeypatch, synthetic_npz_arrays, capsys) -> None:
    sequence = sequence_data_from_arrays(synthetic_npz_arrays, "train")
    descriptor = {"dataset_id": "ds-1", "name": None, "split": "train", "n_windows": 12, "lookback": 5, "n_features": 2, "output_dim": 1, "has_target_dt": True, "has_seq_lengths": True}
    monkeypatch.setattr("juniper_recurrence.data.load_sequence_data", lambda **kwargs: (sequence, descriptor))

    rc = cli.main(["train", "--dataset", "ds-1", "--d", "4", "--ridge", "gcv"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Readout:" in out
    assert "  readout: linear" in out
    assert "  ridge_mode: gcv" in out
    assert "  theta: " in out
