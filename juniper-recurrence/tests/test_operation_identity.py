"""Operation identity: ``operation_id`` on train / 409 / status, terminal ``failed``, ``expect_operation_id`` (W1.5).

Closes the service half of W1.5 in juniper-ml
``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md`` (F-S6,
F-CON1, F-CON2). The busy-service tests hold a REAL fit under the lock -- a data client whose download
(or a lifecycle whose fit) blocks on a :class:`threading.Event` while a second request arrives on
another thread -- rather than setting state by hand, because the defect was about what a caller
sees while a fit is genuinely running.

Every fixture lives in this module (no ``conftest.py`` edits); ``synthetic_npz_arrays`` and
``fake_data`` are the shared ones, read-only.
"""

from __future__ import annotations

import re
import threading
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from juniper_data_client import JuniperDataTimeoutError
from juniper_recurrence_model.model import LMUSerializer
from juniper_service_core import TrainingLifecycle

from juniper_recurrence.app import build_app
from juniper_recurrence.settings import Settings
from juniper_recurrence.state import AppState, Operation

#: Upper bound on any wait in this module, so a broken gate fails the test instead of hanging it.
_WAIT = 20.0

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")

_TRAIN = {"dataset": {"dataset_id": "ds-1"}}


def _app(**settings_kwargs):
    settings_kwargs.setdefault("api_keys", None)
    return build_app(Settings(**settings_kwargs))


def _status(client: TestClient) -> dict:
    resp = client.get("/v1/training/status")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _inline(arrays) -> dict:
    """An inline ``/v1/predict`` body over the fixture windows."""
    return {"X": arrays["X_train"].tolist(), "dt": arrays["dt_train"].tolist()}


def _parse_utc(text: str) -> datetime:
    """Parse ``busy_since``: ISO-8601, UTC, ``Z``-suffixed."""
    assert text.endswith("Z"), text
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0)
    return parsed


class _Gate:
    """Two events: ``entered`` is set when the gated call starts, ``release`` lets it continue.

    ``error``, when set, is what a gated fit raises once released.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.error: BaseException | None = None

    def hold(self) -> None:
        self.entered.set()
        if not self.release.wait(timeout=_WAIT):
            raise TimeoutError("test gate never opened")


class _Request(threading.Thread):
    """One request on its own thread, through its own ``TestClient`` (hence its own event loop)."""

    def __init__(self, app, method: str, url: str, **kwargs) -> None:
        super().__init__(daemon=True)
        self._app, self._method, self._url, self._kwargs = app, method, url, kwargs
        self.response = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.response = TestClient(self._app).request(self._method, self._url, **self._kwargs)
        except Exception as exc:  # handed back to the test by result()
            self.error = exc

    def result(self):
        self.join(_WAIT)
        assert not self.is_alive(), "request thread did not finish"
        if self.error is not None:
            raise self.error
        return self.response


def _install_client(monkeypatch, arrays, *, download=None) -> None:
    """A fake juniper-data client serving ``arrays``; ``download(dataset_id)`` overrides the download."""

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def get_latest(self, name):
            return {"dataset_id": f"latest-of-{name}"}

        def create_dataset(self, **kwargs):
            return {"dataset_id": "created-1"}

        def download_artifact_npz(self, dataset_id):
            if download is not None:
                return download(dataset_id)
            return arrays

        def close(self) -> None:
            pass

    monkeypatch.setattr("juniper_recurrence.data.JuniperDataClient", _FakeClient)
    monkeypatch.setattr("juniper_recurrence.data.validate_npz_contract", lambda arrays, **kw: "sequence")


@pytest.fixture
def gated_download(monkeypatch, synthetic_npz_arrays):
    """The dataset download blocks until ``gate.release`` -- a fit held under the lock in its fetch phase."""
    gate = _Gate()

    def _download(dataset_id):
        gate.hold()
        return synthetic_npz_arrays

    _install_client(monkeypatch, synthetic_npz_arrays, download=_download)
    yield gate
    gate.release.set()  # never leave a request thread parked


@pytest.fixture
def gated_fit(monkeypatch, fake_data):
    """The fit itself blocks until ``gate.release``, then runs -- or raises ``gate.error`` if one is set."""
    gate = _Gate()

    class _GatedLifecycle(TrainingLifecycle):
        def run(self, *args, **kwargs):
            gate.hold()
            if gate.error is not None:
                raise gate.error
            return super().run(*args, **kwargs)

    monkeypatch.setattr("juniper_recurrence.routers.training.TrainingLifecycle", _GatedLifecycle)
    yield gate
    gate.release.set()


class _RaisingLifecycle(TrainingLifecycle):
    """A fit that raises the way a numerical failure would."""

    def run(self, *args, **kwargs):
        raise FloatingPointError("design matrix is singular")


# --- operation_id on train + status -------------------------------------------------------------


def test_train_mints_an_operation_id_and_status_reports_it(fake_data):
    client = TestClient(_app())
    resp = client.post("/v1/train", json=_TRAIN, headers={"X-Request-ID": "caller-a"})
    assert resp.status_code == 200, resp.text
    operation_id = resp.json()["operation_id"]
    assert _HEX32.match(operation_id), operation_id

    body = _status(client)
    assert body["state"] == "trained"
    assert body["operation_id"] == operation_id
    assert body["operation"] == "train"
    assert body["model_operation_id"] == operation_id
    assert body["requested_by"] == "caller-a"
    assert body["dataset_id"] == "ds-1"
    assert body["busy_since"] is None, "busy_since is only set while an operation holds the lock"
    assert body["failure"] is None
    # Every pre-W1.5 field is still served, with its old meaning.
    assert set(body["final_metrics"]) >= {"mse", "rmse", "mae", "r2", "loss"}
    assert [e["type"] for e in body["events"]] == ["training_start", "epoch_end", "training_end"]
    assert body["restored_from"] is None


def test_each_fit_mints_a_new_operation_id(fake_data):
    client = TestClient(_app())
    first = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    second = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    assert first != second
    assert _status(client)["operation_id"] == second


def test_requested_by_is_null_without_an_x_request_id(fake_data):
    client = TestClient(_app())
    client.post("/v1/train", json=_TRAIN)
    assert _status(client)["requested_by"] is None


def test_idle_status_names_no_operation():
    body = _status(TestClient(_app()))
    assert body["state"] == "idle"
    for field in ("operation_id", "operation", "busy_since", "dataset_id", "requested_by", "model_operation_id", "failure"):
        assert body[field] is None, field


# --- busy: 409 carries the holder, status reports the run in flight ------------------------------


def test_busy_409_names_the_holder_while_a_real_fit_holds_the_lock(gated_download):
    """F-S6 / F-CON1: the refused caller learns WHOSE run it hit and since when.

    Request A takes the lock and parks in its dataset download; B arrives meanwhile. The 409's
    holder, the in-flight status and A's eventual response must all name the same operation.
    """
    app = _app()
    first = _Request(app, "POST", "/v1/train", json={"dataset": {"dataset_id": "ds-a"}}, headers={"X-Request-ID": "caller-a"})
    first.start()
    try:
        assert gated_download.entered.wait(_WAIT), "the first fit never reached its dataset fetch"
        client = TestClient(app)

        in_flight = _status(client)
        assert in_flight["state"] == "training"
        assert _HEX32.match(in_flight["operation_id"])
        assert in_flight["operation"] == "train"
        assert in_flight["requested_by"] == "caller-a"
        assert in_flight["dataset_id"] == "ds-a"
        assert in_flight["model_operation_id"] is None, "nothing has been trained yet"
        assert in_flight["final_metrics"] is None and in_flight["events"] == []
        busy_since = _parse_utc(in_flight["busy_since"])
        assert abs(datetime.now(tz=UTC) - busy_since) < timedelta(seconds=_WAIT)

        refused = client.post("/v1/train", json={"dataset": {"dataset_id": "ds-b"}}, headers={"X-Request-ID": "caller-b"})
        assert refused.status_code == 409
        detail = refused.json()["detail"]
        assert detail["message"] == "a training run is already in progress"
        assert detail["operation_id"] == in_flight["operation_id"]
        assert detail["busy_since"] == in_flight["busy_since"]
        assert detail["operation"] == "train"
        assert detail["requested_by"] == "caller-a"
        assert detail["dataset_id"] == "ds-a"
        # The refused request is not an operation: it took nothing and recorded nothing.
        assert _status(client)["operation_id"] == in_flight["operation_id"]
    finally:
        gated_download.release.set()

    done = first.result()
    assert done.status_code == 200, done.text
    assert done.json()["operation_id"] == in_flight["operation_id"]
    after = _status(TestClient(app))
    assert after["state"] == "trained"
    assert after["operation_id"] == after["model_operation_id"] == in_flight["operation_id"]
    assert after["busy_since"] is None


def test_in_flight_status_carries_the_resolved_dataset_id(gated_fit):
    """A generator reference has no id until the fetch resolves it; once it has, status and 409 say which."""
    app = _app()
    first = _Request(app, "POST", "/v1/train", json={"dataset": {"generator": "equities_seq"}})
    first.start()
    try:
        assert gated_fit.entered.wait(_WAIT), "the first fit never started"
        client = TestClient(app)
        in_flight = _status(client)
        assert in_flight["state"] == "training"
        assert in_flight["dataset_id"] == "created-1"
        refused = client.post("/v1/train", json=_TRAIN)
        assert refused.status_code == 409
        assert refused.json()["detail"]["dataset_id"] == "created-1"
    finally:
        gated_fit.release.set()
    assert first.result().status_code == 200


# --- terminal failed ---------------------------------------------------------------------------


def test_a_failed_dataset_fetch_leaves_a_terminal_failed_record(monkeypatch, synthetic_npz_arrays):
    def _timed_out(dataset_id):
        raise JuniperDataTimeoutError("read timed out after 120s")

    _install_client(monkeypatch, synthetic_npz_arrays, download=_timed_out)
    client = TestClient(_app())
    resp = client.post("/v1/train", json={"dataset": {"dataset_id": "ds-cold"}}, headers={"X-Request-ID": "caller-a"})
    assert resp.status_code == 502, resp.text  # mapped exactly as before W1.5

    body = _status(client)
    assert body["state"] == "failed"
    assert _HEX32.match(body["operation_id"])
    assert body["operation"] == "train"
    assert body["requested_by"] == "caller-a"
    assert body["dataset_id"] == "ds-cold"
    assert body["failure"] == {"detail": resp.json()["detail"], "status_code": 502}
    assert body["model_operation_id"] is None
    assert body["busy_since"] is None
    assert body["final_metrics"] is None and body["events"] == []


def test_a_fit_that_raises_is_recorded_failed_and_the_earlier_model_survives(fake_data, monkeypatch):
    app = _app()
    client = TestClient(app, raise_server_exceptions=False)
    first = client.post("/v1/train", json=_TRAIN).json()["operation_id"]

    monkeypatch.setattr("juniper_recurrence.routers.training.TrainingLifecycle", _RaisingLifecycle)
    resp = client.post("/v1/train", json=_TRAIN, headers={"X-Request-ID": "caller-b"})
    assert resp.status_code == 500

    body = _status(client)
    assert body["state"] == "failed"
    failed = body["operation_id"]
    assert _HEX32.match(failed) and failed != first
    assert body["requested_by"] == "caller-b"
    assert body["failure"] == {"detail": "FloatingPointError: design matrix is singular", "status_code": 500}
    # The failed fit published nothing: the earlier model is still loaded, and still the one scored.
    assert body["model_operation_id"] == first
    assert client.post("/v1/predict", json={**_inline(fake_data), "expect_operation_id": first}).status_code == 200
    stale = client.post("/v1/predict", json={**_inline(fake_data), "expect_operation_id": failed})
    assert stale.status_code == 409
    assert stale.json()["detail"]["model_operation_id"] == first


def test_the_failing_fit_still_raises_exactly_as_before(fake_data, monkeypatch):
    """Recording the failure must not swallow or remap it: the same exception reaches the server."""
    monkeypatch.setattr("juniper_recurrence.routers.training.TrainingLifecycle", _RaisingLifecycle)
    with pytest.raises(FloatingPointError, match="design matrix is singular"):
        TestClient(_app()).post("/v1/train", json=_TRAIN)


def test_the_failed_record_names_the_operation_seen_in_flight(gated_fit):
    """The id a poller sees while the fit runs is the id of its terminal ``failed`` record."""
    gated_fit.error = FloatingPointError("design matrix is singular")
    app = _app()
    first = _Request(app, "POST", "/v1/train", json=_TRAIN, headers={"X-Request-ID": "caller-a"})
    first.start()
    try:
        assert gated_fit.entered.wait(_WAIT), "the fit never started"
        in_flight = _status(TestClient(app))
        assert in_flight["state"] == "training"
    finally:
        gated_fit.release.set()
    with pytest.raises(FloatingPointError):
        first.result()

    body = _status(TestClient(app))
    assert body["state"] == "failed"
    assert body["operation_id"] == in_flight["operation_id"]
    assert body["requested_by"] == "caller-a"
    assert body["failure"]["status_code"] == 500


def test_a_success_after_a_failure_reports_trained_again(fake_data, monkeypatch):
    client = TestClient(_app(), raise_server_exceptions=False)
    with monkeypatch.context() as patch:
        patch.setattr("juniper_recurrence.routers.training.TrainingLifecycle", _RaisingLifecycle)
        assert client.post("/v1/train", json=_TRAIN).status_code == 500
    assert _status(client)["state"] == "failed"

    operation_id = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    body = _status(client)
    assert body["state"] == "trained"
    assert body["operation_id"] == body["model_operation_id"] == operation_id
    assert body["failure"] is None


def test_a_busy_refusal_does_not_disturb_the_recorded_outcome(fake_data):
    """A 409 is not an operation: the status keeps describing the run that actually happened."""
    app = _app()
    client = TestClient(app)
    operation_id = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    app.state.app_state.train_lock.acquire()  # a lock taken outside the operation machinery
    try:
        refused = client.post("/v1/train", json=_TRAIN)
    finally:
        app.state.app_state.train_lock.release()
    assert refused.status_code == 409
    assert refused.json()["detail"] == {"message": "a training run is already in progress", "operation_id": None, "operation": None, "busy_since": None, "requested_by": None, "dataset_id": None}
    body = _status(client)
    assert body["state"] == "trained"
    assert body["operation_id"] == operation_id


# --- expect_operation_id on /v1/predict ----------------------------------------------------------


def test_predict_expect_operation_id_stale_is_409_naming_both_ids(fake_data):
    client = TestClient(_app())
    first = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    second = client.post("/v1/train", json=_TRAIN).json()["operation_id"]

    resp = client.post("/v1/predict", json={**_inline(fake_data), "expect_operation_id": first})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["expected_operation_id"] == first
    assert detail["model_operation_id"] == second
    assert first in detail["message"] and second in detail["message"]


def test_predict_expect_operation_id_matching_is_200(fake_data):
    client = TestClient(_app())
    operation_id = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    resp = client.post("/v1/predict", json={**_inline(fake_data), "expect_operation_id": operation_id})
    assert resp.status_code == 200, resp.text
    assert resp.json()["shape"] == [12, 1]


def test_predict_without_expect_operation_id_is_unchanged(fake_data):
    client = TestClient(_app())
    client.post("/v1/train", json=_TRAIN)
    client.post("/v1/train", json=_TRAIN)  # another fit: no id was named, so no check applies
    assert client.post("/v1/predict", json=_inline(fake_data)).status_code == 200


def test_predict_expect_operation_id_on_a_dataset_ref_is_checked_before_any_fetch(fake_data):
    client = TestClient(_app())
    client.post("/v1/train", json=_TRAIN)
    resp = client.post("/v1/predict", json={"dataset": {"dataset_id": "ds-1"}, "expect_operation_id": "0" * 32})
    assert resp.status_code == 409


def test_predict_with_no_model_keeps_its_plain_409_even_with_an_expected_id():
    resp = TestClient(_app()).post("/v1/predict", json={"X": [[[0.0, 0.0]]], "expect_operation_id": "0" * 32})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "no trained model; call POST /v1/train first"


def test_an_empty_expect_operation_id_is_a_422_not_a_silent_no_op(fake_data):
    client = TestClient(_app())
    client.post("/v1/train", json=_TRAIN)
    assert client.post("/v1/predict", json={**_inline(fake_data), "expect_operation_id": ""}).status_code == 422


# --- expect_operation_id on POST /v1/model/snapshots ----------------------------------------------


@pytest.fixture
def snap_app(tmp_path):
    return _app(snapshots_dir=tmp_path / "snaps")


def test_save_with_a_stale_expect_operation_id_is_409_and_writes_nothing(snap_app, fake_data):
    client = TestClient(snap_app)
    first = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    second = client.post("/v1/train", json=_TRAIN).json()["operation_id"]

    resp = client.post("/v1/model/snapshots", json={"expect_operation_id": first})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["expected_operation_id"] == first
    assert detail["model_operation_id"] == second
    assert client.get("/v1/model/snapshots").json()["snapshots"] == [], "a refused save must not write a snapshot"


def test_save_with_the_matching_expect_operation_id_is_201(snap_app, fake_data):
    client = TestClient(snap_app)
    operation_id = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    resp = client.post("/v1/model/snapshots", json={"expect_operation_id": operation_id, "description": "mine"})
    assert resp.status_code == 201, resp.text
    assert resp.json()["description"] == "mine"


def test_save_without_expect_operation_id_is_unchanged(snap_app, fake_data):
    client = TestClient(snap_app)
    client.post("/v1/train", json=_TRAIN)
    assert client.post("/v1/model/snapshots", json={}).status_code == 201


# --- restore: its own operation id --------------------------------------------------------------


def test_a_restore_mints_its_own_operation_id(snap_app, fake_data):
    """A restored model carries the RESTORE's id, minted at restore -- not the id of the fit it came from."""
    client = TestClient(snap_app)
    fitted = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    snapshot_id = client.post("/v1/model/snapshots", json={"expect_operation_id": fitted}).json()["id"]

    resp = client.post(f"/v1/model/snapshots/{snapshot_id}/restore", headers={"X-Request-ID": "caller-r"})
    assert resp.status_code == 200, resp.text
    restored = resp.json()["operation_id"]
    assert _HEX32.match(restored) and restored != fitted

    body = _status(client)
    assert body["state"] == "restored"
    assert body["restored_from"] == snapshot_id
    assert body["operation_id"] == body["model_operation_id"] == restored
    assert body["operation"] == "restore"
    assert body["requested_by"] == "caller-r"
    assert body["dataset_id"] is None

    assert client.post("/v1/predict", json={**_inline(fake_data), "expect_operation_id": fitted}).status_code == 409
    assert client.post("/v1/predict", json={**_inline(fake_data), "expect_operation_id": restored}).status_code == 200


def test_a_failed_restore_is_recorded_and_keeps_the_loaded_model(snap_app, fake_data, tmp_path):
    client = TestClient(snap_app)
    fitted = client.post("/v1/train", json=_TRAIN).json()["operation_id"]
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]
    (tmp_path / "snaps" / f"{snapshot_id}.npz").write_bytes(b"not an npz at all")

    resp = client.post(f"/v1/model/snapshots/{snapshot_id}/restore")
    assert resp.status_code == 422  # mapped exactly as before W1.5

    body = _status(client)
    assert body["state"] == "failed"
    assert body["operation"] == "restore"
    assert body["failure"]["status_code"] == 422
    assert body["failure"]["detail"] == resp.json()["detail"]
    assert body["model_operation_id"] == fitted
    assert body["restored_from"] is None


def test_a_restore_in_flight_is_named_by_status_and_by_a_refused_train(snap_app, fake_data, monkeypatch):
    client = TestClient(snap_app)
    client.post("/v1/train", json=_TRAIN)
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]

    gate = _Gate()

    class _GatedSerializer(LMUSerializer):
        def load(self, *args, **kwargs):
            gate.hold()
            return super().load(*args, **kwargs)

    monkeypatch.setattr("juniper_recurrence.routers.snapshots.LMUSerializer", _GatedSerializer)
    restore = _Request(snap_app, "POST", f"/v1/model/snapshots/{snapshot_id}/restore", headers={"X-Request-ID": "caller-r"})
    restore.start()
    try:
        assert gate.entered.wait(_WAIT), "the restore never started loading"
        in_flight = _status(client)
        assert in_flight["state"] == "restoring"
        assert in_flight["operation"] == "restore"
        _parse_utc(in_flight["busy_since"])

        refused = client.post("/v1/train", json=_TRAIN)
        assert refused.status_code == 409
        detail = refused.json()["detail"]
        assert detail["message"] == "a snapshot restore is in progress"
        assert detail["operation"] == "restore"
        assert detail["operation_id"] == in_flight["operation_id"]
        assert detail["requested_by"] == "caller-r"
    finally:
        gate.release.set()
    done = restore.result()
    assert done.status_code == 200, done.text
    assert done.json()["operation_id"] == in_flight["operation_id"]


# --- published contract -------------------------------------------------------------------------


def test_openapi_publishes_the_operation_identity_contract():
    schema = _app().openapi()
    components = schema["components"]["schemas"]

    assert "operation_id" in components["TrainResponse"]["required"]
    assert "operation_id" in components["RestoreResponse"]["required"]
    status_props = components["StatusResponse"]["properties"]
    for name in ("state", "final_metrics", "stopped_reason", "events", "restored_from"):
        assert name in status_props, f"pre-W1.5 status field {name} must survive"
    for name in ("operation_id", "operation", "busy_since", "dataset_id", "requested_by", "model_operation_id", "failure"):
        assert name in status_props, name
    assert "expect_operation_id" in components["PredictRequest"]["properties"]
    assert "expect_operation_id" in components["SnapshotRequest"]["properties"]

    train_post = schema["paths"]["/v1/train"]["post"]
    assert train_post["responses"]["409"]["content"]["application/json"]["schema"]["$ref"].endswith("/BusyResponse")
    assert any(param["in"] == "header" and param["name"].lower() == "x-request-id" for param in train_post["parameters"])
    for path in ("/v1/predict", "/v1/model/snapshots"):
        conflict = schema["paths"][path]["post"]["responses"]["409"]["content"]["application/json"]["schema"]
        assert conflict["$ref"].endswith("/OperationConflictResponse"), path


# --- AppState unit behaviour ----------------------------------------------------------------------


def test_try_begin_is_exclusive_and_names_the_holder():
    state = AppState()
    first, holder = state.try_begin("train", requested_by="caller-a", dataset_id="ds-1")
    assert first is not None and holder is None
    second, holder = state.try_begin("restore")
    assert second is None
    assert holder == first
    state.end(first)
    third, holder = state.try_begin("restore")
    assert third is not None and holder is None and third.operation_id != first.operation_id
    state.end(third)
    assert not state.train_lock.locked()


def test_try_begin_against_a_lock_taken_outside_the_api_names_no_holder():
    state = AppState()
    state.train_lock.acquire()
    try:
        assert state.try_begin("train") == (None, None)
    finally:
        state.train_lock.release()


def test_end_releases_the_lock_without_clearing_another_operations_record():
    state = AppState()
    stale = Operation(operation_id="0" * 32, kind="train", started_at="2026-10-05T00:00:00.000Z")
    state.train_lock.acquire()
    state.end(stale)  # not the active operation: nothing to clear, the lock is still released
    assert not state.train_lock.locked()

    current, _ = state.try_begin("train")
    state.note_dataset(stale, "ds-stale")  # a stale operation cannot rewrite the active record
    assert state.status_snapshot().operation == current
    state.fail(stale, detail="stale", status_code=500)
    assert state.status_snapshot().state == "training", "an operation holding the lock outranks any failure"
    state.end(current)
    assert state.status_snapshot().failure.operation == stale


def test_the_pre_w1_5_status_tuple_and_restored_from_still_work(fake_data):
    app = _app()
    TestClient(app).post("/v1/train", json=_TRAIN)
    state = app.state.app_state
    name, result, events = state.status()
    assert name == "trained"
    assert result is not None and result.n_epochs == 1
    assert [event.type for event in events] == ["training_start", "epoch_end", "training_end"]
    assert state.restored_from is None


def test_the_pre_w1_5_status_tuple_is_the_snapshot_in_every_state(snap_app, fake_data, monkeypatch):
    """``AppState.status()`` became a thin view of ``status_snapshot()`` (W1.5); its contract did not shrink.

    It returns the same ``(state, result, events)`` it always did for ``idle`` / ``trained`` /
    ``restored`` -- result and events only for a fit -- and reports the new ``failed`` state the
    same way the route does, so a caller of the old method is never told "trained" about a fit
    that raised.
    """
    client = TestClient(snap_app, raise_server_exceptions=False)
    state = snap_app.state.app_state

    def _agrees() -> tuple:
        snapshot = state.status_snapshot()
        legacy = state.status()
        assert legacy == (snapshot.state, snapshot.result, snapshot.events)
        assert legacy[0] == _status(client)["state"]
        return legacy

    assert _agrees() == ("idle", None, [])

    client.post("/v1/train", json=_TRAIN)
    name, result, events = _agrees()
    assert name == "trained" and result is not None and len(events) == 3

    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]
    client.post(f"/v1/model/snapshots/{snapshot_id}/restore")
    assert _agrees() == ("restored", None, [])

    monkeypatch.setattr("juniper_recurrence.routers.training.TrainingLifecycle", _RaisingLifecycle)
    assert client.post("/v1/train", json=_TRAIN).status_code == 500
    assert _agrees() == ("failed", None, [])
