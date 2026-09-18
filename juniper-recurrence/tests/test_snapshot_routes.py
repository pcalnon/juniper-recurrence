"""Route tests for model snapshots: save / list / get / restore.

Design of record: juniper-ml
``notes/JUNIPER_2026-09-16_JUNIPER-RECURRENCE_MODEL-PERSISTENCE-DESIGN.md``. These close Y2 of
the canopy selection-reachability arc, where a snapshot workflow reported success at both ends
while never writing or reading model state.

The data path is faked (the shared ``fake_data`` fixture) but the model, the serializer and the
filesystem are real — a snapshot here is a genuine ``.npz`` that a genuine ``LMUSerializer``
round-trips. That is the point: the defect being fixed was a workflow that looked real and was
not, so a test that mocked the serializer would reproduce the defect rather than catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from juniper_recurrence.app import build_app
from juniper_recurrence.routers.snapshots import _safe_id
from juniper_recurrence.settings import Settings


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A client whose snapshots land in a per-test tmp dir."""
    return TestClient(build_app(Settings(api_keys=None, snapshots_dir=tmp_path / "snaps")))


def _train(client: TestClient) -> None:
    resp = client.post("/v1/train", json={"dataset": {"dataset_id": "ds-1"}})
    assert resp.status_code == 200, resp.text


# --- save -----------------------------------------------------------------------------


def test_save_refuses_when_no_model(client):
    """409, not 500.

    ``LMUSerializer.save`` raises ``RuntimeError`` on an unfitted model. Letting that surface
    would turn a precondition failure into an internal error, so the route checks first —
    mirroring ``GET /v1/model``'s existing refusal.
    """
    resp = client.post("/v1/model/snapshots", json={})
    assert resp.status_code == 409
    assert "no trained model" in resp.json()["detail"]


def test_save_writes_a_real_npz(client, fake_data, tmp_path):
    _train(client)
    resp = client.post("/v1/model/snapshots", json={"description": "after the first fit"})
    assert resp.status_code == 201, resp.text
    body = resp.json()

    assert body["id"].startswith("lmu-")
    assert body["description"] == "after the first fit"
    assert body["size_bytes"] > 0

    # THE regression: a file that actually carries model state, not a metadata stub. The old
    # canopy fallback wrote timestamps and hyperparameters and reported success, which is the
    # defect this whole feature exists to close.
    written = tmp_path / "snaps" / f"{body['id']}.npz"
    assert written.is_file()
    with np.load(written, allow_pickle=False) as data:
        assert any(key.startswith("readout__") for key in data.files), "no readout state in the snapshot"
        assert "meta" in data.files
    assert body["meta"]["schema"] == 2
    assert body["meta"]["d"] >= 1


# --- list / get -----------------------------------------------------------------------


def test_list_is_empty_before_any_save(client):
    assert client.get("/v1/model/snapshots").json()["snapshots"] == []


def test_list_returns_newest_first(client, fake_data):
    _train(client)
    first = client.post("/v1/model/snapshots", json={"description": "one"}).json()["id"]
    second = client.post("/v1/model/snapshots", json={"description": "two"}).json()["id"]
    assert first != second, "ids must be unique per save"

    listed = client.get("/v1/model/snapshots").json()["snapshots"]
    assert [s["id"] for s in listed][:2] == [second, first]


def test_get_unknown_snapshot_is_404(client):
    assert client.get("/v1/model/snapshots/lmu-nope").status_code == 404


@pytest.mark.parametrize(
    "bad",
    ["../etc/passwd", "..", ".", "a/b", "", ".hidden", "lmu-x/../../etc/passwd", "lmu-x\x00", "lmu x", "l" * 65],
)
def test_the_id_guard_rejects_anything_that_is_not_an_id(bad):
    """Tested at the GUARD, not through the URL — deliberately.

    A first version of this asserted 404 from the routes for these inputs and "passed" for three
    of them for the wrong reason: Starlette normalises the path BEFORE routing, so ``""`` reaches
    the list route (200) and ``".."`` resolves to ``/v1/model`` (409). Neither input ever reached
    the handler, so neither exercised the guard. Calling ``_safe_id`` directly is the only way to
    assert what it actually rejects.

    The guard is an anchored allowlist rather than a denylist of traversal spellings: ``..``,
    ``%2e%2e`` and unicode homoglyphs are an unbounded set, and a character class is not.
    """
    with pytest.raises(HTTPException) as excinfo:
        _safe_id(bad)
    assert excinfo.value.status_code == 404


@pytest.mark.parametrize("ok", ["lmu-20260917T101500123456", "a", "A-1_b", "l" * 64])
def test_the_id_guard_accepts_a_real_id(ok):
    # Without this the guard could reject everything and the test above would still pass.
    assert _safe_id(ok) == ok


def test_a_symlink_out_of_the_directory_is_refused(client, fake_data, tmp_path):
    """The containment check, which the NAME check cannot do.

    A legal-looking id whose file is a symlink pointing outside ``snapshots_dir`` passes
    ``_safe_id`` — the name is fine, it is the target that is not. ``_snapshot_path`` resolves
    before comparing, so the link is followed and the escape caught. This is why there are two
    checks rather than one, and it is the half that survives the allowlist being loosened.
    """
    _train(client)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"not yours")
    snaps = tmp_path / "snaps"
    snaps.mkdir(parents=True, exist_ok=True)
    (snaps / "lmu-escape.npz").symlink_to(outside)

    assert client.get("/v1/model/snapshots/lmu-escape").status_code == 404
    assert client.post("/v1/model/snapshots/lmu-escape/restore").status_code == 404


def test_an_unroutable_id_is_refused_at_the_route_too(client):
    """The end-to-end half, using an id that survives URL normalisation.

    ``lmu-..foo`` reaches the handler intact (no path segment is ``..``), so this exercises the
    route's use of the guard rather than Starlette's normaliser.
    """
    assert client.get("/v1/model/snapshots/lmu-..foo").status_code == 404
    assert client.post("/v1/model/snapshots/lmu-..foo/restore").status_code == 404


# --- restore --------------------------------------------------------------------------


def test_restore_round_trips_the_model(client, fake_data):
    """The whole point: predictions survive the round trip.

    A snapshot that restores a DIFFERENT model is the same lie as one that restores nothing,
    so this asserts on predictions rather than on the restore call returning 200.
    """
    _train(client)
    before = client.post("/v1/predict", json={"X": fake_data["X_train"].tolist(), "dt": fake_data["dt_train"].tolist()})
    assert before.status_code == 200, before.text
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]

    # Retrain to move the in-process model off the saved one, so a no-op restore cannot pass.
    _train(client)

    resp = client.post(f"/v1/model/snapshots/{snapshot_id}/restore")
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "restored"
    assert resp.json()["restored_from"] == snapshot_id

    after = client.post("/v1/predict", json={"X": fake_data["X_train"].tolist(), "dt": fake_data["dt_train"].tolist()})
    assert after.status_code == 200, after.text
    np.testing.assert_allclose(
        np.asarray(after.json()["predictions"], dtype=float),
        np.asarray(before.json()["predictions"], dtype=float),
        rtol=1e-6,
        atol=1e-8,
    )


def test_restore_reports_the_third_state_and_invents_no_run(client, fake_data):
    """``restored`` is not a flavour of ``trained``.

    The model is present and predictable, but this process never fitted it — so there is no
    result and no event stream, and none is synthesised to make the status shape uniform.
    Synthesising one would report a run that never happened, which is the defect class Y2 is
    about.
    """
    _train(client)
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]
    client.post(f"/v1/model/snapshots/{snapshot_id}/restore")

    body = client.get("/v1/training/status").json()
    assert body["state"] == "restored"
    assert body["restored_from"] == snapshot_id
    assert body["final_metrics"] is None
    assert body["stopped_reason"] is None
    assert body["events"] == []


def test_the_models_own_metrics_survive_a_restore(client, fake_data):
    """Withholding the RUN must not withhold the MODEL's metrics.

    ``GET /v1/model`` reports ``model.metrics()``, which travels inside the snapshot. So a
    restored model is still fully described — the status omits the run, not the model.
    """
    _train(client)
    before = client.get("/v1/model").json()
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]
    client.post(f"/v1/model/snapshots/{snapshot_id}/restore")

    after = client.get("/v1/model").json()
    assert set(after["metrics"]) == set(before["metrics"])
    assert "accuracy" not in after["metrics"]  # RK-6


def test_a_fit_after_a_restore_reports_trained_again(client, fake_data):
    """A fit SUPERSEDES a restore.

    If the marker survived, a freshly-fitted model would keep reporting itself as loaded from
    disk — the mirror image of the original defect.
    """
    _train(client)
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]
    client.post(f"/v1/model/snapshots/{snapshot_id}/restore")
    assert client.get("/v1/training/status").json()["state"] == "restored"

    _train(client)
    body = client.get("/v1/training/status").json()
    assert body["state"] == "trained"
    assert body["restored_from"] is None
    assert body["final_metrics"] is not None


def test_restore_of_a_corrupt_file_is_422_not_500(client, fake_data, tmp_path):
    """A file that is not one of ours is a bad request, not an internal error."""
    _train(client)
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]
    (tmp_path / "snaps" / f"{snapshot_id}.npz").write_bytes(b"not an npz at all")

    resp = client.post(f"/v1/model/snapshots/{snapshot_id}/restore")
    assert resp.status_code == 422
    assert "could not be loaded" in resp.json()["detail"]


def test_restore_is_refused_while_a_fit_holds_the_lock(client, fake_data):
    """Otherwise the training thread publishes over the restore moments later.

    The caller would be told the restore succeeded and then silently get the fitted model.
    """
    _train(client)
    snapshot_id = client.post("/v1/model/snapshots", json={}).json()["id"]

    app_state = client.app.state.app_state
    assert app_state.train_lock.acquire(blocking=False)
    try:
        resp = client.post(f"/v1/model/snapshots/{snapshot_id}/restore")
    finally:
        app_state.train_lock.release()
    assert resp.status_code == 409
    assert "training run is in progress" in resp.json()["detail"]


# --- retention ------------------------------------------------------------------------


def test_nothing_prunes(client, fake_data):
    """Retention is INHERITED from §6.4's ratified no-deletion (juniper-ml#1296).

    That ruling also says do not build deletion tooling. This pins the absence: saving many
    snapshots must not age any out, so a future "tidy up" has to fail here first.
    """
    _train(client)
    ids = [client.post("/v1/model/snapshots", json={}).json()["id"] for _ in range(6)]
    listed = {s["id"] for s in client.get("/v1/model/snapshots").json()["snapshots"]}
    assert listed == set(ids), "a snapshot disappeared; retention is no-deletion"
