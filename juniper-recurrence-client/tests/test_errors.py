"""HTTP/transport error-mapping tests (status code -> typed exception)."""

from __future__ import annotations

import pytest
import responses

from juniper_recurrence_client import (
    JuniperRecurrenceClient,
    JuniperRecurrenceClientError,
    JuniperRecurrenceConflictError,
    JuniperRecurrenceConnectionError,
    JuniperRecurrenceNotFoundError,
    JuniperRecurrenceValidationError,
)

BASE_URL = "http://recurrence.test:8211"


def _client() -> JuniperRecurrenceClient:
    return JuniperRecurrenceClient(base_url=BASE_URL, retries=0)


@responses.activate
def test_404_maps_to_not_found() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/model", json={"detail": "no model"}, status=404)
    with pytest.raises(JuniperRecurrenceNotFoundError, match="no model"):
        _client().get_model()


@responses.activate
def test_409_maps_to_conflict() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/train", json={"detail": "training already in progress"}, status=409)
    with pytest.raises(JuniperRecurrenceConflictError, match="in progress"):
        _client().train(name="equities")


@responses.activate
def test_422_maps_to_validation() -> None:
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"detail": "n_folds must be >= 2"}, status=422)
    with pytest.raises(JuniperRecurrenceValidationError, match="n_folds"):
        _client().crossval(name="equities", n_folds=1)


@responses.activate
def test_500_maps_to_client_error() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/dataset", json={"detail": "boom"}, status=500)
    with pytest.raises(JuniperRecurrenceClientError, match="500"):
        _client().get_dataset()


@responses.activate
def test_connection_error_maps() -> None:
    # No response registered for this URL -> responses raises a ConnectionError.
    with pytest.raises(JuniperRecurrenceConnectionError):
        _client().get_model()


@responses.activate
def test_malformed_json_raises_client_error() -> None:
    responses.add(responses.GET, f"{BASE_URL}/v1/dataset", body="not-json", status=200, content_type="application/json")
    with pytest.raises(JuniperRecurrenceClientError, match="Malformed JSON"):
        _client().get_dataset()


@responses.activate
def test_non_object_json_body_raises_client_error() -> None:
    # A syntactically valid body that is not a JSON object breaks every caller's
    # declared dict[str, Any]; _parse_json rejects it with the typed error rather
    # than letting it surface as an AttributeError downstream (APD-RCLIENT-003).
    responses.add(responses.GET, f"{BASE_URL}/v1/dataset", json=[1, 2, 3], status=200)
    with pytest.raises(JuniperRecurrenceClientError, match="Expected a JSON object"):
        _client().get_dataset()


@responses.activate
def test_non_special_error_status_maps_to_base_client_error() -> None:
    # 403 is neither retryable (not in RETRYABLE_STATUS_CODES) nor one of the specially mapped
    # statuses (404/409/400/422), so the generic ``else`` arm raises the base client error with
    # the JSON ``detail`` and the status code in the message.
    responses.add(responses.GET, f"{BASE_URL}/v1/model", json={"detail": "forbidden"}, status=403)
    with pytest.raises(JuniperRecurrenceClientError, match=r"403.*forbidden"):
        _client().get_model()


@responses.activate
def test_error_body_not_json_falls_back_to_raw_text() -> None:
    # An error response whose body is not JSON: ``response.json()`` raises, so detail extraction
    # falls back to the raw response text. 501 is non-retryable and non-special -> the ``else`` arm.
    responses.add(responses.GET, f"{BASE_URL}/v1/dataset", body="upstream said no", status=501, content_type="text/plain")
    with pytest.raises(JuniperRecurrenceClientError, match="upstream said no"):
        _client().get_dataset()


# ---------------------------------------------------------------------------
# APD-RCLIENT-001: exceptions must carry machine-readable context.
#
# Every exception subclassed ``Exception`` with nothing on it, so a 400 and a
# 422 raised the same type with the same text and the only way to tell them
# apart was substring-matching the message. Ported from juniper-data-client#158;
# the three clients are separately released packages with no shared code, so
# nothing mechanical keeps them aligned -- these tests are the alignment.
# ---------------------------------------------------------------------------

#: A real FastAPI 422 body: ``detail`` is a list of error objects, not a string.
FASTAPI_422_DETAIL = [
    {"type": "missing", "loc": ["body", "seed"], "msg": "Field required"},
    {"type": "int_parsing", "loc": ["body", "n_folds"], "msg": "Input should be a valid integer"},
]


@responses.activate
def test_status_code_separates_400_from_422() -> None:
    """The core of APD-RCLIENT-001: these two were byte-identical."""
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"detail": "bad"}, status=400)
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"detail": "bad"}, status=422)

    client = _client()
    with pytest.raises(JuniperRecurrenceValidationError) as first:
        client.crossval(name="equities", n_folds=1)
    with pytest.raises(JuniperRecurrenceValidationError) as second:
        client.crossval(name="equities", n_folds=1)

    assert {first.value.status_code, second.value.status_code} == {400, 422}


@responses.activate
def test_every_mapped_branch_carries_its_status() -> None:
    """404, 409 and the generic ``else`` arm populate the context too."""
    responses.add(responses.GET, f"{BASE_URL}/v1/model", json={"detail": "no model"}, status=404)
    responses.add(responses.POST, f"{BASE_URL}/v1/train", json={"detail": "already running"}, status=409)
    responses.add(responses.GET, f"{BASE_URL}/v1/dataset", json={"detail": "teapot"}, status=418)

    client = _client()

    with pytest.raises(JuniperRecurrenceNotFoundError) as not_found:
        client.get_model()
    assert not_found.value.status_code == 404
    assert not_found.value.detail == "no model"

    with pytest.raises(JuniperRecurrenceConflictError) as conflict:
        client.train(name="equities")
    assert conflict.value.status_code == 409

    with pytest.raises(JuniperRecurrenceClientError) as generic:
        client.get_dataset()
    assert generic.value.status_code == 418


@responses.activate
def test_422_detail_list_is_preserved_as_structure() -> None:
    """The caller gets the list itself, not a rendering of it."""
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"detail": FASTAPI_422_DETAIL}, status=422)

    with pytest.raises(JuniperRecurrenceValidationError) as excinfo:
        _client().crossval(name="equities", n_folds=1)

    assert excinfo.value.detail == FASTAPI_422_DETAIL
    assert excinfo.value.detail[0]["loc"] == ["body", "seed"]


@responses.activate
def test_422_message_is_readable_not_a_python_repr() -> None:
    """Previously the list was f-string-interpolated, so the message was a repr.

    This is the same defect juniper-data-client tracks as APD-DCLIENT-003; it
    was never recorded against this client.
    """
    responses.add(responses.POST, f"{BASE_URL}/v1/crossval", json={"detail": FASTAPI_422_DETAIL}, status=422)

    with pytest.raises(JuniperRecurrenceValidationError) as excinfo:
        _client().crossval(name="equities", n_folds=1)

    message = str(excinfo.value)
    assert "body.seed: Field required" in message
    assert "body.n_folds: Input should be a valid integer" in message
    # Fingerprints of the old repr-of-a-list behaviour.
    assert "'type':" not in message
    assert "[{" not in message


@responses.activate
def test_response_is_attached_for_header_access() -> None:
    """``response`` reaches headers the message cannot carry."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/model",
        json={"detail": "no model"},
        status=404,
        headers={"X-Request-ID": "abc123"},
    )

    with pytest.raises(JuniperRecurrenceNotFoundError) as excinfo:
        _client().get_model()

    assert excinfo.value.response is not None
    assert excinfo.value.response.headers["X-Request-ID"] == "abc123"


def test_locally_raised_errors_have_no_status_code() -> None:
    """Backward compatibility: no HTTP response means the fields stay None."""
    error = JuniperRecurrenceClientError("something local went wrong")

    assert error.status_code is None
    assert error.detail is None
    assert error.response is None
    assert str(error) == "something local went wrong"


def test_positional_message_construction_still_works() -> None:
    """The added parameters are keyword-only, so existing call sites are safe."""
    for factory in (JuniperRecurrenceClientError, JuniperRecurrenceNotFoundError, JuniperRecurrenceValidationError):
        error = factory("plain message")
        assert str(error) == "plain message"
        assert error.status_code is None


def test_context_survives_pickle_and_copy() -> None:
    """``BaseException.__reduce__`` rebuilds from ``args``, which holds only the
    message -- so without the override a round-trip returns an exception that
    looks correct and has lost the context (flake8-bugbear B042).
    """
    import copy as copy_module

    # Bandit blacklists pickle (B403/B301) for UNTRUSTED data; the payload here
    # is produced by the ``dumps`` below, in-process, from an exception this
    # test just built. The suppressions are the trailing inline markers only --
    # a comment line that *begins* with the marker word is itself parsed as a
    # directive, and the following prose is read as test IDs.
    import pickle  # nosec B403

    original = JuniperRecurrenceValidationError("Validation error (422)", status_code=422, detail=[{"msg": "Field required"}])
    round_tripped = pickle.loads(pickle.dumps(original))  # nosec B301

    for rebuilt in (round_tripped, copy_module.copy(original), copy_module.deepcopy(original)):
        assert isinstance(rebuilt, JuniperRecurrenceValidationError)
        assert rebuilt.status_code == 422
        assert rebuilt.detail == [{"msg": "Field required"}]
        assert str(rebuilt) == str(original)
