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
