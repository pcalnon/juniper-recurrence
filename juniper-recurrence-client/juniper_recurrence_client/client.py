"""REST API client for the juniper-recurrence service.

A lean ``requests``-based client wrapping the juniper-recurrence FastAPI app's REST surface
(train / predict / cross-validate / inspect / health), for consumers such as juniper-canopy's
recurrence backend adapter. Mirrors juniper-data-client's transport machinery: an idempotent-only
retry policy, ``X-API-Key`` auth with ``_FILE`` Docker-secret indirection, typed exceptions, the
optional ``on_request`` instrumentation hook, and best-effort ``X-Request-ID`` propagation.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from juniper_recurrence_client.constants import (
    API_KEY_ENV_VAR,
    API_KEY_FILE_ENV_VAR,
    API_KEY_HEADER_NAME,
    API_VERSION_PATH_SUFFIX,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BASE_URL,
    DEFAULT_READY_POLL_INTERVAL,
    DEFAULT_READY_TIMEOUT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_URL_SCHEME_PREFIX,
    ENDPOINT_CROSSVAL,
    ENDPOINT_CROSSVAL_STATUS,
    ENDPOINT_DATASET,
    ENDPOINT_HEALTH,
    ENDPOINT_HEALTH_READY,
    ENDPOINT_MODEL,
    ENDPOINT_PREDICT,
    ENDPOINT_TRAIN,
    ENDPOINT_TRAINING_STATUS,
    HEALTH_READY_STATUS,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_MAXSIZE,
    RETRY_ALLOWED_METHODS,
    RETRYABLE_STATUS_CODES,
    URL_SCHEME_PREFIXES,
)
from juniper_recurrence_client.exceptions import (
    JuniperRecurrenceClientError,
    JuniperRecurrenceConflictError,
    JuniperRecurrenceConnectionError,
    JuniperRecurrenceNotFoundError,
    JuniperRecurrenceTimeoutError,
    JuniperRecurrenceValidationError,
)

logger = logging.getLogger("juniper_recurrence_client.client")


def _resolve_api_key_from_env() -> Optional[str]:
    """Resolve the juniper-recurrence API key from the environment.

    Honors the Docker-secret ``JUNIPER_RECURRENCE_API_KEY_FILE`` indirection (a file whose
    stripped contents are the key) before the plain ``JUNIPER_RECURRENCE_API_KEY`` env var, so a
    consumer that mounts the key as a file and leaves ``api_key`` unset still authenticates.
    """
    file_path = os.environ.get(API_KEY_FILE_ENV_VAR)
    if file_path:
        try:
            content = Path(file_path).read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        if content:
            return content
    return os.environ.get(API_KEY_ENV_VAR)


#: Optional instrumentation hook, invoked once per HTTP call with
#: ``(method, url, status, duration_ms, error)``. ``error is None`` is the canonical success
#: signal (``status`` may be set even on the typed-error paths). Mirrors juniper-data-client so
#: canopy/cascor can pass the same Prometheus/structured-log closure they already use.
RequestHook = Callable[[str, str, Optional[int], float, Optional[BaseException]], None]


def _noop_request_hook(
    method: str,
    url: str,
    status: Optional[int],
    duration_ms: float,
    error: Optional[BaseException],
) -> None:
    """Default :data:`RequestHook` — does nothing (named so the default is a real callable)."""


def _dataset_ref(
    *,
    dataset_id: Optional[str],
    name: Optional[str],
    generator: Optional[str],
    params: Optional[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Build the app's ``DatasetRef`` body from selection kwargs.

    Exactly one of ``dataset_id`` / ``name`` / ``generator`` is expected; the server validates
    that invariant and returns 422 otherwise.
    """
    ref: dict[str, Any] = {"split": split}
    if dataset_id is not None:
        ref["dataset_id"] = dataset_id
    if name is not None:
        ref["name"] = name
    if generator is not None:
        ref["generator"] = generator
    if params is not None:
        ref["params"] = params
    return ref


class JuniperRecurrenceClient:
    """Client for the juniper-recurrence REST API (train / predict / cross-validate / inspect).

    Automatic retry (idempotent methods only), connection pooling, and ``X-API-Key`` auth.

    Example:
        >>> client = JuniperRecurrenceClient("http://localhost:8211")
        >>> client.train(name="equities", d=16)
        >>> preds = client.predict(dataset_id="ds-1")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        api_key: Optional[str] = None,
        on_request: Optional[RequestHook] = None,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the juniper-recurrence app (default ``http://localhost:8211``).
            timeout: Per-request timeout in seconds.
            retries: Retry attempts for transient failures on idempotent methods.
            backoff_factor: Exponential backoff factor for retries.
            api_key: API key for ``X-API-Key`` auth. If unset, resolved from
                ``JUNIPER_RECURRENCE_API_KEY`` (and its ``_FILE`` form).
            on_request: Optional instrumentation hook (see :data:`RequestHook`); defaults to a
                no-op. Hook exceptions are caught and logged so instrumentation never crashes a
                request path.
        """
        self.base_url = self._normalize_url(base_url)
        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session()
        self._on_request: RequestHook = on_request or _noop_request_hook

        resolved_api_key = api_key or _resolve_api_key_from_env()
        if resolved_api_key:
            self.session.headers[API_KEY_HEADER_NAME] = resolved_api_key

    def _normalize_url(self, url: str) -> str:
        """Normalize the base URL: ensure a scheme, drop a trailing slash and any ``/v1`` suffix."""
        url = url.strip()
        if not url.startswith(URL_SCHEME_PREFIXES):
            url = f"{DEFAULT_URL_SCHEME_PREFIX}{url}"
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        if normalized.endswith(API_VERSION_PATH_SUFFIX):
            normalized = normalized[: -len(API_VERSION_PATH_SUFFIX)]
        return normalized

    def _create_session(self) -> requests.Session:
        """Create a ``requests.Session`` with the idempotent-only retry policy + pooling."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=RETRYABLE_STATUS_CODES,
            allowed_methods=RETRY_ALLOWED_METHODS,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=HTTP_POOL_CONNECTIONS,
            pool_maxsize=HTTP_POOL_MAXSIZE,
        )
        for scheme in URL_SCHEME_PREFIXES:
            session.mount(scheme, adapter)
        return session

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:  # noqa: C901
        """Make an HTTP request, mapping transport/HTTP errors to typed exceptions.

        Raises:
            JuniperRecurrenceConnectionError / JuniperRecurrenceTimeoutError: transport failures.
            JuniperRecurrenceNotFoundError (404), JuniperRecurrenceConflictError (409),
            JuniperRecurrenceValidationError (400/422), or JuniperRecurrenceClientError (other).
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)

        # Best-effort X-Request-ID propagation via juniper-observability (no-op if absent or
        # unset); a caller-supplied X-Request-ID always wins.
        headers = dict(kwargs.get("headers") or {})
        if "X-Request-ID" not in headers:
            try:
                from juniper_observability import request_id_var  # noqa: PLC0415

                rid = request_id_var.get()
                if rid:
                    headers["X-Request-ID"] = rid
                    kwargs["headers"] = headers
            except (ImportError, LookupError):
                pass

        start = time.monotonic()
        response: Optional[requests.Response] = None
        outgoing_error: Optional[BaseException] = None
        try:
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.exceptions.ConnectionError as e:
                outgoing_error = JuniperRecurrenceConnectionError(f"Failed to connect to juniper-recurrence at {self.base_url}: {e}")
                raise outgoing_error from e
            except requests.exceptions.Timeout as e:
                outgoing_error = JuniperRecurrenceTimeoutError(f"Request to {url} timed out after {self.timeout}s: {e}")
                raise outgoing_error from e
            except requests.exceptions.RequestException as e:
                outgoing_error = JuniperRecurrenceClientError(f"Request failed: {e}")
                raise outgoing_error from e

            if response.ok:
                return response

            error_detail = response.text
            try:
                error_json = response.json()
                if "detail" in error_json:
                    error_detail = error_json["detail"]
            except (ValueError, KeyError):
                error_detail = response.text

            if response.status_code == HTTP_404_NOT_FOUND:
                outgoing_error = JuniperRecurrenceNotFoundError(f"Resource not found: {error_detail}")
                raise outgoing_error
            elif response.status_code == HTTP_409_CONFLICT:
                outgoing_error = JuniperRecurrenceConflictError(f"Conflict: {error_detail}")
                raise outgoing_error
            elif response.status_code in (HTTP_400_BAD_REQUEST, HTTP_422_UNPROCESSABLE_ENTITY):
                outgoing_error = JuniperRecurrenceValidationError(f"Validation error: {error_detail}")
                raise outgoing_error
            else:
                outgoing_error = JuniperRecurrenceClientError(f"Request failed ({response.status_code}): {error_detail}")
                raise outgoing_error
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            status = response.status_code if response is not None else None
            try:
                self._on_request(method, url, status, duration_ms, outgoing_error)
            except Exception:  # noqa: BLE001 — instrumentation must not crash production paths
                logger.warning("on_request hook raised; suppressed to keep request path resilient", exc_info=True)

    @staticmethod
    def _parse_json(response: requests.Response) -> Any:
        """Parse a response body as JSON, surfacing a typed error on a malformed body."""
        try:
            return response.json()
        except ValueError as e:
            preview = (response.text or "")[:200]
            raise JuniperRecurrenceClientError(f"Malformed JSON response from {response.url}: {e}: {preview!r}") from e

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        *,
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
        generator: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        split: str = "train",
        d: Optional[int] = None,
        theta: Optional[float] = None,
        ridge: Optional[float] = None,
    ) -> dict[str, Any]:
        """``POST /v1/train`` — synchronously fit the LMU regressor on a dataset split.

        Supply exactly one of ``dataset_id`` / ``name`` / ``generator``. Returns the
        ``TrainResponse`` (``final_metrics``, ``n_epochs``, ``stopped_reason``, ``dataset``).
        Raises :class:`JuniperRecurrenceConflictError` (409) if a run is already in progress.
        """
        body: dict[str, Any] = {"dataset": _dataset_ref(dataset_id=dataset_id, name=name, generator=generator, params=params, split=split)}
        if d is not None:
            body["d"] = d
        if theta is not None:
            body["theta"] = theta
        if ridge is not None:
            body["ridge"] = ridge
        return self._parse_json(self._request("POST", ENDPOINT_TRAIN, json=body))

    def training_status(self) -> dict[str, Any]:
        """``GET /v1/training/status`` — current training state, metrics, and emitted events."""
        return self._parse_json(self._request("GET", ENDPOINT_TRAINING_STATUS))

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict(
        self,
        *,
        X: Optional[Any] = None,
        dt: Optional[Any] = None,
        target_dt: Optional[Any] = None,
        seq_lengths: Optional[Any] = None,
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
        generator: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        split: str = "train",
    ) -> dict[str, Any]:
        """``POST /v1/predict`` — predictions from the trained model.

        Supply exactly one of inline ``X`` (optionally with ``dt`` / ``target_dt`` /
        ``seq_lengths``) or a dataset reference. Returns ``{"predictions": ..., "shape": ...}``.
        """
        body: dict[str, Any] = {}
        if X is not None:
            body["X"] = X
            if dt is not None:
                body["dt"] = dt
            if target_dt is not None:
                body["target_dt"] = target_dt
            if seq_lengths is not None:
                body["seq_lengths"] = seq_lengths
        if dataset_id is not None or name is not None or generator is not None:
            body["dataset"] = _dataset_ref(dataset_id=dataset_id, name=name, generator=generator, params=params, split=split)
        return self._parse_json(self._request("POST", ENDPOINT_PREDICT, json=body))

    # ─── Cross-validation ──────────────────────────────────────────────────────

    def crossval(
        self,
        *,
        n_folds: int,
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
        generator: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        split: str = "train",
        scheme: str = "expanding",
        embargo: int = 0,
        min_train: Optional[int] = None,
        d: Optional[int] = None,
        theta: Optional[float] = None,
        ridge: Optional[float] = None,
    ) -> dict[str, Any]:
        """``POST /v1/crossval`` — synchronous walk-forward cross-validation over the ``_full`` split.

        Returns the ``CrossValResponse`` (per-fold ``folds`` + ``eval_aggregate`` / ``eval_std``).
        ``scheme`` is ``"expanding"`` or ``"rolling"``. Raises 409 if a CV run is already running.
        """
        body: dict[str, Any] = {
            "dataset": _dataset_ref(dataset_id=dataset_id, name=name, generator=generator, params=params, split=split),
            "n_folds": n_folds,
            "scheme": scheme,
            "embargo": embargo,
        }
        if min_train is not None:
            body["min_train"] = min_train
        if d is not None:
            body["d"] = d
        if theta is not None:
            body["theta"] = theta
        if ridge is not None:
            body["ridge"] = ridge
        return self._parse_json(self._request("POST", ENDPOINT_CROSSVAL, json=body))

    def crossval_status(self) -> dict[str, Any]:
        """``GET /v1/crossval/status`` — the most recent cross-validation result, if any."""
        return self._parse_json(self._request("GET", ENDPOINT_CROSSVAL_STATUS))

    # ─── Inspection ────────────────────────────────────────────────────────────

    def get_model(self) -> dict[str, Any]:
        """``GET /v1/model`` — the trained model's topology + metrics (409 if none trained)."""
        return self._parse_json(self._request("GET", ENDPOINT_MODEL))

    def get_dataset(self) -> dict[str, Any]:
        """``GET /v1/dataset`` — descriptor of the split the model was trained on (409 if none)."""
        return self._parse_json(self._request("GET", ENDPOINT_DATASET))

    # ─── Health / Readiness ────────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """``GET /v1/health`` — liveness."""
        return self._parse_json(self._request("GET", ENDPOINT_HEALTH))

    def is_ready(self) -> bool:
        """``GET /v1/health/ready`` — ``True`` iff the service reports ready."""
        try:
            payload = self._parse_json(self._request("GET", ENDPOINT_HEALTH_READY))
        except JuniperRecurrenceClientError:
            return False
        return payload.get("status") == HEALTH_READY_STATUS

    def wait_for_ready(self, timeout: float = DEFAULT_READY_TIMEOUT, poll_interval: float = DEFAULT_READY_POLL_INTERVAL) -> bool:
        """Poll ``/v1/health/ready`` until ready or ``timeout`` seconds elapse."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_ready():
                return True
            time.sleep(poll_interval)
        return self.is_ready()

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> "JuniperRecurrenceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
