"""Protocol-level constants for the juniper-recurrence REST client.

Centralizes the literals used by ``client.py`` — base URL, endpoint paths, header names,
HTTP/retry configuration — mirroring juniper-data-client's constants module so the wire
contract is discoverable in one place.
"""

from typing import List, Tuple

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "DEFAULT_BACKOFF_FACTOR",
    "RETRYABLE_STATUS_CODES",
    "RETRY_ALLOWED_METHODS",
    "HTTP_POOL_CONNECTIONS",
    "HTTP_POOL_MAXSIZE",
    "URL_SCHEME_PREFIXES",
    "DEFAULT_URL_SCHEME_PREFIX",
    "API_VERSION_PATH_SUFFIX",
    "DEFAULT_READY_TIMEOUT",
    "DEFAULT_READY_POLL_INTERVAL",
    "HEALTH_READY_STATUS",
    "API_KEY_HEADER_NAME",
    "API_KEY_ENV_VAR",
    "API_KEY_FILE_ENV_VAR",
    "ENDPOINT_HEALTH",
    "ENDPOINT_HEALTH_READY",
    "ENDPOINT_TRAIN",
    "ENDPOINT_TRAINING_STATUS",
    "ENDPOINT_PREDICT",
    "ENDPOINT_MODEL",
    "ENDPOINT_DATASET",
    "ENDPOINT_CROSSVAL",
    "ENDPOINT_CROSSVAL_STATUS",
]

# ─── Service Configuration ───────────────────────────────────────────────────

# The juniper-recurrence app binds container port 8210; juniper-deploy maps host 8211 -> 8210,
# so the default host-facing base URL is 8211 (mirrors the deploy port map).
DEFAULT_BASE_URL: str = "http://localhost:8211"

# ─── HTTP Configuration ──────────────────────────────────────────────────────

DEFAULT_TIMEOUT: int = 30
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 0.5
RETRYABLE_STATUS_CODES: List[int] = [429, 500, 502, 503, 504]
# Auto-retry is restricted to idempotent methods (RFC 9110 §9.2.2). The recurrence POSTs
# (train / predict / crossval) carry server-side state — train and crossval are lock-guarded —
# so a transient-5xx retry must not silently re-issue them. Only GET/HEAD auto-retry.
RETRY_ALLOWED_METHODS: List[str] = ["HEAD", "GET"]
HTTP_POOL_CONNECTIONS: int = 10
HTTP_POOL_MAXSIZE: int = 10

# ─── URL Normalization ───────────────────────────────────────────────────────

URL_SCHEME_PREFIXES: Tuple[str, ...] = ("http://", "https://")
DEFAULT_URL_SCHEME_PREFIX: str = "http://"
API_VERSION_PATH_SUFFIX: str = "/v1"

# ─── Readiness Polling ───────────────────────────────────────────────────────

DEFAULT_READY_TIMEOUT: float = 30.0
DEFAULT_READY_POLL_INTERVAL: float = 0.5
HEALTH_READY_STATUS: str = "ready"

# ─── Authentication ──────────────────────────────────────────────────────────

# The recurrence app enforces the X-API-Key header, reading its accepted keys from
# JUNIPER_RECURRENCE_API_KEYS (plural; CSV or JSON array, with _FILE indirection). The client
# sends a single key, resolved from the singular JUNIPER_RECURRENCE_API_KEY (and its _FILE
# Docker-secret form) — document the singular/plural asymmetry in AGENTS.md.
API_KEY_HEADER_NAME: str = "X-API-Key"
API_KEY_ENV_VAR: str = "JUNIPER_RECURRENCE_API_KEY"
API_KEY_FILE_ENV_VAR: str = f"{API_KEY_ENV_VAR}_FILE"

# ─── REST Endpoints (the juniper-recurrence app surface) ─────────────────────

ENDPOINT_HEALTH: str = "/v1/health"
ENDPOINT_HEALTH_READY: str = "/v1/health/ready"
ENDPOINT_TRAIN: str = "/v1/train"
ENDPOINT_TRAINING_STATUS: str = "/v1/training/status"
ENDPOINT_PREDICT: str = "/v1/predict"
ENDPOINT_MODEL: str = "/v1/model"
ENDPOINT_DATASET: str = "/v1/dataset"
ENDPOINT_CROSSVAL: str = "/v1/crossval"
ENDPOINT_CROSSVAL_STATUS: str = "/v1/crossval/status"

# ─── HTTP Status Codes ───────────────────────────────────────────────────────

HTTP_400_BAD_REQUEST: int = 400
HTTP_404_NOT_FOUND: int = 404
HTTP_409_CONFLICT: int = 409
HTTP_422_UNPROCESSABLE_ENTITY: int = 422
