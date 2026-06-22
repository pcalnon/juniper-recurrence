"""Settings for the juniper-recurrence service.

Subclasses :class:`juniper_service_core.SettingsBase` (which supplies
``service_name`` / ``host`` / ``port`` / ``log_level``) and reads the
``JUNIPER_RECURRENCE_`` environment namespace.

Three hardening choices, each a recorded ecosystem incident (plan §7 / §15):

* **No ``env_file=``** — setting it is the pydantic-settings ``.env``-leak class
  (cascor #309 / canopy #325 / data #153). Isolation relies on ``env_prefix`` +
  ``extra="ignore"`` only.
* **Docker ``_FILE`` secret indirection** — ``api_keys`` and the outbound
  ``juniper_data_api_key`` resolve through :func:`juniper_service_core.get_secret`,
  which prefers ``<VAR>_FILE`` (a mounted path) over ``<VAR>`` (worker-secret
  incident precedent).
* **``api_keys`` accepts CSV or JSON-array** — :data:`NoDecode` keeps
  pydantic-settings from JSON-decoding the env value, so a plain secret-file
  payload (``"k1,k2"``) never raises the JSON-list ``ValidationError`` (cascor
  ``_parse_api_keys`` precedent / secrets.example incident).
"""

from __future__ import annotations

import ipaddress
import json
from typing import Annotated, Any, Literal

from juniper_service_core import SettingsBase, get_secret
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import NoDecode, SettingsConfigDict

__all__ = ["Settings"]


class Settings(SettingsBase):
    """Runtime configuration for the juniper-recurrence app (env prefix ``JUNIPER_RECURRENCE_``)."""

    model_config = SettingsConfigDict(env_prefix="JUNIPER_RECURRENCE_", extra="ignore")

    # --- service identity / bind (override SettingsBase defaults) ---------------------
    service_name: str = "juniper-recurrence"
    # Container default binds all interfaces; for a local ``serve`` set
    # ``JUNIPER_RECURRENCE_HOST=127.0.0.1`` (design §6.8). The bind-all is
    # intentional for the containerised service, so the bandit pre-commit hook's
    # B104 (hardcoded_bind_all_interfaces) finding is suppressed inline here.
    host: str = "0.0.0.0"  # nosec B104 — intentional container bind-all (design §6.8)
    port: int = 8210  # container port; deploy maps host 8211 -> ctr 8210 (design §6.8)

    # --- API-key auth + rate limiting -------------------------------------------------
    api_keys: Annotated[list[str] | None, NoDecode] = Field(default=None)
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60

    # --- upstream juniper-data (outbound, consumed by the PR-2 data path) -------------
    juniper_data_url: str = Field(
        default="http://localhost:8100",
        validation_alias=AliasChoices("juniper_data_url", "JUNIPER_DATA_URL", "JUNIPER_RECURRENCE_JUNIPER_DATA_URL"),
    )
    juniper_data_api_key: str | None = Field(default=None)

    # --- LMU hyperparameter defaults (consumed by the PR-2 training path) -------------
    default_d: int = 16
    default_theta: float | None = None
    default_ridge: float | Literal["gcv"] = 0.0

    # --- observability: Prometheus /metrics (IP-allowlist gated) ----------------------
    metrics_enabled: bool = True
    # Loopback-only by default (mirrors juniper-data); Docker / Compose deployments
    # extend this with the compose-network CIDR via JUNIPER_RECURRENCE_METRICS_TRUSTED_IPS,
    # e.g. '["127.0.0.1","::1","172.18.0.0/16"]'. MetricsAuthMiddleware does the gating.
    metrics_trusted_ips: list[str] = Field(default_factory=lambda: ["127.0.0.1", "::1"])

    # --- secret resolution (honor Docker ``_FILE`` indirection) -----------------------
    @model_validator(mode="before")
    @classmethod
    def _load_secrets_from_files(cls, data: Any) -> Any:
        """Populate ``api_keys`` / ``juniper_data_api_key`` from ``*_FILE`` secrets.

        ``get_secret`` checks ``<VAR>_FILE`` (a mounted path) before ``<VAR>`` so
        Docker / Compose secrets resolve without code change. The outbound
        juniper-data key reads the shared, unprefixed ``JUNIPER_DATA_API_KEY``
        (and ``JUNIPER_DATA_API_KEY_FILE``) — the cross-service convention used by
        cascor / canopy — falling back to the ``JUNIPER_RECURRENCE_``-prefixed form
        via the field's own env binding when set.
        """
        if isinstance(data, dict):
            if not data.get("api_keys"):
                secret = get_secret("JUNIPER_RECURRENCE_API_KEYS")
                if secret:
                    data["api_keys"] = secret
            if not data.get("juniper_data_api_key"):
                secret = get_secret("JUNIPER_DATA_API_KEY")
                if secret:
                    data["juniper_data_api_key"] = secret
        return data

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, value: Any) -> list[str] | None:
        """Normalise ``api_keys`` to ``list[str] | None`` from CSV, JSON-array, or list.

        Accepts a plain secret-file string (``"k1,k2"`` or ``'["k1","k2"]'``) without
        the pydantic-settings JSON-list ``ValidationError``. Empty / whitespace-only
        input collapses to ``None`` (auth disabled / open access).
        """
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return cleaned or None
        return value

    @field_validator("metrics_trusted_ips")
    @classmethod
    def _validate_metrics_trusted_ips(cls, value: list[str]) -> list[str]:
        """Reject unparseable IP / CIDR allowlist entries at construction.

        Mirrors juniper-data: a typo like ``172.18.0.0/164`` fails loudly here rather
        than silently never-matching at request time. ``MetricsAuthMiddleware`` applies
        the same parsing, so this is an early, friendlier echo of that check.
        """
        for entry in value:
            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError as exc:
                raise ValueError(f"invalid metrics_trusted_ips entry {entry!r}: {exc}") from exc
        return value

    def resolve_api_keys(self) -> list[str]:
        """The configured API keys as a plain list (empty ⇒ auth disabled / open access)."""
        return list(self.api_keys or [])
