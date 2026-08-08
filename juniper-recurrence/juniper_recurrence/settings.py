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
import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from juniper_service_core import SettingsBase, get_secret
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import NoDecode, PydanticBaseSettingsSource, SettingsConfigDict

__all__ = ["Settings"]


# ---------------------------------------------------------------------------------------------------
# Experiment YAML config layer (Wave 3.3 -- CLI experimentation plan SS5.1/SS5.2/SS5.6, juniper-ml
# notes/JUNIPER_2026-07-29_JUNIPER-ECOSYSTEM_CASCOR-RECURRENCE-CLI-TEST-VALIDATION-EXPERIMENTATION-PLAN.md)
# ---------------------------------------------------------------------------------------------------

_EXPERIMENT_CONFIG_ENV_VAR: str = "JUNIPER_RECURRENCE_CONFIG_FILE"
_EXPERIMENT_SCHEMA_VERSION_MAX: int = 1
# The SS5.4/SS5.5 top-level surface. Only ``service:`` is projected into Settings; the
# other blocks belong to the driver / launcher layers (plan SS6) and are ignored here.
_EXPERIMENT_TOP_LEVEL_BLOCKS: frozenset = frozenset({"schema_version", "experiment", "service", "dataset", "training", "train", "crossval", "predict", "runtime", "outputs"})
# SS5.6 rule 6 (SS5.5 comment): infrastructure is launcher-owned (CLI flags / process env)
# and rejected outright in experiment YAML.
_EXPERIMENT_SERVICE_FORBIDDEN_KEYS: frozenset = frozenset({"host", "port", "juniper_data_url"})

_experiment_config_logger = logging.getLogger("juniper_recurrence.settings")


class ExperimentConfigError(ValueError):
    """Invalid experiment YAML for the ``service:`` projection -- fail loud BEFORE boot (SS5.6)."""


class ExperimentYamlSettingsSource(PydanticBaseSettingsSource):
    """Project ONLY the experiment YAML's ``service:`` block into Settings values (SS5.2).

    The stock ``YamlConfigSettingsSource`` reads the file's TOP-LEVEL mapping as field
    values, and an experiment YAML's top level is ``schema_version`` / ``experiment`` /
    ``service`` / ``dataset`` / ``train`` / ... -- none of which is a Settings field.
    With this model's ``extra="ignore"`` every key would be dropped silently: the stock
    source would no-op the whole layer. This source parses the file once, validates
    fail-loud (unknown top-level blocks, ``schema_version``, unknown or launcher-owned
    ``service:`` keys -- SS5.6 rules 1/2/6), and yields the ``service:`` keys so YAML
    sits ABOVE env in the SS5.1 precedence (a run stays reproducible from its YAML even
    in a shell with stale exported ``JUNIPER_RECURRENCE_*`` vars).
    """

    def __init__(self, settings_cls: type, yaml_file: str) -> None:
        super().__init__(settings_cls)
        self._yaml_file = yaml_file
        self._service_block = self._load_and_validate()

    def _load_and_validate(self) -> dict[str, Any]:
        try:
            raw = Path(self._yaml_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ExperimentConfigError(f"{_EXPERIMENT_CONFIG_ENV_VAR}={self._yaml_file!r} is unreadable: {exc}") from exc
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ExperimentConfigError(f"experiment config {self._yaml_file!r} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ExperimentConfigError(f"experiment config {self._yaml_file!r} must be a YAML mapping, got {type(data).__name__}")

        unknown_blocks = sorted(set(data) - _EXPERIMENT_TOP_LEVEL_BLOCKS)
        if unknown_blocks:
            raise ExperimentConfigError(f"experiment config {self._yaml_file!r}: unknown top-level block(s) {', '.join(unknown_blocks)} (allowed: {', '.join(sorted(_EXPERIMENT_TOP_LEVEL_BLOCKS))})")

        version = data.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool) or not 1 <= version <= _EXPERIMENT_SCHEMA_VERSION_MAX:
            raise ExperimentConfigError(f"experiment config {self._yaml_file!r}: schema_version must be an integer in 1..{_EXPERIMENT_SCHEMA_VERSION_MAX}, got {version!r}")

        service = data.get("service") or {}
        if not isinstance(service, dict):
            raise ExperimentConfigError(f"experiment config {self._yaml_file!r}: service must be a mapping, got {type(service).__name__}")

        forbidden = sorted(set(service) & _EXPERIMENT_SERVICE_FORBIDDEN_KEYS)
        if forbidden:
            raise ExperimentConfigError(f"experiment config {self._yaml_file!r}: service key(s) {', '.join(forbidden)} rejected (SS5.6 rule 6) -- host/port/juniper_data_url are launcher-owned (CLI flags / process env), never experiment YAML")

        known_fields = set(self.settings_cls.model_fields)
        unknown = sorted(key for key in service if key not in known_fields)
        if unknown:
            raise ExperimentConfigError(f"experiment config {self._yaml_file!r}: unknown service key(s) {', '.join(unknown)} -- Settings is extra='ignore', so an unvalidated key would be dropped silently; fix the YAML (fields include e.g. log_level, log_format, metrics_enabled, rate_limit_enabled, default_d, default_theta, default_ridge)")

        if service:
            _experiment_config_logger.info("experiment config %s: projecting service keys %s (YAML > env, SS5.1)", self._yaml_file, sorted(service))
        return dict(service)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        if field_name in self._service_block:
            return self._service_block[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._service_block)


class Settings(SettingsBase):
    """Runtime configuration for the juniper-recurrence app (env prefix ``JUNIPER_RECURRENCE_``)."""

    model_config = SettingsConfigDict(env_prefix="JUNIPER_RECURRENCE_", extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        """CLI/init > YAML ``service:`` block > env > defaults (SS5.1; Wave 3.3).

        The YAML source is inserted only when ``JUNIPER_RECURRENCE_CONFIG_FILE`` is set,
        so the layer is inert for every existing env/compose deployment (plan risk R-4).
        (No ``.env`` tier here by design -- see the module docstring's env-file-leak note.)
        """
        yaml_path = os.environ.get(_EXPERIMENT_CONFIG_ENV_VAR)
        sources: list[Any] = [init_settings]
        if yaml_path:
            sources.append(ExperimentYamlSettingsSource(settings_cls, yaml_file=yaml_path))
        sources += [env_settings, dotenv_settings, file_secret_settings]
        return tuple(sources)

    # --- service identity / bind (override SettingsBase defaults) ---------------------
    service_name: str = "juniper-recurrence"
    # Container default binds all interfaces; for a local ``serve`` set
    # ``JUNIPER_RECURRENCE_HOST=127.0.0.1`` (design §6.8). The bind-all is
    # intentional for the containerised service, so the bandit pre-commit hook's
    # B104 (hardcoded_bind_all_interfaces) finding is suppressed inline here.
    host: str = "0.0.0.0"  # nosec B104 — intentional container bind-all (design §6.8)
    port: int = 8210  # container port; deploy maps host 8211 -> ctr 8210 (design §6.8)

    # --- logging ----------------------------------------------------------------------
    # ``SettingsBase`` supplies ``log_level`` (default "INFO"); ``log_format`` selects the
    # output style consumed by ``juniper_observability.configure_logging`` — "json" for
    # structured-JSON (log shippers) or "text" for human-readable. "text" is the ecosystem
    # default (matches juniper-data / juniper-cascor / juniper-canopy).
    log_format: str = "text"

    # --- API-key auth + rate limiting -------------------------------------------------
    api_keys: Annotated[list[str] | None, NoDecode] = Field(default=None)
    # SEC-F01: the INTENDED auth posture, fed to enforce_auth_posture at boot.
    # False (default) = an unset/blank JUNIPER_RECURRENCE_API_KEYS only WARNs
    # (service runs open — bare/dev profile); True = boot REFUSES (CRITICAL +
    # AuthPostureError) when no real key is configured. Set
    # JUNIPER_RECURRENCE_REQUIRE_AUTH=true wherever secrets are provisioned
    # (the composed juniper-deploy stack).
    require_auth: bool = False
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
