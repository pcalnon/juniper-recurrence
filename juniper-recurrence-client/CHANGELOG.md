# Changelog

All notable changes to the `juniper-recurrence-client` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with [PEP 440](https://peps.python.org/pep-0440/) pre-release identifiers.

## [Unreleased]

### Added

- **`readout` selection forwarded by `train()` / `crossval()` (DP-3 P2c).** Both methods gain
  `readout: Optional[Literal["linear", "rff"]]`, `rff_features: Optional[int]`, and
  `rff_gamma: Optional[Union[float, Literal["median"]]]`, forwarded verbatim in the request body so
  callers can select the service's nonlinear RFF readout (Rung 2a). Backward compatible — all optional;
  unset ⇒ an unchanged request body.

- **`ridge="gcv"` accepted by `train()` / `crossval()` (DP-3 P1).** The `ridge` parameter widens
  from `Optional[float]` to `Optional[Union[float, Literal["gcv"]]]`, so callers can request the
  service's closed-form GCV selection of the readout penalty. The value is forwarded verbatim in
  the request body (no client-side validation change).

## [0.1.0] - 2026-06-18

### Added

- **Initial `juniper-recurrence-client` package** — a lean `requests`-based HTTP client for the
  juniper-recurrence service, the 3rd distribution in the `pcalnon/juniper-recurrence` repo
  alongside `juniper-recurrence-model` and `juniper-recurrence`. Mirrors `juniper-data-client` /
  `juniper-cascor-client` so juniper-canopy's recurrence `BackendProtocol` adapter drives every
  Juniper backend the same way.
- **`JuniperRecurrenceClient`** wrapping the app's REST surface: `train` / `training_status`,
  `predict`, `crossval` / `crossval_status`, `get_model`, `get_dataset`, and
  `health_check` / `is_ready` / `wait_for_ready`. Idempotent-only retry policy (GET/HEAD only — the
  train/predict/crossval POSTs carry server-side state), a pooled `requests.Session`, `X-API-Key`
  auth with `_FILE` Docker-secret indirection, the optional `on_request` instrumentation hook, and
  best-effort `X-Request-ID` propagation via `juniper-observability` (guarded — never required).
- **Typed exception hierarchy** (`JuniperRecurrenceClientError` + connection / timeout /
  not-found / **conflict (409)** / validation / configuration leaves). The 409 path is unique to
  the recurrence app (lock-guarded train/crossval; "no trained model yet").
- **33 unit tests** (`responses`-mocked) covering URL normalization, every client method, auth
  resolution (explicit / env / `_FILE` precedence + empty-file fallback), and the full
  error-mapping matrix (404/409/422/500/connection/malformed-JSON).
