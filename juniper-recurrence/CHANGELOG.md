# Changelog

All notable changes to `juniper-recurrence` (the application package) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The model package (`juniper-recurrence-model`) maintains its own changelog under
`juniper-recurrence-model/CHANGELOG.md`.

## [Unreleased]

### Added

- **Application logging is now configured at startup** (audit H1). The `serve` entrypoint calls a new
  `juniper_recurrence.logging_config.init_logging`, which configures the root logger from
  `settings.log_level` / the new `settings.log_format` — preferring the shared
  `juniper_observability.configure_logging` (structured-JSON with `request_id` correlation) when the
  `[observability]` extra is installed, and falling back to stdlib `logging` when it is not. The
  train / predict / crossval routers now emit operational log lines (run start/complete with dataset,
  duration, and metrics; upstream-data failures; `409` lock contention; `503` torch-readout gaps).
- **`log_format` setting** (`JUNIPER_RECURRENCE_LOG_FORMAT`, default `"text"`; `"json"` for structured
  logs) selecting the logging output style.

## [0.2.0] - 2026-06-24

### Fixed

- **CLI `--rff-features` / `--rff-gamma` are now rejected without `--readout rff`** (P2c follow-up).
  The rule was enforced at the HTTP edge (422) but the `train` CLI silently dropped the RFF-only knobs
  on the linear readout. The check now lives in the shared `build_lmu_regressor`, so the CLI and HTTP
  behave identically (the CLI exits 2 with an error message).

### Added

- **HTTP readout enum — select the DP-3 readout over `/v1/train` and `/v1/crossval` (P2c).** The
  train + crossval request bodies (and the `train` CLI) accept `readout: "linear" | "rff"` plus the
  RFF params `rff_features` / `rff_gamma`; the service constructs the matching immutable readout spec
  (`LinearReadoutSpec` / `RFFReadoutSpec`) and passes it to `LMURegressor`. `readout` defaults to the
  back-compat linear readout (which uses `ridge`); `"rff"` selects the numpy nonlinear
  random-Fourier-feature readout (Rung 2a). `rff_features` / `rff_gamma` are rejected unless
  `readout="rff"`. Floor-bumps the model pin to `juniper-recurrence-model>=0.1.4` (the release that
  ships `RFFReadoutSpec`). See the DP-3 design §6 / §8a.

- **HTTP readout enum — `readout="mlp"`, the torch MLP readout (DP-3 P3).** Extends the `readout`
  enum on `POST /v1/train`, `POST /v1/crossval`, and the `train` CLI with `"mlp"` (Rung 2b), plus the
  optional MLP hyperparameters `mlp_hidden` / `mlp_weight_decay` / `mlp_lr` / `mlp_max_epochs` /
  `mlp_patience` (each defaults to the `MLPReadoutSpec` value; rejected unless `readout="mlp"`, and
  `ridge` is rejected with `"mlp"` since the MLP regularises via weight decay). Floor-bumps the model
  pin to `juniper-recurrence-model>=0.1.5` (the release that ships `MLPReadoutSpec`). The MLP needs
  torch **at runtime**, kept optional via a new **`[torch]` extra**
  (`pip install 'juniper-recurrence[torch]'`); a deployment without it still starts and rejects
  `readout="mlp"` with a clear **503** (the spec import itself is torch-free). See the DP-3 design §6
  and the evaluation findings §3.4.

- **Container image — `Dockerfile` + `.dockerignore` (OUT-4 / WS-7).** A multi-stage,
  slim (~77 MB) image. The LMU stack is numpy-only (no torch), so the build installs the
  app + the `[observability]` extra from PyPI with no CPU-torch lock dance. Runs as a
  non-root `juniper` user; `ENTRYPOINT ["juniper-recurrence"]` / `CMD ["serve"]` launches
  uvicorn on container port 8210 (the deploy compose maps host 8211 → container 8210). An
  HTTP `HEALTHCHECK` probes `/v1/health` with a 40s `start-period` (the pure-Python stack
  imports for ~10-15s before uvicorn binds). A new `Docker Build & Smoke Test` CI job in
  `ci-recurrence-app.yml` builds the image and asserts `/v1/health` → 200. This makes the
  published app deployable (the WS-7 compose integration — host 8211 — follows next).
- **`ridge="gcv"` at the API + CLI edge (DP-3 P1).** `POST /v1/train` / `POST /v1/crossval`
  (and `juniper-recurrence train --ridge gcv`) now accept `ridge="gcv"` in addition to a
  non-negative float — requesting the model's closed-form generalised-cross-validation selection
  of the readout L2 penalty. `default_ridge` widens to `float | Literal["gcv"]` (default `0.0`,
  unchanged); floats still validate `>= 0`.

### Changed

- **`juniper-recurrence-model` pin floor → `>=0.1.5`** (was `>=0.1.2`; ceiling unchanged at
  `<0.2.0`). 0.1.5 ships the full DP-3 readout spectrum — `ridge="gcv"`, the numpy RFF readout, and
  the torch MLP readout (the `[torch]` extra) — which the widened API selects at runtime.
  Publish-first: 0.1.5 must reach PyPI before this app change installs.

- **`[bench]` extra now pins `juniper-data>=0.9.0`** (was `>=0.6.0`). juniper-data 0.7.0
  publishes the synthetic Δt generators (#187/#188) + scaling-meta channel (#189), 0.8.0 adds the
  equities `regression_target` option (#195), and 0.9.0 ships the `delay_product` capacity
  generator (#203) the DP-3 nonlinear-readout benchmark requires — so the benchmark harness installs
  cleanly from PyPI (the editable-sibling / cross-repo-clone workaround is no longer needed).
  `>=0.9.0` is required (not just `>=0.7.0`): `EquitiesParams` ignores unknown kwargs (pydantic
  `extra='ignore'`), so an older pin would silently drop the new target and re-measure the raw
  non-stationary close.
- **Benchmark `equities_seq` re-bench — stationary target + regularized-readout LMU.** The
  real-data row now uses a stationary next-day **log-return** target and adds
  `lmu_var/fixed_d16_ridge1.0` rows for a fair comparison (the ratified ridge=0 synthetic primary
  bands are unchanged). Finding: the published r²≈−50 equities failure was dominantly an
  unregularized-readout artifact (ridge=0 — a model-core conformance default), not target
  non-stationarity; with a regularized readout on the stationary target the Δt-LMU reaches the
  efficient-market predictability ceiling (r²≈0) and beats linear ridge. See
  juniper-recurrence#28 and the juniper-ml findings doc §3.2.

## [0.1.1] - 2026-06-17

### Added

- **Cross-validation endpoint (`POST /v1/crossval` + `GET /v1/crossval/status`).** The indirect
  evaluation route: a dataset selection → synchronous walk-forward cross-validation over the
  `_full` split via `juniper_model_core.crossval.cross_validate` (a fresh `LMURegressor` per fold,
  held-out scoring, per-metric mean/std aggregate) → an aggregated result. Accepts
  `n_folds` / `scheme` (`expanding`|`rolling`) / `embargo` / `min_train` plus the LMU
  hyperparameters; the most recent result is persisted in-process and returned by
  `GET /v1/crossval/status`. Regression-generic (RK-6 — no `accuracy`, no `argmax`).
- **Prometheus `/metrics` endpoint (IP-allowlist gated).** When `metrics_enabled` is set, mounts a
  Prometheus exposition endpoint at `/metrics` via `juniper-observability` (the `[observability]`
  extra): `PrometheusMiddleware` records HTTP-request + build-info metrics, and `MetricsAuthMiddleware`
  IP-gates the path against `metrics_trusted_ips`. `/metrics` is exempt from the API-key
  `SecurityMiddleware` (service-core `EXEMPT_PATHS`, SEC-16) and IP-gated instead. Guarded — the app
  still runs (with a warning) when `juniper-observability` is absent.

### Changed

- **Δt contract validation is now mandatory.** Bumped the `juniper-data-client` pin to `>=0.4.2,<0.5.0` (the release that publishes `validate_npz_contract`) and removed the optional-import guard in `juniper_recurrence/data.py`: the full NPZ contract gate now always runs on a downloaded artifact, instead of silently falling back to model-side shape checks when the installed client lacked the validator. Closes roadmap I1 / D-2.
- **Adopt the juniper-model-core 0.2.0 cross-validation layer.** Bumped `juniper-model-core` to `[crossval]>=0.2.0,<0.3.0` (the app now imports `juniper_model_core.crossval` at runtime) and `juniper-recurrence-model` to `>=0.1.2,<0.2.0` (the crossval-capable model release that admits model-core 0.2.0).

## [0.1.0] - 2026-06-17

### Added

- **App skeleton (WS-4b PR-1).** New `juniper-recurrence` application distribution:
  the import package `juniper_recurrence/` with `Settings` (env prefix
  `JUNIPER_RECURRENCE_`, port 8210, Docker `_FILE` secret indirection, no
  `env_file=`), the `build_app()` factory wiring `juniper-service-core`'s
  `create_app` plus the canonical `RequestBodyLimit` → `SecurityHeaders` →
  `Security` middleware stack, the module-level `app` for uvicorn, and the
  `juniper-recurrence serve` CLI subcommand.
- CI workflow `ci-recurrence-app.yml` (lint + test matrix 3.12/3.13/3.14 + build).
- **Routes + data path + training (WS-4b PR-2).** The REST surface over the LMU:
  `POST /v1/train` (synchronous — `TrainingLifecycle(LMURegressor(...)).run(...)` inline,
  returns the `TrainResult`), `GET /v1/training/status`, `POST /v1/predict` (inline
  arrays or a dataset ref; continuous `ŷ`), `GET /v1/model` (topology + metrics), and
  `GET /v1/dataset` (descriptor). Backed by `state.py` (in-process model/result/event
  holder behind a lock), `events.py` (ring-buffer event sink), `data.py` (juniper-data-client
  → 3-D NPZ → model kwargs), and `schemas.py`. Adds the headless `juniper-recurrence train`
  CLI subcommand. All routes are regression-generic (RK-6: no `argmax`, no `accuracy`).
- **Publish workflow + docs (WS-4b PR-3).** `publish-recurrence-app.yml` publishes the
  app to PyPI on a `juniper-recurrence-v*` tag — TestPyPI first (hardened `--no-deps`,
  no-fallback install verification), then PyPI, via OIDC trusted publishing. README
  documents the full endpoint surface, the headless `train` CLI, and the publish flow.

### Notes

- The app pins published-PyPI upstreams directly (`juniper-service-core`,
  `juniper-model-core`, `juniper-recurrence-model`, `juniper-data-client`); no
  editable-sibling installs are required.
- The published `juniper-data-client 0.4.1` does not yet export `validate_npz_contract`
  (added upstream after 0.4.1); `data.py` guards the import and falls back to the model's
  `sequence_data_from_arrays` contract checks until the validator publishes.
- A `juniper-recurrence` entry in the `juniper-ml [servers]` extra follows in WS-7,
  after the app is on PyPI.
