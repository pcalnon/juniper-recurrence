# Changelog

All notable changes to `juniper-recurrence` (the application package) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The model package (`juniper-recurrence-model`) maintains its own changelog under
`juniper-recurrence-model/CHANGELOG.md`.

## [Unreleased]

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

### Notes

- The publish workflow (`publish-recurrence-app.yml`) and expanded docs arrive in
  WS-4b PR-3.
