# Changelog

All notable changes to `juniper-recurrence` (the application package) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The model package (`juniper-recurrence-model`) maintains its own changelog under
`juniper-recurrence-model/CHANGELOG.md`.

## [Unreleased]

### Changed

- **Δt contract validation is now mandatory.** Bumped the `juniper-data-client` pin to `>=0.4.2,<0.5.0` (the release that publishes `validate_npz_contract`) and removed the optional-import guard in `juniper_recurrence/data.py`: the full NPZ contract gate now always runs on a downloaded artifact, instead of silently falling back to model-side shape checks when the installed client lacked the validator. Closes roadmap I1 / D-2.

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
