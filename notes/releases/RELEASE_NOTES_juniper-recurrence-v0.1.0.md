# juniper-recurrence v0.1.0 Release Notes

**Release Date:** 2026-06-17
**Version:** 0.1.0
**Codename:** WS-4b — Application Layer
**Release Type:** MINOR (initial release of the application distribution)

---

## Overview

First release of `juniper-recurrence`, the FastAPI + CLI application that serves the Δt-native LMU regressor (`juniper-recurrence-model`) over a regression-generic REST surface. It is built on the shared `juniper-service-core` framework, trains synchronously in-request via a closed-form `lstsq` readout, and loads 3-D windowed sequences (`equities_seq`) through `juniper-data-client`.

> **Status:** ALPHA — initial research release. Single-worker, in-process state; synchronous training; numpy-only model.

---

## Release Summary

- **Release type:** MINOR (initial 0.1.0 of a new package)
- **Primary focus:** NEW APPLICATION — the WS-4b service/CLI layer over the LMU model
- **Breaking changes:** NO (new package)
- **Priority summary:** WS-4b delivered (PR-1 skeleton + PR-2 routes + PR-3 publish/docs)

---

## Features Summary

| ID         | Feature                                               | Status | Version |
| ---------- | ----------------------------------------------------- | ------ | ------- |
| WS-4b PR-1 | App skeleton (settings, `build_app`, serve CLI, CI)   | Done   | 0.1.0   |
| WS-4b PR-2 | Routes + data path + synchronous training + train CLI | Done   | 0.1.0   |
| WS-4b PR-3 | Publish workflow + docs                               | Done   | 0.1.0   |

---

## What's New

### Application service (on `juniper-service-core`)

- `build_app()` wires `create_app(routers=…)` plus the canonical `RequestBodyLimit → SecurityHeaders → Security` middleware stack; module-level `app` for uvicorn.
- Settings: env prefix `JUNIPER_RECURRENCE_`, port 8210, Docker `_FILE` secret indirection, no `.env` leak.

### CLI (dual-mode)

- `juniper-recurrence serve` → uvicorn; `juniper-recurrence train` → headless fit + persist via `LMUSerializer`.

### Data path

- `data.py`: `juniper-data-client` → 3-D NPZ (`equities_seq`) → model kwargs (`X`, `y`, `dt`, `target_dt`, `readout_mask`, `seq_lengths`).

---

## API Changes — New Endpoints

| Method | Endpoint                         | Description                                                   |
| ------ | -------------------------------- | ------------------------------------------------------------- |
| `GET`  | `/v1/health`, `/v1/health/ready` | Liveness / readiness (auth-exempt; from service-core)         |
| `POST` | `/v1/train`                      | Synchronous training; returns `TrainResult`                   |
| `GET`  | `/v1/training/status`            | Last status + ordered training events                         |
| `POST` | `/v1/predict`                    | Continuous ŷ (inline arrays or dataset ref); 409 before train |
| `GET`  | `/v1/model`                      | Topology + metrics; 409 if no model                           |
| `GET`  | `/v1/dataset`                    | Dataset descriptor                                            |

All routes are regression-generic (RK-6: no `argmax`, no `accuracy`).

---

## Test Results

- Full app CI suite green across **Python 3.12 / 3.13 / 3.14** (verified in PR #9 and on `main` post-merge); coverage gate ≥90%.
- Model dependency `juniper-recurrence-model 0.1.0`: 53/53 tests + 10/10 model-core conformance.

---

## Upgrade / Install

```bash
pip install juniper-recurrence
juniper-recurrence serve                 # FastAPI service on :8210
juniper-recurrence train --dataset …     # headless fit + persist
```

This is a new package — no migration required.

---

## Known Issues

- `juniper-data-client 0.4.1` does not export `validate_npz_contract` (added upstream after 0.4.1); `data.py` guards the import and falls back to the model's `sequence_data_from_arrays` checks. The pin will be tightened once the validator publishes.
- Single-worker, in-process state — no persistence / horizontal scale-out in v1.
- `/v1/metrics` (Prometheus), async/WebSocket, and distributed training are deferred (WS-8).
- Not yet in the `juniper-ml[servers]` extra — follows in WS-7, after the app is on PyPI.

---

## What's Next

- **WS-7:** add `juniper-recurrence` to `juniper-ml[servers]`/`[all]`.
- **model-core 0.2.0 `crossval`** layer + a `/v1/crossval` endpoint.
- `/v1/metrics` observability; deferred model increments (`target_dt` memory-advance, trained/nonlinear readout).
- See `notes/JUNIPER_RECURRENCE_STATE_AND_ROADMAP_2026-06-17.md`.

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date       | Description                               |
| ------- | ---------- | ----------------------------------------- |
| 0.1.0   | 2026-06-17 | Initial release — WS-4b application layer |
