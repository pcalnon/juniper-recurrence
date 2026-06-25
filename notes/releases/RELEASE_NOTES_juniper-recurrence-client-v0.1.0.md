# juniper-recurrence-client v0.1.0 Release Notes

**Release Date:** 2026-06-18
**Version:** 0.1.0
**Release Type:** MINOR (initial package release)

---

## Overview

The first release of **`juniper-recurrence-client`** — a lean `requests`-based HTTP client for the juniper-recurrence service, the 3rd distribution in the `pcalnon/juniper-recurrence` repo (alongside the Δt-native LMU model and the FastAPI/CLI app). It mirrors `juniper-data-client` / `juniper-cascor-client` so consumers — notably **juniper-canopy**'s recurrence `BackendProtocol` adapter — drive every Juniper backend the same way.

> **Status:** ALPHA — initial public surface; ready for canopy's backend adapter and ad-hoc use.

---

## Release Summary

- **Release type:** MINOR (initial package)
- **Primary focus:** First HTTP client for the recurrence app
- **Breaking changes:** N/A (new package)
- **Priority summary:** Provides the HTTP path juniper-canopy's recurrence backend (WS-5) needs (roadmap H7)

---

## What's New

### `JuniperRecurrenceClient`

Wraps the recurrence app's REST surface:

| Method | Endpoint |
|--------|----------|
| `train` / `training_status` | `POST /v1/train`, `GET /v1/training/status` |
| `predict` | `POST /v1/predict` |
| `crossval` / `crossval_status` | `POST /v1/crossval`, `GET /v1/crossval/status` |
| `get_model` / `get_dataset` | `GET /v1/model`, `GET /v1/dataset` |
| `health_check` / `is_ready` / `wait_for_ready` | `GET /v1/health[/ready]` |

- **Idempotent-only retry** (GET/HEAD) — the train/predict/crossval POSTs carry server-side state (train & crossval are lock-guarded), so they never auto-retry on a transient 5xx.
- **`X-API-Key` auth** with `_FILE` Docker-secret indirection (`JUNIPER_RECURRENCE_API_KEY` / `..._FILE`).
- **Typed exception hierarchy** including a `JuniperRecurrenceConflictError` (409) unique to the recurrence app (run-in-progress / no-trained-model).
- Optional `on_request` instrumentation hook + guarded `juniper-observability` `X-Request-ID` propagation (never required).

---

## Install

```bash
pip install juniper-recurrence-client
```

`requests`-only at the core; `pip install juniper-recurrence-client[observability]` adds the optional `juniper-observability` integration.

---

## Test Results

- **33 unit tests** (`responses`-mocked), **94% coverage** (`client.py` 93%); `ruff` clean; `python -m build` + `twine check` PASS.

---

## Known Issues

None known at time of release.

---

## What's Next

- juniper-canopy's recurrence `BackendProtocol` adapter (WS-5) will pin and consume this client when its recurrence backend goes live.

---

## Links

- Changelog: `juniper-recurrence-client/CHANGELOG.md`
- Introducing PR: <https://github.com/pcalnon/juniper-recurrence/pull/24>
- Roadmap (H7): [JUNIPER_RECURRENCE_STATE_ASSESSMENT_AND_ROADMAP_2026-06-17.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_STATE_ASSESSMENT_AND_ROADMAP_2026-06-17.md)

---

## Contributors

- Paul Calnon (@pcalnon)
