# juniper-recurrence v0.1.1 Release Notes

**Release Date:** 2026-06-17
**Version:** 0.1.1
**Release Type:** MINOR (additive — new endpoints; backward-compatible)

---

## Overview

The first capability release of the **juniper-recurrence** FastAPI/CLI application after the initial 0.1.0 skeleton+routes. It adds the **indirect evaluation route** — a synchronous walk-forward **cross-validation** endpoint that drives the `LMURegressor` through the new `juniper-model-core` 0.2.0 crossval layer over irregular-Δt data — plus a **Prometheus `/metrics`** endpoint, and makes the Δt data-contract gate **mandatory**. Backward-compatible with 0.1.0.

> **Status:** STABLE — additive endpoints + a hardened data gate; no breaking changes.

---

## Release Summary

- **Release type:** MINOR (additive)
- **Primary focus:** Cross-validation evaluation endpoint; observability; mandatory Δt validation
- **Breaking changes:** NO
- **Priority summary:** Wave 2 — the recurrence-side consumption of the model-core 0.2.0 crossval layer (roadmap C2 / I2)

---

## What's New

### Cross-validation endpoint — `POST /v1/crossval` + `GET /v1/crossval/status`

The indirect evaluation route: a dataset selection → **synchronous walk-forward cross-validation** over the `_full` split via `juniper_model_core.crossval.cross_validate` (a fresh `LMURegressor` per fold, held-out scoring, per-metric mean/std aggregate) → an aggregated result. Accepts `n_folds` / `scheme` (`expanding`|`rolling`) / `embargo` / `min_train` plus the LMU hyperparameters; the most recent result is persisted in-process and returned by `GET /v1/crossval/status`. Regression-generic (RK-6 — never `accuracy`, never `argmax`).

### Prometheus metrics — `GET /metrics`

A Prometheus exposition endpoint, gated by an IP allowlist (the ecosystem's metrics-auth pattern), for scraping service/runtime metrics.

## Changed

- **Δt contract validation is now mandatory.** The `juniper-data-client` pin is `>=0.4.2,<0.5.0` (the release that publishes `validate_npz_contract`) and the optional-import guard in `data.py` is removed — the full NPZ contract gate now always runs on a downloaded artifact instead of silently degrading to model-side shape checks. Closes roadmap I1 / D-2.
- **Adopt the `juniper-model-core` 0.2.0 cross-validation layer.** Pins `juniper-model-core[crossval]>=0.2.0,<0.3.0` (the app now imports `juniper_model_core.crossval` at runtime) and `juniper-recurrence-model>=0.1.2,<0.2.0` (the crossval-capable model release that admits model-core 0.2.0).

---

## What's Changed (commits)

- `5989778` feat(app): Prometheus `/metrics` endpoint (IP-allowlist gated)
- `d194b71` feat(recurrence-app): POST `/v1/crossval` endpoint (model-core 0.2.0 crossval)
- `eb36d32` feat(app): make Δt contract validation mandatory (pin juniper-data-client>=0.4.2)

---

## Upgrade Notes

Backward-compatible; no migration. All upstream pins resolve against published PyPI versions: `juniper-model-core[crossval]>=0.2.0`, `juniper-recurrence-model>=0.1.2`, `juniper-data-client>=0.4.2`, `juniper-service-core>=0.1.0`.

```bash
pip install --upgrade juniper-recurrence
```

---

## Known Issues

None known at time of release.

---

## What's Next

- Wave 2 capstone: a reproducible **cross-validation benchmark report** on ≥1 irregular-Δt and ≥1 regular-Δt dataset with baselines (the OQ-7 "completed-state" exit criterion), now that the generators (`ar_p` / `mackey_glass` / `multi_sine` / `irregular_sine`) and the `/v1/crossval` route both exist.

---

## Links

- Changelog: `juniper-recurrence/CHANGELOG.md`
- Cross-validation layer design: [JUNIPER_MODEL_CORE_CROSSVAL_LAYER_DESIGN_2026-06-16.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-16_JUNIPER-ML_MODEL-CORE-CROSSVAL-LAYER-DESIGN.md)
- Roadmap (Wave 2): [JUNIPER_RECURRENCE_STATE_ASSESSMENT_AND_ROADMAP_2026-06-17.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-17_JUNIPER-RECURRENCE_STATE-ASSESSMENT-AND-ROADMAP.md)
- **Full Changelog:** https://github.com/pcalnon/juniper-recurrence/compare/juniper-recurrence-v0.1.0...juniper-recurrence-v0.1.1

---

## Contributors

- Paul Calnon (@pcalnon)
