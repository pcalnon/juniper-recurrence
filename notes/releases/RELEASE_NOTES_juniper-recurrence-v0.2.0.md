# juniper-recurrence v0.2.0 Release Notes

**Release Date:** 2026-06-24
**Version:** 0.2.0
**Codename:** DP-3 Readout Spectrum at the Edge + Container Image
**Release Type:** MINOR

> Authored from the canonical `juniper-ml/notes/templates/TEMPLATE_RELEASE_NOTES.md`.

---

## Overview

The first release that exposes the **full DP-3 readout spectrum over the service edge**. `POST /v1/train`,
`POST /v1/crossval`, and the `train` CLI now accept a `readout` enum — `"linear"` (default, back-compat),
`"rff"` (numpy random-Fourier-features, Rung 2a), and `"mlp"` (torch MLP, Rung 2b) — plus `ridge="gcv"`
for closed-form GCV penalty selection. The service builds the matching immutable readout spec and passes
it to `LMURegressor`. This release also ships a deployable **container image**. Every change is additive
and backward compatible (an unset `readout` is byte-identical to the prior linear-readout behaviour), so
this is a MINOR release.

> **Status:** STABLE — additive / backward-compatible. The base service stays **numpy-only**; torch is
> pulled only by the opt-in `[torch]` extra and a torch-less deployment still starts (and rejects
> `readout="mlp"` with a clear 503).

---

## Release Summary

- **Release type:** MINOR (a batch of additive, backward-compatible edge features since 0.1.1)
- **Primary focus:** the DP-3 `readout` enum (`linear` / `rff` / `mlp`) + `ridge="gcv"` over HTTP & CLI
- **Breaking changes:** NO (an unset `readout` / `ridge` reproduces 0.1.1 behaviour exactly)
- **Also:** a slim multi-stage container image; PyPI-only install of the model + bench dependencies

---

## Features Summary

| Feature                                              | Status | Phase |
| ---------------------------------------------------- | ------ | ----- |
| `ridge="gcv"` over `/v1/train` `/v1/crossval` + CLI  | Done   | P1    |
| `readout="rff"` (Rung 2a) + `rff_features`/`rff_gamma`| Done   | P2c   |
| `readout="mlp"` (Rung 2b) + `mlp_*` + `[torch]` extra | Done   | P3    |
| Container image (`Dockerfile`, Docker CI smoke test)  | Done   | WS-7  |

---

## What's New

### DP-3 readout enum at the edge

- **`readout="rff"` (P2c)** — selects the numpy nonlinear random-Fourier-feature readout (Rung 2a),
  configured by `rff_features` / `rff_gamma`; rejected unless `readout="rff"`.
- **`readout="mlp"` (P3)** — selects the torch MLP readout (Rung 2b), configured by the optional
  `mlp_hidden` / `mlp_weight_decay` / `mlp_lr` / `mlp_max_epochs` / `mlp_patience` (each defaults to the
  `MLPReadoutSpec` value). `ridge` is rejected with `"mlp"` (the MLP regularises via weight decay). The
  MLP needs torch **at runtime**, kept optional behind a new **`[torch]` extra**; a deployment without it
  still starts and rejects `readout="mlp"` with a clear **503** (the spec import itself is torch-free).
- **`ridge="gcv"` (P1)** — `ridge` accepts `"gcv"` in addition to a non-negative float, requesting the
  model's closed-form generalised-cross-validation penalty selection. `default_ridge` widens to
  `float | Literal["gcv"]` (default `0.0`, unchanged).
- The readout enum and its params are wired identically across `/v1/train`, `/v1/crossval`, and the
  `train` CLI through one shared translation point (`_readout.build_lmu_regressor`); a rung's params are
  rejected (422 at the edge, exit 2 on the CLI) when a different rung is selected — no silent no-op.

### Container image (WS-7)

A multi-stage, slim (~77 MB) image. The LMU stack is numpy-only, so the build installs the app + the
`[observability]` extra from PyPI (no CPU-torch lock dance). Runs as a non-root `juniper` user;
`ENTRYPOINT ["juniper-recurrence"]` / `CMD ["serve"]`; an HTTP `HEALTHCHECK` probes `/v1/health` (40s
start-period). A `Docker Build & Smoke Test` CI job builds the image and asserts `/v1/health` → 200.

---

## Bug Fixes

- **CLI `--rff-features` / `--rff-gamma` rejected without `--readout rff`** (P2c follow-up). The rule was
  enforced at the HTTP edge but the `train` CLI silently dropped the RFF-only knobs; the check now lives
  in the shared `build_lmu_regressor`, so the CLI and HTTP behave identically (the CLI exits 2).

---

## API Changes

Additive only. `TrainRequest` / `CrossValRequest` gain `readout`, `rff_features`, `rff_gamma`, and
`mlp_hidden` / `mlp_weight_decay` / `mlp_lr` / `mlp_max_epochs` / `mlp_patience` (all optional); `ridge`
widens to accept `"gcv"`. The `train` CLI gains `--readout`, `--rff-*`, and `--mlp-*` flags. New optional
`[torch]` extra. No response-shape changes; an unset `readout`/`ridge` reproduces 0.1.1 behaviour.

---

## Dependency Changes

- **`juniper-recurrence-model` floor → `>=0.1.5`** (`<0.2.0` unchanged) — 0.1.5 ships `MLPReadoutSpec`
  (the Rung 2b readout) the `readout="mlp"` enum constructs. Publish-first: 0.1.5 is already live on PyPI.
- **New `[torch]` extra** (`juniper-recurrence-model[torch]>=0.1.5`) — the runtime torch dependency for
  `readout="mlp"`, kept off the base deps (install `pip install 'juniper-recurrence[torch]'`).
- **`[bench]` extra → `juniper-data>=0.9.0`** — the synthetic Δt + `delay_product` generators are on PyPI,
  so the benchmark installs cleanly from PyPI (no editable-sibling step). A `[bench-torch]` extra adds the
  MLP bench row.

---

## Test Results

The app's `ci-recurrence-app.yml` (pre-commit, ruff, the unit matrix on 3.12/3.13/3.14, build, the Docker
smoke test) is green. The MLP edge tests toggle `find_spec` rather than importing torch, so the readout
wiring + the torch-absent 503 path are covered in the torch-free unit job.

---

## Upgrade Notes

Backward-compatible MINOR release. No migration steps.

```bash
pip install --upgrade juniper-recurrence==0.2.0           # base (numpy-only service)
pip install --upgrade "juniper-recurrence[torch]==0.2.0"  # + the optional Rung 2b MLP readout
```

A deployment that never selects `readout="mlp"` needs no torch. `readout="mlp"` on a torch-less
deployment returns a clear **503** (the service still starts and serves every other readout).

---

## Known Issues

- **`readout="mlp"` requires the `[torch]` extra at runtime** — without it the enum is accepted but the
  request returns a 503 (by design — the capability is genuinely unavailable on that deployment).
- The WS-7 deploy-compose integration (host 8211 → container 8210) follows in juniper-deploy.

---

## What's Next

- The juniper-ml meta-package `[recurrence]` extra pin update (`juniper-recurrence>=0.2.0`) so the
  aggregate install resolves this release.
- The WS-7 compose integration for the container image.

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date       | Description                                                              |
| ------- | ---------- | ------------------------------------------------------------------------ |
| 0.2.0   | 2026-06-24 | DP-3 readout enum (`rff`+`mlp`) + `ridge="gcv"` over HTTP/CLI; `[torch]` extra; container image |
| 0.1.1   | 2026-06-17 | Early service hardening (see CHANGELOG)                                  |
| 0.1.0   | 2026-06-17 | First `juniper-recurrence` application release                           |

---

## Links

- [Full Changelog](../../juniper-recurrence/CHANGELOG.md)
- [DP-3 design-of-record](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-20_JUNIPER-RECURRENCE_DP3-READOUT-SPECTRUM-DESIGN.md)
- [Evaluation findings (§3.4 MLP capacity)](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-18_JUNIPER-RECURRENCE_EVALUATION-FINDINGS.md)
- [Model release (Rung 2b)](RELEASE_NOTES_juniper-recurrence-model-v0.1.5.md)
