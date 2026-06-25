# juniper-recurrence-model v0.1.0 Release Notes

**Release Date:** 2026-06-16
**Version:** 0.1.0
**Codename:** Δt-Native LMU — First Recurrence Release
**Release Type:** MINOR (initial release)

---

## Overview

First published release of **`juniper-recurrence-model`** — the model-specific core for the `juniper-recurrence` application and the Juniper platform's first **irregular-Δt time-series** model. It ships the **P3-C / Approach-C Legendre Memory Unit** (a closed-form, continuous-time, C1-clean recurrent memory), a fixed-order LMU **regressor** implementing the shared `juniper-model-core` `TrainableModel` contract, and a lean loader that consumes the WS-1 3-D/Δt NPZ data contract end-to-end.

> **Status:** ALPHA — first release; the public API may evolve before 1.0. numpy-only, autodiff-free.

---

## Release Summary

- **Release type:** MINOR (initial 0.1.0)
- **Primary focus:** NEW FEATURES — the WS-4 model build + the §9.1c data-consumer wiring
- **Breaking changes:** NO (initial release)
- **Priority summary:** WS-4 (model) complete; §9.1a (fixed-Δt negative control) + §9.1c (3-D ingestion, recurrence-side) delivered

---

## Features Summary

| ID       | Feature                                                | Status  | Version |
| -------- | ------------------------------------------------------ | ------- | ------- |
| WS-4     | Fixed-order Δt-native LMU regressor (`LMURegressor`)   | Done    | 0.1.0   |
| D-WS4-1  | Per-feature identity read-in                           | Done    | 0.1.0   |
| D-WS4-2  | `target_dt` as a readout feature                       | Done    | 0.1.0   |
| §9.1a    | Fixed-Δt negative control (`FixedStepLMUMemory` foil)  | Done    | 0.1.0   |
| §9.1c    | 3-D sequence NPZ consumer path (`load_sequence_npz`)   | Done    | 0.1.0   |
| §4.2/WS-8| Grown-cascade LMU / distributed worker                 | Planned | —       |

---

## What's New

### Δt-native memory (Approach-C)

#### `VariableStepLMUMemory` + `lmu_matrices`

The Legendre Memory Unit core: the fixed, closed-form LegT (HiPPO) state matrices and their **exact variable-step zero-order-hold discretisation** via a one-time eigendecomposition — the memory is evaluated at each observation's *real* Δt. C1-clean: no ODE solver, no autodiff-through-a-solver; `A`/`B` are never trained.

**Changes:**

- `rollout_batch` — batched, multi-channel eigenbasis ZOH rollout (parity-tested per-(sample, feature) against the reference `rollout`).
- Grid-invariance verified (`e_irr < 3·e_reg + 0.02`) plus the §9.1a **fixed-Δt negative control** (`FixedStepLMUMemory` baked at the mean gap) proving the per-step Δt adaptation does real work — it degrades ~2–4× on an irregular grid.

### The regressor

#### `LMURegressor` (`TrainableModel`) + `LMUSerializer`

A fixed-order, Δt-native regressor implementing `juniper-model-core`'s `TrainableModel` contract.

**Changes:**

- Per-feature identity read-in (D-WS4-1); a **closed-form least-squares readout** (the only trained surface) — numpy-only, no torch.
- `target_dt` as an optional readout feature (D-WS4-2); data-driven `theta` default (median per-window elapsed time), or pin it explicitly.
- `predict(X, *, dt=…)` widens the contract with optional sequence keywords (`dt`/`target_dt`/`readout_mask`/`seq_lengths`; uniform-`dt` fallback when omitted); regression-only metrics, never `accuracy` (RK-6).
- `LMUSerializer` — lossless `.npz` + JSON round-trip (the fixed memory is recomputed from `d`/θ on load).

### The data path (§9.1c)

#### `load_sequence_npz` + `SequenceData`

A lean, numpy-only loader for the **WS-1 3-D/Δt NPZ contract** (per-split `X` / `y_reg` / `dt`-or-`t` / `target_dt` / `seq_lengths`) that feeds `LMURegressor` via `SequenceData.fit_kwargs()`. **No `juniper-data-client` dependency** — data-client's `validate_npz_contract` remains the app's authoritative fetch-path validator.

This is the **§9.1c consumer wiring**: `juniper-recurrence` ingests the irregular-Δt 3-D contract, while **`juniper-cascor` stays a 2-D stateless feed-forward model and is untouched** (per the design-direction review — the "cascor 3-D ingestion gate" analysis showed cascor cannot cheaply ingest 3-D, which is precisely the justification for a *separate* recurrence model rather than grafting recurrence onto cascor).

---

## Improvements

### Test coverage

| Component                  | Coverage |
| -------------------------- | -------- |
| `units/lmu_varstep.py`     | 100%     |
| `data.py`                  | 100%     |
| `model.py`                 | 99%      |
| **Total**                  | **~99%** |

---

## API Changes

N/A — this is a library package (no HTTP service endpoints). The public Python surface added in 0.1.0: `LMURegressor`, `LMUSerializer`, `VariableStepLMUMemory`, `lmu_matrices`, `load_sequence_npz`, `SequenceData`.

---

## Test Results

| Metric           | Result                                     |
| ---------------- | ------------------------------------------ |
| **Tests passed** | 53                                         |
| **Tests failed** | 0                                          |
| **Coverage**     | ~99% (CI gate 90%)                         |
| **Python**       | 3.12 / 3.13 / 3.14 (CI matrix)             |
| **Lint / build** | `ruff` clean; `build` + `twine check` pass |

---

## Upgrade Notes

Initial release — install from PyPI:

```bash
pip install juniper-recurrence-model==0.1.0
```

Pulls `numpy>=1.24` and `juniper-model-core>=0.1.0,<0.2.0`. Backward-compatible (first release; nothing to migrate).

---

## Known Issues

None functional. The following are **deliberately deferred** (not defects):

- **Dense many-to-many readout** — 0.1.0 is many-to-one (one target per window).
- **Trained projection read-in / nonlinear readout** — the point at which a torch dependency would enter; deferred.
- **Service / app layer** — the `juniper-service-core`-backed `juniper-recurrence` app (exposing the model + the authoritative `validate_npz_contract` fetch path as a service) is a later workstream.
- **Star-free ceiling** — the LMU inherits it by design (a property of diagonal/nonnegative state-space models); no Juniper dataset requires breaking it, so this is not a functional limitation for the platform's workload.

---

## What's Next

### Planned

- **`juniper-recurrence` application** — a `juniper-service-core`-backed FastAPI/CLI app wrapping `LMURegressor` + the data path.
- **`juniper-canopy` generalization** — a schema-driven, model-agnostic UI (regression metrics, time-series plots, topology rendering from `describe_topology()`).
- **Model increments** — trained projection read-in / nonlinear readout; dense many-to-many readout.

---

## Contributors

- Paul Calnon (@pcalnon)
- Claude Code (Claude Opus 4.8)

---

## Version History

| Version | Date       | Description                                                                  |
| ------- | ---------- | ---------------------------------------------------------------------------- |
| 0.1.0   | 2026-06-16 | Initial release — Δt-native LMU regressor (WS-4) + 3-D sequence path (§9.1c) |

---

## Links

- **PyPI:** <https://pypi.org/project/juniper-recurrence-model/0.1.0/>
- **Full changelog:** [`juniper-recurrence-model/CHANGELOG.md`](https://github.com/pcalnon/juniper-recurrence/blob/main/juniper-recurrence-model/CHANGELOG.md)
- **Design of record:** `notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md` (juniper-ml)
- **WS-4 build plan:** `notes/JUNIPER_RECURRENCE_WS4_MODEL_BUILD_PLAN_2026-06-15.md` (juniper-ml)
- **Δt-handling / Approach-C math:** `notes/JUNIPER_RECURSE_DELTA_T_HANDLING_2026-06-05.md` (juniper-ml)

---

🤖 Release notes generated with [Claude Code](https://claude.com/claude-code), following `notes/templates/TEMPLATE_RELEASE_NOTES.md`.
