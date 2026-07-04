# juniper-recurrence-model v0.1.4 Release Notes

**Release Date:** 2026-06-21
**Version:** 0.1.4
**Codename:** DP-3 P2a — RFF Nonlinear Readout
**Release Type:** PATCH

> Authored from the canonical `juniper-ml/notes/templates/TEMPLATE_RELEASE_NOTES.md`.

---

## Overview

DP-3 readout-spectrum **phase P2a**: adds `RFFReadout` / `RFFReadoutSpec`, a genuine **nonlinear**
readout for the LMU regressor with **no torch** — it maps the LMU memory block through random Fourier
features before a ridge solve (`standardize(M) → RFF → GCV-ridge`). The default linear readout is
unchanged, so this is an additive patch release. Design-of-record:
`juniper-ml/notes/JUNIPER_2026-06-20_JUNIPER-RECURRENCE_DP3-READOUT-SPECTRUM-DESIGN.md` (ratified §8a, Rung 2a).

> **Status:** STABLE — additive / backward-compatible; the default readout and all 0.1.3 behaviour are
> unchanged, and the package remains numpy-only (no torch dependency).

---

## Release Summary

- **Release type:** PATCH
- **Primary focus:** New feature — the RFF (random Fourier features) nonlinear readout (Rung 2a)
- **Breaking changes:** NO (the default `LinearReadoutSpec` path is byte-identical to 0.1.3)
- **Priority summary:** DP-3 P2a delivered; numpy-only preserved (the torch MLP readout is the gated Rung 2b)

---

## Features Summary

| ID       | Feature                                   | Status  | Version | Phase |
| -------- | ----------------------------------------- | ------- | ------- | ----- |
| Rung 0/1 | Linear readout + GCV ridge selection      | Done    | 0.1.3   | P1    |
| Rung 2a  | RFF nonlinear readout (`RFFReadoutSpec`)   | Done    | 0.1.4   | P2a   |
| Rung 2b  | torch MLP readout (behind a `[torch]` extra) | Planned | -     | P3    |

**Cumulative DP-3 Phase Status:**

| Phase | Items | Status |
| ----- | ----- | ------ |
| P1: readout-spec refactor + Rung 0/1 (GCV) | linear readout, GCV, serializer schema 2 | Complete (0.1.3) |
| P2a: Rung 2a (RFF) | RFF readout + conformance subclass | Complete (0.1.4) |
| P2 (remaining): capacity dataset + bench + HTTP enum | juniper-data generator, bench rows, app enum | In Progress |
| P3: Rung 2b (torch) | MLP readout, gated on a measured 2a lift | Planned |

---

## What's New

### Nonlinear readout (Rung 2a)

#### `RFFReadout` / `RFFReadoutSpec`

A numpy nonlinear readout: `φ(M) = √(2/D)·cos(standardize(M)·W + b)` with `W ~ 𝒩(0, γ²I)` and
`b ~ U[0, 2π)` sampled once at `fit` from the model's `random_seed` (data-independent, fixed across
folds — cross-fold-safe via the immutable spec). The design matrix is `[ φ(standardize(M)) | target_dt | 1 ]`:
the RFF map applies to the **memory block only**; the `target_dt` side-channel and the bias stay linear
(D-WS4-2 preserved). Use via `LMURegressor(readout=RFFReadoutSpec(…))`.

**Changes:**

- New public exports `RFFReadout`, `RFFReadoutSpec`; `"rff"` registered in the readout registry.
- Bandwidth `γ` chosen by the **median heuristic** on standardized `M` (`gamma="median"`, default;
  ridge/GCV cannot select `γ`), or a fixed float.
- Readout penalty **GCV-selected by default** (`ridge="gcv"`); ridge is mandatory for this
  high-variance rung. `D` is capped to the fold size (`p/n` guard; ridge handles the remainder).
- **Mandatory per-column standardization of `M`** (train-fold-only); zero-variance columns guarded
  (std → 1) so predictions stay finite (a `NaN` would fail the `np.array_equal` serialization contract).

### Serialization (schema 2, `"rff"`)

`RFFReadout` persists `W`, `b`, the standardization stats, and the solved coefficients as float64
`readout__*` arrays plus a `meta["readout"]` descriptor (`kind`, `gamma`, `ridge`, `n_features_out`).

**Changes:**

- The bit-exact lossless round-trip for the `cos`-of-matmul path is **gated by an RFF conformance
  subclass** (`TestRFFLMUConformance`) — in-process bit-exactness; no cross-machine claim.
- `model._coef` is `None` for the (nonlinear) RFF readout, consistent with any non-linear rung.
- `model.py` is **unchanged** — the P1 readout-spec refactor is readout-agnostic, so RFF is purely additive.

---

## Bug Fixes

None — additive feature release.

---

## Improvements

### Test Count Growth

| Version | Tests | Change |
| ------- | ----- | ------ |
| 0.1.3   | 75    | —      |
| 0.1.4   | 96    | +21    |

**Total new tests in 0.1.4:** 21 (RFF conformance subclass + RFF unit tests, including a readout-level
capacity demo: RFF fits a bilinear `y = M₀·M₁` target a linear readout provably cannot).

---

## API Changes

None at the Python-library surface beyond the additive exports above (`RFFReadout`, `RFFReadoutSpec`).
This package has no HTTP API — readout selection over HTTP is the recurrence app's P2c increment.

---

## Test Results

### Test Suite

| Metric            | Result      |
| ----------------- | ----------- |
| **Tests passed**  | 96          |
| **Tests skipped** | 0           |
| **Tests failed**  | 0           |
| **Runtime**       | ~2.5 seconds |
| **Coverage**      | 99.6% overall |

### Coverage Details

| Component        | Coverage | Target (CI gate) | Status      |
| ---------------- | -------- | ---------------- | ----------- |
| `readouts.py`    | 100%     | 90%              | Exceeded    |
| `model.py`       | 99%      | 90%              | Exceeded    |
| `units/lmu_varstep.py` | 100% | 90%            | Exceeded    |
| `data.py`        | 100%     | 90%              | Exceeded    |

---

## Upgrade Notes

This is a backward-compatible PATCH release. No migration steps required.

```bash
pip install --upgrade juniper-recurrence-model==0.1.4
```

`LMURegressor(d, theta, ridge=…)` is byte-identical to 0.1.3; the RFF readout is opt-in via
`LMURegressor(readout=RFFReadoutSpec(…))`.

---

## Known Issues

- **`model.py` coverage at 99%** — the two uncovered lines are the defensive `X must be 3-D` guards in
  `_memory_block` / `fit`; not a functional issue.
- **RFF bit-exactness is in-process only** — the lossless round-trip is guaranteed for save→load within
  one process/BLAS; no cross-machine bit-exactness is claimed (torch/`cos`-of-matmul forwards are not
  cross-BLAS bit-stable, and the conformance kit only re-runs `predict` in-process).

---

## What's Next

### Planned for the rest of DP-3 P2

- **Capacity dataset** — a juniper-data generator `y = x(t−τ₁)·x(t−τ₂)` (a target that is a quadratic
  form in the LMU memory state, so a linear readout provably cannot fit it) to demonstrate the RFF
  readout's nonlinear capacity (a clear nonlinear ≫ linear r² gap).
- **Bench + findings** — the RFF row across datasets (a tie on the existing near-linear datasets; the
  gap on the capacity dataset) and the findings-doc update.
- **HTTP `readout` enum** — expose readout selection over the recurrence app API + client.

### Planned for P3

- **Rung 2b (torch MLP)** — behind a `[torch]` extra, built only on a measured 2a lift.

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date       | Description                                            |
| ------- | ---------- | ------------------------------------------------------ |
| 0.1.4   | 2026-06-21 | DP-3 P2a — RFF nonlinear readout (Rung 2a)             |
| 0.1.3   | 2026-06-20 | DP-3 P1 — readout-spec refactor + GCV ridge selection |
| 0.1.2   | 2026-06-17 | Adopt the juniper-model-core 0.2.0 cross-validation layer |
| 0.1.1   | 2026-06-17 | CHANGELOG / test-count fixes                           |
| 0.1.0   | 2026-06-16 | Δt-native LMU — first recurrence release               |

---

## Links

- [Full Changelog](../../juniper-recurrence-model/CHANGELOG.md)
- [DP-3 design-of-record](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-20_JUNIPER-RECURRENCE_DP3-READOUT-SPECTRUM-DESIGN.md)
- [Previous Release](RELEASE_NOTES_juniper-recurrence-model-v0.1.3.md)
