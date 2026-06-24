# juniper-recurrence-model v0.1.5 Release Notes

**Release Date:** 2026-06-24
**Version:** 0.1.5
**Codename:** DP-3 P3 — torch MLP Readout (Rung 2b)
**Release Type:** PATCH

> Authored from the canonical `juniper-ml/notes/templates/TEMPLATE_RELEASE_NOTES.md`.

---

## Overview

DP-3 readout-spectrum **phase P3**: adds the optional **torch MLP readout** (`MLPReadout` /
`MLPReadoutSpec`, Rung 2b) behind a new `[torch]` extra, plus the `LMURegressor` validation plumbing
that feeds it. The base package stays **numpy-only / torch-free** (torch is imported lazily), the
default linear readout is unchanged, and all closed-form rungs are byte-identical to 0.1.4 — so this
is an additive PATCH release. Design-of-record:
`juniper-ml/notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md` §4 (Rung 2b), ratified
**GO** 2026-06-23 (`juniper-ml/notes/JUNIPER_DECISIONS_RATIFIED_2026-06-23.md` D5) — built as capability
insurance for future complex / hybrid datasets and as a falsifiable demonstration; the current dataset
catalog does not itself require it.

> **Status:** STABLE — additive / backward-compatible. The base package remains numpy-only; torch is
> pulled only by the opt-in `juniper-recurrence-model[torch]` extra, and `torch` is never imported at
> base package load.

---

## Release Summary

- **Release type:** PATCH
- **Primary focus:** New feature — the optional torch MLP readout (Rung 2b) + `LMURegressor` validation plumbing
- **Breaking changes:** NO (the default `LinearReadoutSpec` path and the closed-form rungs are byte-identical to 0.1.4)
- **Priority summary:** DP-3 P3 delivered; numpy-only base preserved (torch is opt-in behind the `[torch]` extra)

---

## Features Summary

| ID       | Feature                                       | Status | Version | Phase |
| -------- | --------------------------------------------- | ------ | ------- | ----- |
| Rung 0/1 | Linear readout + GCV ridge selection          | Done   | 0.1.3   | P1    |
| Rung 2a  | RFF nonlinear readout (`RFFReadoutSpec`)       | Done   | 0.1.4   | P2a   |
| Rung 2b  | torch MLP readout (behind a `[torch]` extra)   | Done   | 0.1.5   | P3    |

**Cumulative DP-3 Phase Status:**

| Phase | Items | Status |
| ----- | ----- | ------ |
| P1: readout-spec refactor + Rung 0/1 (GCV) | linear readout, GCV, serializer schema 2 | Complete (0.1.3) |
| P2a: Rung 2a (RFF) | RFF readout + conformance subclass | Complete (0.1.4) |
| P2 (remaining): capacity dataset + bench | juniper-data `delay_product`, bench rows | Complete (data 0.9.0, bench shipped) |
| P3: Rung 2b (torch) | MLP readout + validation plumbing | Complete (0.1.5) |

---

## What's New

### Optional torch MLP readout (Rung 2b)

#### `MLPReadout` / `MLPReadoutSpec`

An optional torch MLP readout behind the new `[torch]` extra. Architecture:
`standardize(M) → GELU trunk (h→h) → linear head over [trunk | extra]`, with internal target
standardization. The `target_dt` side-channel and the head bias enter **after** the nonlinearity
(D-WS4-2 preserved). Use via `LMURegressor(readout=MLPReadoutSpec(…))`.

**Changes:**

- New public exports `MLPReadout`, `MLPReadoutSpec`; `"mlp"` **lazily** registered in the readout
  registry (so the base import never pulls in torch).
- New `[torch]` extra pinning `torch>=2.10.0` — the ecosystem CVE-2025-3001 security floor, which ships
  cp314 wheels (no Python-3.14 gap). A plain dependency never activates another package's extras, so the
  recurrence app / client stay torch-free.
- Training is **CPU-only, single-threaded, float32, `use_deterministic_algorithms(True)`**, with the seed
  set before module init. An in-process save→load→predict round-trip is therefore bit-exact within a
  machine (no cross-machine claim).
- **Validation-driven early stopping** — `MLPReadout` early-stops on the validation block the model
  supplies (relative `min_delta`, `patience`), keeping the best-validation weights; with no validation
  data it trains the full `max_epochs` budget.

### `LMURegressor` validation plumbing

`LMURegressor.fit(X, y, *, X_val=…, y_val=…)` now builds a held-out feature block and hands it to
`readout.fit`, so an early-stopping rung (the MLP) gets a held-out signal. The closed-form linear and
RFF rungs accept and ignore the validation arrays.

**Changes:**

- Optional `*_val` timing kwargs (`dt_val` / `target_dt_val` / `readout_mask_val` / `seq_lengths_val`)
  make the validation block Δt-faithful; when absent (the conformance / crossval convention) it falls
  back to the uniform-grid construction — adequate as an early-stop signal.
- `TrainResult.n_epochs` now reflects the readout's true trained-epoch count
  (`max(1, getattr(readout, "n_epochs_", 1))`) and `TrainResult.stopped_reason` reflects how it stopped
  (`"early_stopping"` / `"max_epochs"` for the MLP). Closed-form readouts expose neither, so they read as
  the canonical single-solve `1` / `"converged"` — **preserving the crossval `n_epochs == 1` invariant**.
- When validation data is present, the `epoch_end` / `training_end` training events carry a `val_metrics`
  payload alongside the existing training metrics.

### Serialization (schema 2, `"mlp"`)

`MLPReadout` persists its layer weights/biases + the standardization stats as float32 `readout__*`
arrays plus a `meta["readout"]` descriptor (`kind`, `hidden`, dims, hyper-parameters). State is saved as
**named numpy arrays — never `torch.save`** (the serializer loads with `allow_pickle=False`).

**Changes:**

- The bit-exact lossless round-trip is **gated by an `MLPReadout` conformance subclass**
  (`TestMLPLMUConformance`) — in-process bit-exactness; no cross-machine claim. The standardization
  stats are stored as float32 so a reloaded readout matches the original exactly.
- `model._coef` is `None` for the (nonlinear) MLP readout, consistent with any non-linear rung.

---

## Bug Fixes

None — additive feature release.

---

## Improvements

### Test Count Growth

| Version | Tests | Change |
| ------- | ----- | ------ |
| 0.1.4   | 96    | —      |
| 0.1.5   | 120   | +24    |

**Total new tests in 0.1.5:** 24 (104 in the base no-torch job + 16 in the optional `test-torch` job).
Covers the MLP unit contract (fit/predict, within-machine determinism, the bit-exact save→load
round-trip, validation-driven early stopping), an `MLPReadout` model-core conformance subclass, the base
validation-plumbing path via the linear readout (the `n_epochs == 1` / `"converged"` invariant, the
`val_metrics` payload, `dt_val` consumption), and the MLP early-stop surfacing through `TrainResult`.

### CI

- New **non-gating `test-torch` CI job** installs `.[test,torch]` and runs the Rung 2b tests, which the
  base 3.12/3.13/3.14 matrix skips (`pytest.importorskip("torch")`). Deliberately kept off the required
  gate — the torch wheel adds ~1-3 GB — so torch-readout regressions are visible but do not block merges.
- `_readout_mlp.py` is omitted from the base coverage gate (it is never executed without the extra).

---

## API Changes

Additive only: the new public exports `MLPReadout` / `MLPReadoutSpec`, the `[torch]` extra, the optional
`X_val` / `y_val` / `*_val` keywords on `LMURegressor.fit`, and the optional `M_val` / `extra_val` /
`y_val` keywords on the `Readout` protocol's `fit` (the closed-form rungs accept and ignore them). This
package has no HTTP API — readout selection over HTTP is the recurrence app's increment.

---

## Test Results

### Test Suite

| Metric            | Result        |
| ----------------- | ------------- |
| **Tests passed**  | 120 (104 base + 16 torch) |
| **Tests skipped** | 0             |
| **Tests failed**  | 0             |
| **Runtime**       | ~3 seconds (base); ~1m torch job |
| **Coverage**      | 97.60% base / 98.08% with `[torch]` |

### Coverage Details

| Component        | Coverage | Target (CI gate) | Status   |
| ---------------- | -------- | ---------------- | -------- |
| `readouts.py`    | 97.26%   | 90%              | Exceeded |
| `model.py`       | 98.15%   | 90%              | Exceeded |
| `units/lmu_varstep.py` | 100% | 90%             | Exceeded |
| `data.py`        | 97.44%   | 90%              | Exceeded |
| `_readout_mlp.py` | (omitted) | —              | torch-only; exercised by the `test-torch` job |

---

## Upgrade Notes

This is a backward-compatible PATCH release. No migration steps required.

```bash
pip install --upgrade juniper-recurrence-model==0.1.5          # base (numpy-only)
pip install --upgrade "juniper-recurrence-model[torch]==0.1.5" # + the optional Rung 2b MLP readout
```

`LMURegressor(d, theta, ridge=…)` is byte-identical to 0.1.4; the MLP readout is opt-in via
`LMURegressor(readout=MLPReadoutSpec(…))` and requires the `[torch]` extra.

---

## Known Issues

- **MLP bit-exactness is in-process only** — the lossless round-trip is guaranteed for save→load within
  one process/BLAS; no cross-machine bit-exactness is claimed (torch forwards are not cross-BLAS
  bit-stable, and the conformance kit only re-runs `predict` in-process).
- **`model.py` coverage at 98.15%** — the two uncovered lines are the defensive `X must be 3-D` guards in
  `_memory_block` / `fit`; not a functional issue.
- **Capacity is dataset-dependent.** Rung 2b is capability insurance: on the current Juniper dataset
  catalog the real-data ceiling is ≈0, so the MLP readout is not indicated for present datasets — it
  exists for future complex / hybrid targets (and to demonstrate, falsifiably, when it adds nothing).

---

## What's Next

### Follow-ups (publish-gated on this 0.1.5 release)

- **Bench `mlp` row** — add the MLP readout to the benchmark on the `delay_product` capacity dataset
  (a quadratic-form target a linear readout provably cannot fit), and update the findings doc.
- **HTTP `readout="mlp"` enum** — expose MLP readout selection + hyper-parameters over the recurrence
  app API + client (the app's increment; floor-bumps the app's model pin to `>=0.1.5`).

---

## Contributors

- Paul Calnon

---

## Version History

| Version | Date       | Description                                                  |
| ------- | ---------- | ------------------------------------------------------------ |
| 0.1.5   | 2026-06-24 | DP-3 P3 — torch MLP readout (Rung 2b) + validation plumbing  |
| 0.1.4   | 2026-06-21 | DP-3 P2a — RFF nonlinear readout (Rung 2a)                   |
| 0.1.3   | 2026-06-20 | DP-3 P1 — readout-spec refactor + GCV ridge selection        |
| 0.1.2   | 2026-06-17 | Adopt the juniper-model-core 0.2.0 cross-validation layer    |
| 0.1.1   | 2026-06-17 | CHANGELOG / test-count fixes                                 |
| 0.1.0   | 2026-06-16 | Δt-native LMU — first recurrence release                     |

---

## Links

- [Full Changelog](../../juniper-recurrence-model/CHANGELOG.md)
- [DP-3 design-of-record](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md)
- [Decisions ratified 2026-06-23 (D5)](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_DECISIONS_RATIFIED_2026-06-23.md)
- [Previous Release](RELEASE_NOTES_juniper-recurrence-model-v0.1.4.md)
