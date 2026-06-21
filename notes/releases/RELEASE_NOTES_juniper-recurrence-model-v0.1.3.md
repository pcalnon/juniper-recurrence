# juniper-recurrence-model v0.1.3 Release Notes

**Release Date:** 2026-06-21
**Version:** 0.1.3
**Release Type:** PATCH (additive, backward-compatible)

---

## Overview

**DP-3 readout-spectrum, phase P1.** Makes the LMU regressor's only trained surface — its
**readout** — a configurable spectrum, and ships the cheap data-ranked lever: a **GCV-selected
regularised linear readout**. Adds an immutable readout-**spec** API (`readout=<spec>`), a
`ridge="gcv"` closed-form generalised-cross-validation penalty selection, and a registry-based
serializer (schema 2) — all while keeping `LMURegressor(d, theta, ridge=…)` **byte-identical** to
0.1.2 and still loading pre-DP-3 `.npz` files.

Design-of-record: `JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md` (juniper-ml,
ratified §8a). Resolves the readout-regularisation follow-up (#28).

> **Status:** STABLE — additive / backward-compatible; no breaking change from 0.1.2.

---

## Release Summary

- **Release type:** PATCH
- **Primary focus:** DP-3 P1 — readout-spec refactor + GCV ridge selection
- **Breaking changes:** NO (byte-identical default path; pre-DP-3 `.npz` files still load)
- **Closes:** #28 (readout-regularisation follow-up)

---

## What's New

### Readout-spec API (`juniper_recurrence_model.readouts`)

- `Readout` / `ReadoutSpec` protocols, `LinearReadout`, and the immutable `LinearReadoutSpec(ridge=…)`.
- `LMURegressor(..., readout=<spec>)`: the readout is configured by an immutable **spec** and
  materialised fresh inside each `fit()` — so a spec shared across cross-validation folds can never
  leak one fold's fitted weights into another (the cross-fold trap a shared *live* readout would have).
- New public exports: `Readout`, `ReadoutSpec`, `LinearReadout`, `LinearReadoutSpec`, `RidgeParam`.

### GCV ridge selection (`ridge="gcv"`)

- Closed-form generalised cross-validation of the readout L2 penalty: **one SVD** of the centred
  design matrix + a **1-D log-grid search** (no held-out split, no inner-CV refit).
- The selected λ is written back to `model.ridge` and persisted in the serialized metadata
  (retraining fidelity).

### Serializer schema 2

- The readout persists its own fitted state as namespaced `readout__*` arrays + a nested
  `meta["readout"]` descriptor (with a `kind` tag), reconstructed via a readout registry on load.
- **Backward compatible:** a pre-DP-3 `.npz` (a top-level `coef`, no `meta["readout"]`) still loads —
  it reconstructs a linear readout from `meta["ridge"]`.

---

## Backward Compatibility

- `LMURegressor(d, theta, ridge=…)` is **byte-identical** to 0.1.2 (the default
  `LinearReadoutSpec(ridge=0.0)`); `ridge` widens to `float | Literal["gcv"]`.
- `meta["d"]` (memory order) and `model_type == "lmu"` stay frozen; `model._coef` is now a read-only
  forwarding property to the linear readout's coefficients (`None` before fit / for nonlinear readouts).
- The readout receives the memory block `M` only; `LMURegressor` keeps owning `target_dt` (a *linear*
  side-channel, appended after any nonlinearity) and the bias column (preserves D-WS4-2).

---

## Testing

- 56 prior tests + **19 new**; coverage **99.5%**; the full `juniper-model-core` conformance suite is
  green (including the bit-exact lossless serialization round-trip); `twine check` PASSED.

---

## Downstream

- The recurrence **app** + **client** expose `ridge="gcv"` over `POST /v1/train`, `POST /v1/crossval`,
  the `juniper-recurrence train --ridge gcv` CLI, and the Python client
  (juniper-recurrence#31 — requires this release).

---

## Installation

```bash
pip install juniper-recurrence-model==0.1.3
```

---

## What's Next (DP-3)

- **P2** — a numpy nonlinear readout (random Fourier features → GCV ridge) + a capacity-demonstrating
  nonlinear dataset + the HTTP `readout` enum.
- **P3** — an optional torch MLP readout behind a `[torch]` extra (gated on a measured P2 lift).
