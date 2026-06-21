# juniper-recurrence-model v0.1.4 Release Notes

**Release Date:** 2026-06-21
**Version:** 0.1.4
**Release Type:** PATCH (additive, backward-compatible)

---

## Overview

**DP-3 readout-spectrum, phase P2a** — the numpy **nonlinear** readout (Rung 2a). Adds
`RFFReadout` / `RFFReadoutSpec`: a genuine nonlinear readout with **no torch**, mapping the LMU
memory block through random Fourier features before a ridge solve
(`standardize(M) → RFF → GCV-ridge`). The default linear readout is unchanged, so this is an
additive patch release.

Design-of-record: `JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md` (juniper-ml,
ratified §8a, Rung 2a).

> **Status:** STABLE — additive / backward-compatible; the default readout and all 0.1.3 behaviour
> are unchanged.

---

## Release Summary

- **Release type:** PATCH
- **Primary focus:** DP-3 P2a — the RFF (random Fourier features) nonlinear readout
- **Breaking changes:** NO (the default `LinearReadoutSpec` path is byte-identical to 0.1.3)
- **numpy-only:** torch is still **not** a dependency (the torch MLP readout is the gated Rung 2b)

---

## What's New

### `RFFReadout` / `RFFReadoutSpec` (Rung 2a)

- `φ(M) = √(2/D)·cos(standardize(M)·W + b)` with `W ~ 𝒩(0, γ²I)` and `b ~ U[0, 2π)` sampled once at
  `fit` from the model's `random_seed` (data-independent, fixed across folds — cross-fold-safe via
  the immutable spec). Use via `LMURegressor(readout=RFFReadoutSpec(…))`.
- The design matrix is `[ φ(standardize(M)) | target_dt | 1 ]`: the RFF map applies to the **memory
  block only**; the `target_dt` side-channel and the bias stay linear (**D-WS4-2** preserved).
- New public exports: `RFFReadout`, `RFFReadoutSpec`. `"rff"` registered in the readout registry.

### Bandwidth + regularisation

- `γ` chosen by the **median heuristic** on standardized `M` (`gamma="median"`, default; ridge/GCV
  cannot select `γ`), or a fixed float.
- The readout penalty is **GCV-selected by default** (`ridge="gcv"`); ridge is mandatory for this
  high-variance rung. `D` is capped to the fold size (`p/n` guard; ridge handles the remainder).

### Robustness

- **Mandatory per-column standardization of `M`** (train-fold-only) keeps the isotropic `W` from
  being dominated by the high-energy low-order Legendre columns (≈25× RMS spread). Zero-variance
  columns are guarded (std → 1) so predictions stay finite (a `NaN` would fail the `np.array_equal`
  serialization contract).

---

## Serialization

`RFFReadout` persists `W`, `b`, the standardization stats, and the solved coefficients as float64
`readout__*` arrays plus a `meta["readout"]` descriptor (`kind`, `gamma`, `ridge`,
`n_features_out`). The bit-exact lossless round-trip for the `cos`-of-matmul path is **gated by an
RFF conformance subclass** (`TestRFFLMUConformance`) — in-process bit-exactness, no cross-machine
claim. `model._coef` is `None` for the (nonlinear) RFF readout.

---

## Testing

96 tests (+21 from 0.1.3): the RFF conformance subclass (full `TrainableModel` contract including
the bit-exact serialization round-trip + finite predictions), standardization (incl. the
zero-variance guard), median-`γ`, seed determinism (cross-fold), `D`-cap, lossless save/load, and a
**readout-level capacity demo** (RFF fits a bilinear `y = M₀·M₁` target a linear readout provably
cannot). Coverage 99.6%; `twine check` PASSED.

---

## Installation

```bash
pip install juniper-recurrence-model==0.1.4
```

---

## What's Next (DP-3 P2)

- **Capacity dataset** — a juniper-data generator `y = x(t−τ₁)·x(t−τ₂)` (a target that is a quadratic
  form in the LMU memory state, so a linear readout provably cannot fit it) to demonstrate the RFF
  readout's nonlinear capacity (a clear nonlinear ≫ linear r² gap).
- **Bench + findings** — the RFF row across datasets (tie on the existing near-linear datasets;
  the gap on the capacity dataset) and the findings-doc update.
- **HTTP `readout` enum** — expose readout selection over the app API + client.
- **Rung 2b (torch MLP)** — gated behind a `[torch]` extra, built only on a measured 2a lift.
