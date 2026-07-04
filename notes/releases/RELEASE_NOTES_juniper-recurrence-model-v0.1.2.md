# juniper-recurrence-model v0.1.2 Release Notes

**Release Date:** 2026-06-17
**Version:** 0.1.2
**Release Type:** PATCH

---

## Overview

Adopts the **`juniper-model-core` 0.2.0 cross-validation layer**: the model's `model-core` dependency ceiling is widened to admit 0.2.0, and a cross-validation **second-implementer proof** is added — driving the real `LMURegressor` through the generic `cross_validate` fold executor over irregular-Δt 3-D data. The model's runtime surface is unchanged from 0.1.1; this is a test + dependency release that unblocks downstream crossval consumption (the recurrence app `/v1/crossval` route).

> **Status:** STABLE — additive / dependency-only; no runtime API change from 0.1.1.

---

## Release Summary

- **Release type:** PATCH
- **Primary focus:** Adopt the model-core 0.2.0 crossval layer; prove it on a real model
- **Breaking changes:** NO
- **Priority summary:** Second-implementer validation of `juniper_model_core.crossval`; unblocks the recurrence app's evaluation route

---

## What's New

### Cross-validation second-implementer proof (`tests/test_crossval.py`)

Drives `LMURegressor` through `juniper_model_core.crossval.cross_validate` over a 3-D Δt fixture with `aux={dt, target_dt, seq_lengths}`, confirming the generic fold executor:

- slices the auxiliary arrays per fold (so the Δt path is engaged on a real model),
- recovers a high held-out **r²** when the correct `dt` is forwarded,
- **degrades measurably when `dt` is shuffled** (the guardrail that the per-step gaps actually matter), and
- is **deterministic** across repeated runs.

This is the cross-validation analogue of the recurrence model being model-core's second conformance implementer — it proves the new crossval layer is genuinely model-agnostic, not LMU-specific.

## Changed

- **`juniper-model-core` dependency ceiling widened to `<0.3.0`** (admits 0.2.0), and `juniper-model-core[crossval]>=0.2.0` added to the `[test]` extra. The model's **runtime surface is unchanged** — this only lets the 0.2.0 crossval submodule be imported by the test suite and by downstream consumers.

---

## What's Changed (commits)

- `039d7c4` test(recurrence-model): cross-validation second-implementer proof (model-core 0.2.0)

---

## Upgrade Notes

Backward-compatible; no migration. The widened `juniper-model-core<0.3.0` ceiling lets consumers that need the crossval layer resolve `juniper-model-core>=0.2.0` alongside `juniper-recurrence-model`.

```bash
pip install --upgrade juniper-recurrence-model
```

---

## Known Issues

None known at time of release.

---

## What's Next

- **Recurrence app `/v1/crossval` route + CLI** (crossval design §7 PR-2): the app-side consumption of the crossval layer, which this release unblocks — the app can now pin `juniper-recurrence-model>=0.1.2` + `juniper-model-core>=0.2.0` and resolve them together.

---

## Links

- Changelog: `juniper-recurrence-model/CHANGELOG.md`
- Cross-validation layer design: [JUNIPER_MODEL_CORE_CROSSVAL_LAYER_DESIGN_2026-06-16.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-16_JUNIPER-ML_MODEL-CORE-CROSSVAL-LAYER-DESIGN.md)
- Roadmap (Wave 2 / I2 / C2): [JUNIPER_RECURRENCE_STATE_ASSESSMENT_AND_ROADMAP_2026-06-17.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-17_JUNIPER-RECURRENCE_STATE-ASSESSMENT-AND-ROADMAP.md)
- **Full Changelog:** https://github.com/pcalnon/juniper-recurrence/compare/juniper-recurrence-model-v0.1.1...juniper-recurrence-model-v0.1.2

---

## Contributors

- Paul Calnon (@pcalnon)
