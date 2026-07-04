# juniper-recurrence-model v0.1.1 Release Notes

**Release Date:** 2026-06-17
**Version:** 0.1.1
**Release Type:** PATCH

---

## Overview

A documentation-only patch that fixes the **`juniper-recurrence-model` PyPI quick-start**, which was broken in 0.1.0: the README referenced symbols that do not exist, so the documented import raised `ImportError` on the published landing page. No code changes — the public API and behaviour are identical to 0.1.0.

> **Status:** STABLE — drop-in for 0.1.0; no migration.

---

## Release Summary

- **Release type:** PATCH
- **Primary focus:** Documentation correctness
- **Breaking changes:** NO
- **Priority summary:** Fixes a broken PyPI quick-start (roadmap D-3 / R3)

---

## Bug Fixes

### README API drift — broken quick-start import (v0.1.1)

**Problem:** The 0.1.0 README quick-start imported `FixedOrderLMURegressor` and `LMURegressorSerializer` and referenced `tests/test_lmu_conformance.py` — none of which exist. `from juniper_recurrence_model import FixedOrderLMURegressor, LMURegressorSerializer` raised `ImportError`, including on the PyPI landing page.

**Root Cause:** The README predated the final public-API names; the symbols were renamed during consolidation but the README was not updated.

**Solution:** Replaced all occurrences with the real public API — `LMURegressor` / `LMUSerializer` (test file `tests/test_conformance.py`). Both README examples now execute end-to-end against the installed package (memory rollout + regressor fit/predict/serialize, r²≈0.78, `model_type=lmu`).

**Files:** `README.md`, `CHANGELOG.md`

---

## What's Changed (commits)

- `534fa69` docs(recurrence-model): fix README API drift (broken quick-start import) (#11)
- `10bfa76` chore(recurrence-model): release 0.1.1 (README quick-start fix) (#13)

---

## Upgrade Notes

Backward-compatible; no migration. Consumers pinning `juniper-recurrence-model>=0.1.0,<0.2.0` pick up 0.1.1 automatically.

```bash
pip install --upgrade juniper-recurrence-model
```

---

## Known Issues

None known at time of release.

---

## What's Next

- Deferred model increments (per the detailed design): trained / nonlinear readout (the point at which torch enters), dense many-to-many readout, and multi-output readout coverage.

---

## Links

- Changelog: `juniper-recurrence-model/CHANGELOG.md`
- Design of record: [JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-14_JUNIPER-RECURRENCE_MODEL-DETAILED-DESIGN.md)
- Roadmap (D-3 / R3): [JUNIPER_RECURRENCE_STATE_ASSESSMENT_AND_ROADMAP_2026-06-17.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-17_JUNIPER-RECURRENCE_STATE-ASSESSMENT-AND-ROADMAP.md)
- **Full Changelog:** https://github.com/pcalnon/juniper-recurrence/compare/juniper-recurrence-model-v0.1.0...juniper-recurrence-model-v0.1.1

---

## Contributors

- Paul Calnon (@pcalnon)
