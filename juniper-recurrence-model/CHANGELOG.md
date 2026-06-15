# Changelog

All notable changes to the `juniper-recurrence-model` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with [PEP 440](https://peps.python.org/pep-0440/) pre-release identifiers.

## [Unreleased]

### Added

- **Initial package scaffold** for the juniper-recurrence model-specific core, homed as a
  same-named subdirectory of the `juniper-recurrence` repo (mirroring the `juniper-cascor-protocol`
  precedent).
- **`juniper_recurrence_model.units.VariableStepLMUMemory`** + **`lmu_matrices`** — the Δt-native
  Legendre Memory Unit (Approach-C): the fixed closed-form LegT state matrices and their exact
  variable-step zero-order-hold discretisation via a one-time eigendecomposition. C1-clean
  (no ODE solver, no autodiff-through-solver; `A`/`B` never trained). numpy-only.
- **`juniper_recurrence_model.models.FixedOrderLMURegressor`** — the first *real recurrence*
  model wired to the shared `juniper-model-core` `TrainableModel` contract (P3-C / Approach-C).
  The LMU memory is **fixed**; only the linear readout is trained, in **closed form** via
  `numpy.linalg.lstsq` (no BPTT, fully deterministic). Δt-native: accepts per-step gaps `dt` and
  a `readout_mask` as `fit`/`predict` keywords, defaulting to uniform gaps and the final step so
  the bare ABC `predict(X)` still works. Emits the `training_start` / `epoch_end` /
  `training_end` events and reports canonical regression metrics (`mse`, `rmse`, `mae`, `r2`).
- **`juniper_recurrence_model.models.LMURegressorSerializer`** — a `ModelSerializer` strategy that
  round-trips the model losslessly via a single `.npz` (hyperparameters + readout `W`; the fixed
  memory is rebuilt from `(d, theta)` on load).
- **Dependency**: now depends on `juniper-model-core>=0.1.0,<0.2.0` (the shared `TrainableModel` /
  `ModelSerializer` contract and the conformance kit). Still autodiff-free.
- **Conformance tests** (`tests/test_lmu_conformance.py`): subclasses model-core's
  `TrainableModelConformance` against `FixedOrderLMURegressor`, running all ~10 contract checks
  (isinstance, task_type, fit→`TrainResult`, fit/predict/metrics round-trip, predict output shape,
  metric keys, the RK-6 no-classification-assumptions guard, renderable topology, legal event
  ordering, lossless serialization round-trip).
- **Conformance tests** (`tests/test_lmu_grid_invariance.py`): matrix stability, delayed-sinusoid
  reconstruction (`e_reg < 0.05`), and grid-invariance (`e_irr < 3·e_reg + 0.02`). Numerics match
  the verified reference `util/ad-hoc/verify_delta_t_reference_code.py` in juniper-ml.

### Notes

- `FixedOrderLMURegressor` is the WS-4 refactor template: a non-cascor model that passes
  model-core's conformance kit unchanged, proving the shared `TrainableModel` seam.
- The torch-backed read-in/readout (a BPTT variant) and the full 3-D NPZ ingestion path remain
  tracked in the design doc (`notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`) and
  are added in later changes.
- Open follow-up (design doc §9.1a): add a fixed-Δt negative control to the conformance suite.

[Unreleased]: https://github.com/pcalnon/juniper-recurrence/commits/main
