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
- **Conformance tests** (`tests/test_lmu_grid_invariance.py`): matrix stability, delayed-sinusoid
  reconstruction (`e_reg < 0.05`), and grid-invariance (`e_irr < 3·e_reg + 0.02`). Numerics match
  the verified reference `util/ad-hoc/verify_delta_t_reference_code.py` in juniper-ml.
- **`juniper_recurrence_model.LMURegressor`** — the fixed-order Δt-native LMU **regressor**
  implementing `juniper-model-core`'s `TrainableModel` (WS-4). Per-feature identity read-in
  (each feature drives its own order-`d` memory through the shared fixed `A`/`B`/θ), a fixed
  LMU-memory rollout, and a **closed-form least-squares readout** (the only trained surface),
  with `target_dt` as an optional readout feature. `predict(X, *, dt=…)` widens the contract
  with optional sequence keywords (uniform-`dt` fallback when omitted). Regression-only metrics
  (`mse`/`rmse`/`mae`/`r2`/`loss`); never `accuracy` (RK-6). numpy-only (closed-form, no torch).
- **`juniper_recurrence_model.LMUSerializer`** — lossless `.npz` + JSON serializer (readout
  coefficients + hyperparameters; the fixed memory is recomputed from `d`/θ on load).
- **`VariableStepLMUMemory.rollout_batch`** — batched, multi-channel eigenbasis ZOH rollout
  reused by the regressor; per-(sample, feature) parity-tested against `rollout`.
- **Model tests** (`tests/test_lmu_model.py`): batched-rollout parity, linear-over-memory
  recovery, determinism, overfit-tiny, bare-`predict(X)`, regression-only metrics, topology
  validity, and serializer round-trip.
- **`juniper-model-core>=0.1.0,<0.2.0`** runtime dependency (the model contract + conformance kit; on PyPI).

### Notes

- **WS-0 ratified 2026-06-14**; this change adds the WS-4 model layer (the fixed-order LMU
  regressor). Deferred to later WS-4 increments / workstreams: dense many-to-many readout, a
  trained projection read-in / nonlinear readout (the point at which torch enters), cascor 3-D
  ingestion (design §9.1c), and the `juniper-service-core`-backed service/app layer. See the
  WS-4 build plan `notes/JUNIPER_RECURRENCE_WS4_MODEL_BUILD_PLAN_2026-06-15.md` (juniper-ml).
- Open follow-up (design doc §9.1a): port the fixed-Δt negative control from the juniper-ml POC
  into the conformance suite — planned for the conformance / Δt-guardrails change (PR-2).

[Unreleased]: https://github.com/pcalnon/juniper-recurrence/commits/main
