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

### Notes

- Pre-implementation scaffold; WS-0 (design ratification) is not yet ratified. The recurrent model
  implementing `juniper-model-core`'s `TrainableModel` interface, the torch-backed read-in/readout,
  and the 3-D NPZ ingestion path are tracked in the design doc
  (`notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`) and are added in later changes.
- Open follow-up (design doc §9.1a): add a fixed-Δt negative control to the conformance suite.

[Unreleased]: https://github.com/pcalnon/juniper-recurrence/commits/main
