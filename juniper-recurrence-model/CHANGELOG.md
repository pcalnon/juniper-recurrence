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
- **`VariableStepLMUMemory.rollout_batch`** — batched, multi-channel eigenbasis ZOH rollout
  reused by the regressor; per-(sample, feature) parity-tested against `rollout`.
- **`juniper_recurrence_model.LMURegressor`** — the fixed-order Δt-native LMU **regressor**
  implementing `juniper-model-core`'s `TrainableModel` (WS-4). Per-feature identity read-in
  (each feature drives its own order-`d` memory through the shared fixed `A`/`B`/θ), a fixed
  LMU-memory rollout, and a **closed-form least-squares readout** (the only trained surface),
  with `target_dt` as an optional readout feature (D-WS4-2). `theta` is **data-driven by default**
  (the median per-window elapsed time) or may be pinned. `predict(X, *, dt=…)` widens the contract
  with optional sequence keywords (`dt`/`target_dt`/`readout_mask`/`seq_lengths`; uniform-`dt`
  fallback when omitted). Regression-only metrics (`mse`/`rmse`/`mae`/`r2`/`loss`); never
  `accuracy` (RK-6). numpy-only (closed-form, no torch).
- **`juniper_recurrence_model.LMUSerializer`** — lossless `.npz` + JSON serializer (readout
  coefficients + hyperparameters; the fixed memory is recomputed from `d`/θ on load).
- **`juniper-model-core>=0.1.0,<0.2.0`** runtime dependency (the shared `TrainableModel` /
  `ModelSerializer` contract + the conformance kit; on PyPI). Still autodiff-free.
- **`juniper_recurrence_model.load_sequence_npz`** + **`SequenceData`** — a lean, numpy-only loader
  for the WS-1 3-D sequence NPZ contract (per-split `X` / `y_reg` / `dt`-or-`t` / `target_dt` /
  `seq_lengths`) that feeds `LMURegressor` via `SequenceData.fit_kwargs()`. **No juniper-data-client
  dependency**; data-client's `validate_npz_contract` stays the app's authoritative fetch-path
  validator. This is the §9.1c consumer wiring — *juniper-recurrence* ingests the irregular-Δt 3-D
  contract (cascor stays 2-D and untouched).
- **End-to-end test** (`tests/test_sequence_data.py`): synthesises an `equities_seq`-shaped 3-D NPZ,
  loads it, and trains/predicts `LMURegressor` end-to-end (plus loader round-trip, `t`→`dt`
  derivation, `y_reg`/`y` fallback, and contract-violation rejections).
- **Model tests** (`tests/test_lmu_model.py`): batched-rollout parity, linear-over-memory
  recovery, determinism, overfit-tiny, bare-`predict(X)`, ridge path, `seq_lengths`/`readout_mask`
  selection, regression-only metrics, topology validity, serializer round-trip, input validation,
  and the **R-Δt-3 shuffle-`dt`** guardrail (predictions degrade when the gaps are reordered —
  proof the model uses timing).
- **Conformance tests** (`tests/test_conformance.py`): subclasses model-core's
  `TrainableModelConformance` against `LMURegressor` over `tiny_regression_3d`, running every
  contract check (isinstance, task_type, fit→`TrainResult`, fit/predict/metrics round-trip,
  predict output shape, metric keys, the RK-6 no-classification-assumptions guard, renderable
  topology, legal event ordering, lossless serialization round-trip).
- **Conformance tests** (`tests/test_lmu_grid_invariance.py`): matrix stability, delayed-sinusoid
  reconstruction (`e_reg < 0.05`), grid-invariance (`e_irr < 3·e_reg + 0.02`), and the **§9.1a
  fixed-Δt negative control** — a `FixedStepLMUMemory` foil (baked at the mean gap) that degrades
  ~2-4× on the irregular grid, proving the per-step Δt adaptation does real work. Numerics match
  the verified reference `util/ad-hoc/verify_delta_t_reference_code.py` in juniper-ml.

### Notes

- **WS-0 ratified 2026-06-14.** The WS-4 model layer is the fixed-order LMU regressor
  (`LMURegressor`). A concurrent-session duplicate (`FixedOrderLMURegressor` / `models/`) was
  **consolidated away** in favour of `LMURegressor` — it carries the ratified D-WS4-2 `target_dt`
  readout feature plus `seq_lengths`/`ridge`/batched rollout; the duplicate's data-driven-`theta`
  default was grafted across. Deferred to later WS-4 increments / workstreams: dense
  many-to-many readout, a trained projection read-in / nonlinear readout (the point at which torch
  enters) and the `juniper-service-core`-backed service/app layer. See the WS-4 build plan `notes/JUNIPER_RECURRENCE_WS4_MODEL_BUILD_PLAN_2026-06-15.md`
  (juniper-ml).
- **§9.1a fixed-Δt negative control: DONE** — ported from the juniper-ml POC into
  `tests/test_lmu_grid_invariance.py` (the degradation *ratio*, not the lenient gate, is the signal).
- **§9.1c reframed + addressed here.** The 3-D/Δt consumer is *juniper-recurrence* (the
  `load_sequence_npz` path), **not cascor** — cascor stays the stateless 2-D feed-forward model.
  The earlier "cascor 3-D ingestion gate" analysis (juniper-ml notes) proved cascor cannot cheaply
  ingest 3-D, which is exactly the justification for a *separate* recurrence model; grafting
  recurrence onto cascor (the deferred §4.2 grown-cascade research) is the only thing that would
  touch cascor, and it is explicitly out of scope here.

[Unreleased]: https://github.com/pcalnon/juniper-recurrence/commits/main
