# Changelog

All notable changes to the `juniper-recurrence-model` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with [PEP 440](https://peps.python.org/pep-0440/) pre-release identifiers.

## [Unreleased]

### Changed

- **Blocking per-file coverage gate wired into both CI lanes** (per-file coverage rollout C-5;
  juniper-ml `notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md`).
  `ci-recurrence-model.yml` now runs `juniper-coverage-gap-map --enforce`
  (`juniper-ci-tools>=0.6.0,<0.7.0`) in **both** the base `test` lane and the `test-torch` lane,
  failing the build if any source file drops below 90% statement coverage or any packaged
  sub-module below 95% pooled (statement-weighted) coverage. The two lanes gate the **union** of
  source files through their existing complementary coverage configs: the base lane's
  `[tool.coverage.run].omit` drops `_readout_mlp.py`, while `.coveragerc.torch`'s `[report] include`
  scopes the torch lane to exactly that module — so every packaged file is gated in exactly one
  lane, with none falling through the crack between lanes. No coverage lift was required — measured
  statement coverage is already 95.90–100% per file (base sub-module `juniper_recurrence_model`
  pooled 97.68%, `units` pooled 98.85%; `_readout_mlp.py` 97.42%; union `juniper_recurrence_model`
  pooled 97.61%). CI / packaging only — no runtime change, no version bump.

### Fixed

- **Input validation hardening (audit MODEL-01 / MODEL-04).** Non-finite (`NaN`/`Inf`) values in
  the `dt` timing channel — and in the feature drive — are now rejected with a clear `ValueError`
  in `VariableStepLMUMemory.rollout` / `rollout_batch` and in `sequence_data_from_arrays`
  (previously a `NaN` gap slipped past the bare `dt < 0` guard and silently produced `NaN`
  predictions / an opaque LAPACK error). `RFFReadout` now rejects `n_features_out < 1` instead of
  failing with a later `ZeroDivisionError`.

## [0.1.5] - 2026-06-23

DP-3 readout-spectrum, **phase P3** — the optional **torch MLP readout (Rung 2b)** plus the
`LMURegressor` validation plumbing that feeds it. Additive + backward compatible (the default readout
and the closed-form rungs are unchanged), hence a patch release. Ratified GO 2026-06-23
(`notes/JUNIPER_DECISIONS_RATIFIED_2026-06-23.md` D5, in juniper-ml) as capability insurance for future
complex / hybrid datasets — the current dataset catalog does not itself require it.

### Added

- **`MLPReadout` / `MLPReadoutSpec` (Rung 2b)** — an optional torch MLP readout behind a new `[torch]`
  extra (`torch>=2.10.0`, the ecosystem CVE-2025-3001 security floor; ships cp314 wheels). Architecture:
  `standardize(M) → GELU trunk (h→h) → linear head over [trunk | extra]`, with internal target
  standardization. Training is **CPU-only, single-threaded, float32, `use_deterministic_algorithms(True)`**,
  so an in-process save→load→predict round-trip is bit-exact within a machine (no cross-machine claim).
  State persists as **named numpy arrays** (never `torch.save`; the serializer loads with
  `allow_pickle=False`). `torch` is imported **lazily**, so the base package stays numpy-only and
  torch-free — the module is exercised by a separate optional `test-torch` CI job and omitted from the
  base coverage gate. New public exports: `MLPReadout`, `MLPReadoutSpec`. Use via
  `LMURegressor(readout=MLPReadoutSpec(…))`.
- **`Readout.fit` validation arguments** — the `Readout` protocol's `fit` gained optional
  `M_val` / `extra_val` / `y_val` keywords. The closed-form linear and RFF rungs accept and ignore them;
  the MLP rung uses them for **validation-driven early stopping** (relative `min_delta`, `patience`),
  keeping the best-validation weights.

### Changed

- **`LMURegressor.fit` plumbs `X_val` / `y_val` through to the readout.** When validation data is supplied
  (directly, or by the crossval / conformance harness), `fit` builds a held-out feature block and passes
  it to `readout.fit`. Optional `*_val` timing kwargs (`dt_val`, `target_dt_val`, `readout_mask_val`,
  `seq_lengths_val`) make the val block Δt-faithful; absent (the conformance / crossval convention) it
  falls back to the uniform-grid construction — adequate as an early-stop signal.
- **`TrainResult` reports the readout's true training diagnostics.** `n_epochs` reflects the readout's
  trained-epoch count (`getattr(readout, "n_epochs_", 1)`) and `stopped_reason` reflects how it stopped
  (`"early_stopping"` / `"max_epochs"` for the MLP). Closed-form readouts expose neither, so they read as
  the canonical single-solve `1` / `"converged"` — preserving the crossval `n_epochs == 1` invariant.
  When validation data is present, the `epoch_end` / `training_end` events carry a `val_metrics` payload.
- **Serializer registry now includes `"mlp"`** (lazily registered, so the base import never pulls in
  torch). The MLP persists its layer weights/biases + standardization stats as float32 `readout__*`
  arrays + a `meta["readout"]` descriptor; bit-exact lossless serialization is gated by an `MLPReadout`
  conformance subclass. `model._coef` is `None` for the (nonlinear) MLP readout, as for any nonlinear rung.

## [0.1.4] - 2026-06-21

DP-3 readout-spectrum, **phase P2a** — the numpy **nonlinear** readout (Rung 2a). Additive +
backward-compatible (the default readout is unchanged), hence a patch release.

### Added

- **`RFFReadout` / `RFFReadoutSpec` (Rung 2a)** — a numpy nonlinear readout:
  `standardize(M) → random Fourier features → ridge`. `φ(M) = √(2/D)·cos(standardize(M)·W + b)`
  with `W ~ 𝒩(0, γ²I)`, `b ~ U[0, 2π)` sampled once at `fit` from the model's `random_seed`
  (data-independent, fixed across folds — cross-fold safe via the immutable spec). The design matrix
  is `[ φ(standardize(M)) | target_dt | 1 ]`: the RFF map applies to the memory block only;
  `target_dt` and the bias stay linear (D-WS4-2). Use via `LMURegressor(readout=RFFReadoutSpec(…))`.
  New public exports: `RFFReadout`, `RFFReadoutSpec`.
- **Bandwidth selection** — `γ` via the median heuristic on standardized `M` (`gamma="median"`,
  default; ridge/GCV cannot select `γ`), or a fixed float. The readout penalty is **GCV-selected by
  default** (`ridge="gcv"`); ridge is mandatory for this rung (`γ`/`D` are high-variance).
- **Mandatory per-column standardization of `M`** (train-fold-only; zero-variance columns guarded to
  std=1 so predictions stay finite) — keeps the isotropic `W` from being dominated by the
  high-energy low-order Legendre columns (≈25× RMS spread).

### Changed

- **Serializer registry now includes `"rff"`.** `RFFReadout` persists `W`, `b`, the standardization
  stats, and the solved coefficients as float64 `readout__*` arrays + a `meta["readout"]` descriptor
  (`kind`, `gamma`, `ridge`, `n_features_out`). Bit-exact lossless serialization for the
  `cos`-of-matmul path is **gated by an RFF conformance subclass** (in-process; no cross-machine
  claim). `D` is capped to the fold size (`p/n` guard; ridge handles the remainder). `model._coef` is
  `None` for the (nonlinear) RFF readout, as for any non-linear rung.

## [0.1.3] - 2026-06-20

DP-3 readout-spectrum, **phase P1** (design-of-record
`notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md` in juniper-ml). Makes the LMU
regressor's only trained surface — its *readout* — a configurable spectrum, and ships the cheap
data-ranked lever (a GCV-selected regularised linear readout). **Fully backward compatible**: an
additive, optional API — hence a patch release (stays inside every existing `<0.2.0` consumer pin).

### Added

- **Readout-spec API** (`juniper_recurrence_model.readouts`): the `Readout` / `ReadoutSpec`
  protocols, `LinearReadout`, and the immutable `LinearReadoutSpec(ridge=…)`. `LMURegressor` now
  accepts an explicit `readout=<spec>` (a *spec*, not a live object — so a spec shared across
  cross-validation folds can never leak one fold's fitted weights into another). New public exports:
  `Readout`, `ReadoutSpec`, `LinearReadout`, `LinearReadoutSpec`, `RidgeParam`.
- **GCV ridge selection** (`ridge="gcv"`): closed-form generalised-cross-validation choice of the
  readout L2 penalty at `fit` — one SVD of the (centred) design matrix + a 1-D log-grid search, no
  held-out split and no inner-CV refit. The selected λ is written back to `model.ridge` and persisted
  in the serialized metadata (retraining fidelity). Resolves juniper-recurrence#28.

### Changed

- **`LMURegressor` delegates its readout** to the spec (the fixed Δt LMU memory rollout is unchanged).
  `LMURegressor(d, theta, ridge=…)` remains **byte-identical** to the pre-DP-3 model (the default
  `LinearReadoutSpec(ridge=0.0)`); `ridge` widened to `float | Literal["gcv"]`. The readout receives
  the memory block `M` only; `LMURegressor` keeps owning `target_dt` (a linear side-channel appended
  after any nonlinearity) and the bias column (preserves D-WS4-2). `model._coef` is now a read-only
  forwarding property to the linear readout's coefficients (`None` before fit / for nonlinear readouts).
- **Serializer schema 2.** `LMUSerializer` now persists the readout's own state as namespaced
  `readout__*` arrays + a nested `meta["readout"]` descriptor (with a `kind` tag), reconstructed via a
  readout registry on load. **Pre-DP-3 `.npz` files still load** (a top-level `coef` + no
  `meta["readout"]` falls back to a linear readout from `meta["ridge"]`). Topology gains a nested
  `meta["readout"]={"kind": …}`; the LMU envelope keys (esp. `meta["d"]` = memory order) stay frozen.

## [0.1.2] - 2026-06-17

### Changed

- **Adopt the `juniper-model-core` 0.2.0 cross-validation layer.** Widened the `juniper-model-core`
  dependency ceiling to `<0.3.0` (admits 0.2.0) and added `juniper-model-core[crossval]>=0.2.0` to
  the `[test]` extra. The model's runtime surface is unchanged.

### Added

- **Cross-validation second-implementer proof** (`tests/test_crossval.py`). Drives `LMURegressor`
  through `juniper_model_core.crossval.cross_validate` over a 3-D Δt fixture with
  `aux={dt, target_dt, seq_lengths}`, confirming the generic fold executor slices the auxiliary
  arrays per fold and engages the Δt path on a real model (with a shuffled-`dt` guardrail and a
  determinism check).

## [0.1.1] - 2026-06-17

### Fixed

- **README API drift** — the quick-start referenced `FixedOrderLMURegressor`, `LMURegressorSerializer`,
  and `tests/test_lmu_conformance.py`, none of which exist; the public API is `LMURegressor` /
  `LMUSerializer` (`tests/test_conformance.py`). The documented import and both runnable examples now
  execute against the real 0.1.0 surface.

## [0.1.0] - 2026-06-15

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

[Unreleased]: https://github.com/pcalnon/juniper-recurrence/compare/juniper-recurrence-model-v0.1.0...main
[0.1.0]: https://github.com/pcalnon/juniper-recurrence/releases/tag/juniper-recurrence-model-v0.1.0
