# juniper-recurrence-model

The model-specific core for the [juniper-recurrence](https://github.com/pcalnon/juniper-recurrence)
application — the selected model **P3-C (LMU + Approach-C)**.

This package ships the **Δt-native Legendre Memory Unit (Approach-C)** — a closed-form,
variable-step LMU discretisation that is the only first-principles-clean ("C1") option natively
handling irregularly-sampled time series — **and** `FixedOrderLMURegressor`, the recurrent model
implementing the shared [`juniper-model-core`](https://github.com/pcalnon/juniper-ml)
`TrainableModel` interface (now that that package has landed). The regressor keeps the LMU memory
**fixed** and trains only a linear readout in **closed form** (least squares — no BPTT, fully
deterministic); it passes model-core's conformance kit unchanged, making it the WS-4 refactor
template (a non-cascor model on the shared model seam).

Design of record (in juniper-ml):
[`notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md).

## Why Approach-C

An LMU's linear memory obeys `theta * m'(t) = A·m(t) + B·u(t)` with **fixed, closed-form** matrices.
Because the system is linear, its *exact* discretisation is a matrix exponential — **no ODE solver,
no autodiff-through-solver**. For irregular sampling, the discrete update is simply evaluated at the
real per-step gap `dt`: the dataset's `dt` channel *is* the discretisation step. `A`/`B` are never
trained; only the read-in/readout are. That is the entire C1-clean, irregular-Δt-native story.

## Install

```bash
pip install juniper-recurrence-model          # once published
pip install -e ".[test]"                       # local development
```

numpy-only at the core (the memory is a fixed linear recurrence requiring no autodiff).

## Quick start

```python
import numpy as np
from juniper_recurrence_model import VariableStepLMUMemory

mem = VariableStepLMUMemory(d=16, theta=1.0)   # order 16, window 1.0 (same unit as dt)

# Irregularly-sampled input: u driven on a non-uniform time grid
t = np.cumsum(np.r_[0.0, np.random.default_rng(0).uniform(0.02, 0.08, 239)])
dt = np.empty_like(t); dt[0] = 0.0; dt[1:] = np.diff(t)
u = np.sin(2.0 * t)

m = mem.rollout(u, dt)                          # (240, 16) memory trajectory
w = mem.decode_weights(rho=1.0)                 # read the input one full window ago
reconstruction = m @ w
```

## Trainable model (`FixedOrderLMURegressor`)

The package also exposes `FixedOrderLMURegressor`, a `juniper-model-core` `TrainableModel`. The
LMU memory is fixed; only a linear readout is fit, in closed form (least squares — no BPTT, fully
deterministic). It is Δt-native: pass per-step gaps `dt` (`(n, T)`) and an optional `readout_mask`
to `fit` / `predict`; both default to uniform gaps and the final step, so the bare ABC
`predict(X)` works too. It reports canonical regression metrics (`mse`, `rmse`, `mae`, `r2`).

```python
import numpy as np
from juniper_recurrence_model import FixedOrderLMURegressor, LMURegressorSerializer

n, T, F = 48, 6, 3
X = np.random.default_rng(0).normal(size=(n, T, F))
y = X.reshape(n, -1) @ np.random.default_rng(1).normal(size=(T * F, 1))
dt = np.zeros((n, T)); dt[:, 1:] = np.random.default_rng(2).integers(1, 4, size=(n, T - 1))

model = FixedOrderLMURegressor(d=6)             # theta resolved data-driven from dt at fit time
result = model.fit(X, y, dt=dt)                 # closed-form readout solve
preds = model.predict(X, dt=dt)                 # (n, 1)
print(result.final_metrics["r2"], model.describe_topology()["model_type"])

LMURegressorSerializer().save(model, "/tmp/lmu")   # writes /tmp/lmu.npz (lossless round-trip)
```

`FixedOrderLMURegressor` passes model-core's conformance kit unchanged
(`tests/test_lmu_conformance.py`), proving the WS-4 refactor template.

## Verified behaviour

| Check | Result |
|---|---|
| `A` (d=16) max eigenvalue real part | **−6.49** (< 0 → stable) |
| Reconstruction RMSE `e_reg` (regular grid) | **≈ 0.035** (< 0.05) |
| Grid-invariance `e_irr` (irregular grid) | **≈ 0.039–0.043** (≈1.15× `e_reg`; < 3·`e_reg` + 0.02) |

Pinned by `tests/test_lmu_grid_invariance.py`. Numerics match the reference
`util/ad-hoc/verify_delta_t_reference_code.py` in juniper-ml.

## Numerical guardrails

- Keep `d ≲ 64` — the eigenvector matrix of `A` becomes ill-conditioned for large `d`
  (Padé scaling-and-squaring is the documented fallback for larger orders).
- Stability is automatic for `dt > 0` (`Re(λ) < 0 ⇒ |e^z| < 1`).
- `dt` may be quantised (e.g. integer calendar-day gaps) and `Abar`/`Bbar` cached per bucket.

## Versioning

PEP 440 + [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Consumers should pin
`juniper-recurrence-model>=A.B,<A+1`. See [`CHANGELOG.md`](./CHANGELOG.md).

## License

MIT — see [LICENSE](https://github.com/pcalnon/juniper-recurrence/blob/main/LICENSE).
