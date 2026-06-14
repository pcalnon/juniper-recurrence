# juniper-recurrence-model

The model-specific core for the [juniper-recurrence](https://github.com/pcalnon/juniper-recurrence)
application — the selected model **P3-C (LMU + Approach-C)**.

This package currently ships the **Δt-native Legendre Memory Unit (Approach-C)**: a closed-form,
variable-step LMU discretisation that is the only first-principles-clean ("C1") option natively
handling irregularly-sampled time series. The recurrent model implementing the shared
[`juniper-model-core`](https://github.com/pcalnon/juniper-ml) `TrainableModel` interface is added
when that package lands.

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
