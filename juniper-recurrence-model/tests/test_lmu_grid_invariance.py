"""Conformance tests for the Δt-native LMU memory (Approach-C).

These pin the load-bearing claims of the model pick:
  - the fixed Legendre matrix A is stable (all eigenvalues have negative real part);
  - the variable-step discretisation reconstructs a delayed sinusoid (the method works);
  - that reconstruction is *grid-invariant* — an irregular sampling grid does not
    materially degrade it (this is the irregular-Δt win Approach-C exists to deliver).

Mirrors the verified reference util/ad-hoc/verify_delta_t_reference_code.py in juniper-ml.

TODO (design doc §9.1a): add a FixedStepLMUMemory negative control asserting a
fixed-Δt discretisation *fails* the e_irr bound that the variable-step memory passes,
converting "fixed-Δt fails" from an analytic assertion into a measurement.
"""

from __future__ import annotations

import numpy as np

from juniper_recurrence_model.units import VariableStepLMUMemory, lmu_matrices


def _err_on(mem, times, theta, omega, rho, w):
    times = np.asarray(times, float)
    dt = np.empty_like(times)
    dt[0] = 0.0
    dt[1:] = np.diff(times)
    u = np.sin(omega * times)
    recon = mem.rollout(u, dt) @ w
    warm = times >= (times[0] + theta)  # score only after one window has filled
    truth = np.sin(omega * (times - rho * theta))
    return float(np.sqrt(np.mean((recon[warm] - truth[warm]) ** 2)))


def test_lmu_matrix_is_stable():
    """All eigenvalues of the fixed LegT matrix A have negative real part."""
    A, _ = lmu_matrices(16)
    max_re = float(np.max(np.linalg.eigvals(A).real))
    assert max_re < 0, f"A(d=16) max eigenvalue real part {max_re} must be < 0"


def test_lmu_grid_invariance():
    """Variable-step reconstruction works (e_reg) and is grid-invariant (e_irr)."""
    d, theta, omega, rho = 16, 1.0, 2.0, 1.0
    mem = VariableStepLMUMemory(d, theta)
    w = mem.decode_weights(rho)

    t_reg = np.linspace(0, 12, 240)
    gaps = np.r_[0.02, np.random.default_rng(0).uniform(0.02, 0.08, 239)]
    t_irr = np.cumsum(gaps)

    e_reg = _err_on(mem, t_reg, theta, omega, rho, w)
    e_irr = _err_on(mem, t_irr, theta, omega, rho, w)

    assert e_reg < 0.05, f"regular-grid RMSE {e_reg} should be < 0.05 (the method works)"
    assert e_irr < 3.0 * e_reg + 0.02, (
        f"irregular-grid RMSE {e_irr} should be < 3*e_reg+0.02 = {3.0 * e_reg + 0.02} "
        "(irregular sampling must not degrade reconstruction)"
    )


def test_rollout_rejects_nonpositive_gap():
    """A non-positive gap for k>=1 is a contract violation."""
    mem = VariableStepLMUMemory(8, 1.0)
    u = np.zeros(5)
    dt = np.array([0.0, 0.1, 0.0, 0.1, 0.1])  # dt[2] == 0 is invalid
    try:
        mem.rollout(u, dt)
    except ValueError:
        return
    raise AssertionError("rollout should raise ValueError on a non-positive gap")
