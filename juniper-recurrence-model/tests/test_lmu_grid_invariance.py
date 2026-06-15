"""Conformance tests for the Δt-native LMU memory (Approach-C).

These pin the load-bearing claims of the model pick:
  - the fixed Legendre matrix A is stable (all eigenvalues have negative real part);
  - the variable-step discretisation reconstructs a delayed sinusoid (the method works);
  - that reconstruction is *grid-invariant* — an irregular sampling grid does not
    materially degrade it (this is the irregular-Δt win Approach-C exists to deliver).

Mirrors the verified reference util/ad-hoc/verify_delta_t_reference_code.py in juniper-ml,
including the §9.1a fixed-Δt negative control (the FixedStepLMUMemory foil below).
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_lmu_matrices_rejects_bad_order():
    with pytest.raises(ValueError):
        lmu_matrices(0)


def test_memory_rejects_nonpositive_theta():
    with pytest.raises(ValueError):
        VariableStepLMUMemory(8, 0.0)


def test_rollout_rejects_shape_mismatch():
    mem = VariableStepLMUMemory(8, 1.0)
    with pytest.raises(ValueError):
        mem.rollout(np.zeros(5), np.zeros(4))


def test_decode_weights_length_matches_order():
    mem = VariableStepLMUMemory(12, 1.0)
    w = mem.decode_weights(0.5)
    assert w.shape == (12,)


class _FixedStepLMUMemory(VariableStepLMUMemory):
    """§9.1a negative control: bakes Ā/B̄ ONCE at the grid's mean gap and applies them
    uniformly, ignoring the actual per-step gaps (i.e. it assumes uniform sampling at the
    average rate). On a regular grid this matches the variable-step memory; on an irregular
    grid it mismodels every step, so its grid-invariance breaks. The empirical foil that
    proves Approach-C's per-step Δt adaptation does real work."""

    def rollout(self, u: np.ndarray, dt: np.ndarray) -> np.ndarray:
        dt = np.asarray(dt, dtype=float)
        gaps = dt[1:]
        dt_bar = float(np.mean(gaps)) if gaps.size else 0.0
        a_bar, b_bar = self.step_matrices(dt_bar)  # baked once at the mean gap
        m = np.zeros((self.d, 1))
        out = np.zeros((len(u), self.d))
        for k in range(1, len(u)):
            m = a_bar @ m + b_bar * u[k - 1]  # same matrices every step
            out[k] = m[:, 0]
        return out


def test_fixed_dt_negative_control_degrades_on_irregular_grid():
    """§9.1a: the variable-step memory passes the grid-invariance bound on every (d, ρ), while
    a fixed-Δt control (baked at the mean gap) reconstructs the irregular grid ~2-4× worse. The
    degradation *ratio*, not the lenient gate, is the load-bearing signal (the gate is generous
    at these small errors). Mirrors verify_delta_t_reference_code.py in juniper-ml."""
    theta, omega = 1.0, 2.0
    rng = np.random.default_rng(0)
    ratios = []
    variable_passes_all = True
    for d in (16, 24):
        variable = VariableStepLMUMemory(d, theta)
        fixed = _FixedStepLMUMemory(d, theta)
        for rho in (0.5, 0.8, 1.0):
            w = variable.decode_weights(rho)
            t_reg = np.linspace(0, 12, 240)
            gaps = np.r_[0.02, rng.uniform(0.02, 0.08, 239)]  # small gaps << theta: the sharp discriminator
            t_irr = np.cumsum(gaps)
            e_var_reg = _err_on(variable, t_reg, theta, omega, rho, w)
            e_var_irr = _err_on(variable, t_irr, theta, omega, rho, w)
            e_fixed_irr = _err_on(fixed, t_irr, theta, omega, rho, w)
            variable_passes_all &= e_var_irr < 3.0 * e_var_reg + 0.02
            ratios.append(e_fixed_irr / max(e_var_irr, 1e-9))
    mean_ratio = float(np.mean(ratios))
    assert variable_passes_all, "variable-step memory must pass the grid-invariance bound on every (d, rho)"
    assert mean_ratio >= 2.0, f"fixed-Δt control should degrade ~2-4x on the irregular grid; got {mean_ratio:.1f}x"
