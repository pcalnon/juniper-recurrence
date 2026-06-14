"""Δt-native Legendre Memory Unit (Approach-C).

A Legendre Memory Unit (LMU; Voelker, Kajić & Eliasmith 2019, NeurIPS) maintains a
continuous-time representation of an input's recent history by orthogonalising it onto
the Legendre polynomial basis over a sliding window of length ``theta``. Its linear
memory state obeys

    theta * m'(t) = A @ m(t) + B * u(t)

with **fixed, closed-form** matrices ``A`` (``d x d``) and ``B`` (``d x 1``) — the
HiPPO-LegT operator (Gu et al. 2020). Because the system is *linear*, its exact
discretisation is a matrix exponential — no numerical ODE solver is required.

**Approach C — the irregular-Δt win.** Standard LMU implementations bake the discrete
``Abar``/``Bbar`` as constants for one fixed step. The only change needed for
irregularly-sampled data is to evaluate them at the *actual* per-step gap ``dt_k`` — i.e.
the dataset's ``dt`` channel *is* the discretisation step (the same role the per-step
``Delta`` parameter plays in S4/Mamba). This is done in closed form via a one-time
eigendecomposition of the fixed ``A``, so each step costs only ``d`` scalar exponentials.

**C1-clean (first-principles).** No ODE solver, no autodiff-through-solver — only scalar
exponentials of the eigenvalues of a FIXED, closed-form matrix. ``A`` and ``B`` are not
learned and not data-dependent; only the read-in (features -> drive ``u``) and the readout
(memory -> output) are trained, and they live outside this module.

Verified reference: the numerics here match ``util/ad-hoc/verify_delta_t_reference_code.py``
in juniper-ml (numpy 2.4.4 / Python 3.13): ``A`` (d=16) max eigenvalue real part = -6.49
(stable); delayed-sinusoid reconstruction ``e_reg`` ≈ 0.035 and grid-invariant
``e_irr`` ≈ 0.039–0.043 (≈1.15×). See
``notes/JUNIPER_RECURSE_DELTA_T_HANDLING_2026-06-05.md`` §8 and
``notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`` Part 3.

References
---------
- Voelker, Kajić & Eliasmith (2019). Legendre Memory Units. NeurIPS.
- Gu, Dao, Ermon, Rudra & Ré (2020). HiPPO. NeurIPS; arXiv:2008.07669.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import Legendre

__all__ = ["lmu_matrices", "VariableStepLMUMemory"]

# Below this magnitude an eigenvalue is treated as zero for the removable
# singularity in (exp(z) - 1) / lambda. LegT's A has no zero eigenvalue, but the
# guard keeps the code correct for hygiene / other bases.
_LAMBDA_ZERO_TOL = 1e-12


def lmu_matrices(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed, closed-form Legendre (HiPPO-LegT) state matrices.

    Parameters
    ----------
    d:
        Memory order (number of Legendre coefficients). Practical range ~4..64;
        the eigenvector matrix of ``A`` becomes ill-conditioned for large ``d``.

    Returns
    -------
    (A, B):
        ``A`` of shape ``(d, d)`` and ``B`` of shape ``(d, 1)``. These depend only
        on ``d`` — they are not learned and not data-dependent.
    """
    if d < 1:
        raise ValueError(f"order d must be >= 1, got {d}")
    A = np.zeros((d, d))
    B = np.zeros((d, 1))
    for i in range(d):
        B[i, 0] = (2 * i + 1) * ((-1) ** i)
        for j in range(d):
            A[i, j] = (2 * i + 1) * (-1.0 if i < j else (-1.0) ** (i - j + 1))
    return A, B


class VariableStepLMUMemory:
    """Irregular-Δt-native LMU memory (Approach-C).

    The linear LMU memory, exactly discretised at an arbitrary per-step real gap
    ``dt`` via the zero-order-hold update

        m_{k+1} = Abar(dt) @ m_k + Bbar(dt) * u_k

    where ``Abar``/``Bbar`` are computed from a one-time eigendecomposition of the
    fixed matrix ``A``::

        z_i      = lambda_i * dt / theta
        Abar(dt) = V @ diag(exp(z_i))            @ Vinv
        Bbar(dt) = V @ diag(expm1(z_i) / lam_i)  @ (Vinv @ B)

    ``expm1`` (not ``exp(z) - 1``) avoids catastrophic cancellation at small ``z``.
    Stability is automatic for ``dt > 0`` because every eigenvalue has negative real
    part, so ``|exp(z_i)| < 1`` and ``Abar`` is a contraction.

    Parameters
    ----------
    d:
        Memory order (see :func:`lmu_matrices`).
    theta:
        Memory window length, in the *same real-time units* as ``dt`` (e.g. calendar
        days for the equities use-case).
    """

    def __init__(self, d: int, theta: float) -> None:
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}")
        self.d = int(d)
        self.theta = float(theta)
        A, B = lmu_matrices(self.d)
        lam, V = np.linalg.eig(A)
        # Precomputed once; depend only on (d, theta).
        self.lam = lam
        self.V = V
        self.Vinv = np.linalg.inv(V)
        self.VinvB = self.Vinv @ B

    def step_matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Abar, Bbar)`` for a single real gap ``dt`` (both real-valued)."""
        z = self.lam * (dt / self.theta)
        Abar = (self.V * np.exp(z)) @ self.Vinv
        with np.errstate(divide="ignore", invalid="ignore"):
            fac = np.expm1(z) / self.lam
        fac = np.where(np.abs(self.lam) < _LAMBDA_ZERO_TOL, dt / self.theta, fac)
        Bbar = (self.V * fac) @ self.VinvB
        return Abar.real, Bbar.real

    def rollout(self, u: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """Roll the memory over a 1-D input ``u`` with per-step gaps ``dt``.

        Zero-order-hold convention: ``u[k-1]`` is held constant across the interval
        ``(t[k-1], t[k]]`` of length ``dt[k]``. ``dt[0]`` is unused (empty window);
        the returned ``out[0]`` is the zero initial state.

        Parameters
        ----------
        u:
            Scalar drive per step, shape ``(n,)``.
        dt:
            Per-step elapsed real time, shape ``(n,)``; ``dt[k] > 0`` for ``k >= 1``.

        Returns
        -------
        np.ndarray
            Memory trajectory of shape ``(n, d)``.
        """
        u = np.asarray(u, dtype=float)
        dt = np.asarray(dt, dtype=float)
        if u.ndim != 1 or dt.ndim != 1 or u.shape[0] != dt.shape[0]:
            raise ValueError(f"u and dt must be 1-D and equal length; got {u.shape}, {dt.shape}")
        n = u.shape[0]
        m = np.zeros((self.d, 1))
        out = np.zeros((n, self.d))
        for k in range(1, n):
            if dt[k] <= 0:
                raise ValueError(f"dt[{k}]={dt[k]} must be > 0 for k >= 1")
            Abar, Bbar = self.step_matrices(float(dt[k]))
            m = Abar @ m + Bbar * u[k - 1]
            out[k] = m[:, 0]
        return out

    def decode_weights(self, rho: float) -> np.ndarray:
        """Readout weights to reconstruct the input at delay ``rho * theta`` into the past.

        Uses shifted-Legendre evaluation: ``w[i] = P_i(2*rho - 1)`` for ``i in 0..d-1``,
        with ``rho in [0, 1]`` (0 = now, 1 = the full window ago).
        """
        x = 2.0 * float(rho) - 1.0
        return np.array([Legendre.basis(i)(x) for i in range(self.d)])
