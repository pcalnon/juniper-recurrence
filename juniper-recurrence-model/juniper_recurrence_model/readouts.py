"""Readout spectrum for the LMU regressor (DP-3).

The LMU regressor's only trained surface is its *readout* — the map from the per-window LMU
memory state ``M ∈ ℝ^{F·d}`` (plus the optional ``target_dt`` horizon and a bias) to the
regression target. DP-3 (design-of-record
``notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md`` in juniper-ml) makes that
readout a *configurable spectrum*:

* **Rung 0** — plain min-norm least-squares (``ridge=0.0``; the back-compat default).
* **Rung 1** — regularised linear (``ridge>0.0``) + closed-form **GCV** penalty selection
  (``ridge="gcv"``). *This module (P1).*
* **Rung 2a** — a numpy nonlinear readout (random Fourier features + ridge). *Added in P2.*
* **Rung 2b** — an optional torch MLP readout behind a ``[torch]`` extra. *Added in P3.*

Design boundary (D-WS4-2). A readout receives the memory block ``M`` only; :class:`LMURegressor`
owns ``target_dt`` — a *linear* side-channel appended **after** any nonlinearity — and the bias
column. So a readout's design matrix is ``[ transform(M) | extra | 1 ]`` where ``transform`` is the
readout's (possibly nonlinear) feature map and ``extra`` is the caller-supplied linear side-channel
(the ``target_dt`` column, or an ``(n, 0)`` array when the model was not fit with a horizon).

Spec vs. live readout. A readout is *configured* by an **immutable spec** (a frozen dataclass) and
*materialised* into a fresh fitted instance inside :meth:`LMURegressor.fit`. A spec carries no
fitted state, so the same spec object shared across cross-validation folds can never leak one fold's
fitted weights into another (the cross-fold trap that a shared *live* readout would suffer). Any
data-independent randomness (e.g. the RFF projection in P2) is sampled inside ``fit`` from the
model's ``random_seed`` — fixed across folds, never from a module-global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

import numpy as np

__all__ = [
    "RidgeParam",
    "Readout",
    "ReadoutSpec",
    "LinearReadout",
    "LinearReadoutSpec",
    "READOUT_REGISTRY",
    "build_readout_from_state",
]

#: A ridge penalty: a non-negative float, or the literal ``"gcv"`` requesting closed-form
#: generalised-cross-validation selection of the penalty at ``fit`` time.
RidgeParam = float | Literal["gcv"]

#: Log-spaced grid of candidate ridge penalties for GCV selection (Rung 1). One SVD of the
#: (centred) design matrix makes each grid evaluation O(rank), so a fine grid is cheap.
_GCV_GRID: np.ndarray = np.logspace(-6.0, 3.0, 60)


class Readout(Protocol):
    """The trained map from the LMU memory block ``M`` to the target (a DP-3 readout rung).

    Implementations transform ``M`` through their (possibly nonlinear) feature map, append the
    caller's linear side-channel ``extra`` and a bias column, and fit / apply the trained weights.
    They own every fitted array and persist it losslessly via :meth:`save_state` /
    ``from_state`` (the bit-exact serialization contract).
    """

    kind: ClassVar[str]

    def fit(self, M: np.ndarray, extra: np.ndarray, y: np.ndarray, *, random_seed: int | None = None) -> None:
        """Fit the readout to memory block ``M`` (n, F·d), linear side-channel ``extra`` (n, k), target ``y`` (n, out)."""
        ...

    def predict(self, M: np.ndarray, extra: np.ndarray) -> np.ndarray:
        """Predict ``(n, out)`` for memory block ``M`` and linear side-channel ``extra``."""
        ...

    @property
    def is_fitted(self) -> bool:
        """``True`` once :meth:`fit` has populated the trained weights."""
        ...

    @property
    def coef(self) -> np.ndarray | None:
        """The linear readout coefficients in ``[transform(M) | extra | 1]`` layout, or ``None`` for a nonlinear readout."""
        ...

    def save_state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Return ``(arrays, descriptor)``: the fitted numpy arrays + a JSON-safe descriptor with a ``kind`` tag."""
        ...


class ReadoutSpec(Protocol):
    """Immutable configuration that materialises a fresh :class:`Readout` per fit (cross-fold-safe)."""

    kind: ClassVar[str]

    def make(self) -> Readout:
        """Build a fresh, unfitted live readout from this spec."""
        ...


def _assemble_design(block: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Append the linear side-channel ``extra`` and a bias column to a (transformed) feature block."""
    n = block.shape[0]
    return np.concatenate([block, extra, np.ones((n, 1))], axis=1)


def _ridge_solve(design: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge normal-equations solve for a ``[features | 1]`` design (the bias column is never penalised)."""
    gram = design.T @ design
    penalty = ridge * np.eye(gram.shape[0])
    penalty[-1, -1] = 0.0  # never regularise the bias column
    return np.linalg.solve(gram + penalty, design.T @ y)


def _gcv_select(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form GCV ridge selection over :data:`_GCV_GRID` (Rung 1).

    Centres ``features`` / ``y`` (so the intercept is unpenalised and recovered separately), takes
    **one** thin SVD, then evaluates ``GCV(λ) = n·RSS(λ) / (n − tr H(λ))²`` for every grid λ in
    O(rank) using the projected coefficients ``g = Uᵀ yc`` — no held-out split and no inner-CV
    refit (design §4 Rung 1). ``tr H(λ) = Σ sᵢ²/(sᵢ²+λ)``; the unpenalised intercept contributes
    one further degree of freedom, so the effective trace is ``1 + tr H``. Returns the coefficient
    vector in the bias-augmented ``[features | 1]`` layout and the selected λ.
    """
    n = features.shape[0]
    feat_mean = features.mean(axis=0)
    y_mean = y.mean(axis=0)
    fc = features - feat_mean
    yc = y - y_mean
    u, s, vt = np.linalg.svd(fc, full_matrices=False)  # fc = u @ diag(s) @ vt
    g = u.T @ yc  # (rank, out): projection of yc onto the left singular vectors
    s2 = s**2
    yc_energy = float(np.sum(yc**2))  # ||yc||_F^2
    resid_perp = yc_energy - float(np.sum(g**2))  # energy orthogonal to range(U) (>= 0); λ-independent
    best_lam = float(_GCV_GRID[0])
    best_gcv = np.inf
    for lam in _GCV_GRID:
        tr_h = 1.0 + float(np.sum(s2 / (s2 + lam)))  # +1 for the unpenalised intercept
        shrink = (lam / (s2 + lam)) ** 2  # (1 - f_i)^2 per singular component
        rss = float(np.sum(shrink[:, None] * (g**2))) + resid_perp
        denom = (n - tr_h) ** 2
        gcv = (n * rss / denom) if denom > 0 else np.inf
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)
    filt = s / (s2 + best_lam)  # s_i/(s_i^2+λ)
    coef_feat = vt.T @ (filt[:, None] * g)  # (p, out)
    coef_bias = y_mean - feat_mean @ coef_feat  # (out,)
    return np.concatenate([coef_feat, coef_bias[None, :]], axis=0), best_lam


class LinearReadout:
    """Rung 0/1 — a (optionally ridge-regularised) linear readout over ``[ M | extra | 1 ]``.

    ``ridge=0.0`` is the back-compat min-norm ``lstsq`` solve; ``ridge>0.0`` is the regularised
    normal-equation solve (the bias column is never penalised); ``ridge="gcv"`` selects the penalty
    by closed-form generalised cross-validation at ``fit`` and writes the selected λ back to
    :attr:`ridge` (so it is persisted for retraining fidelity).
    """

    kind: ClassVar[str] = "linear"

    def __init__(self, ridge: RidgeParam = 0.0) -> None:
        self.ridge: RidgeParam = ridge
        self._coef: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._coef is not None

    @property
    def coef(self) -> np.ndarray | None:
        return self._coef

    def _design(self, M: np.ndarray, extra: np.ndarray) -> np.ndarray:
        return _assemble_design(M, extra)

    def fit(self, M: np.ndarray, extra: np.ndarray, y: np.ndarray, *, random_seed: int | None = None) -> None:
        if isinstance(self.ridge, str):
            if self.ridge != "gcv":
                raise ValueError(f"unknown ridge mode {self.ridge!r}; expected a float or 'gcv'")
            features = np.concatenate([M, extra], axis=1)  # GCV centres internally; bias recovered separately
            self._coef, self.ridge = _gcv_select(features, y)
            return
        design = self._design(M, extra)
        if self.ridge > 0.0:
            self._coef = _ridge_solve(design, y, float(self.ridge))
        else:
            self._coef, *_ = np.linalg.lstsq(design, y, rcond=None)

    def predict(self, M: np.ndarray, extra: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("readout is not fitted")
        return self._design(M, extra) @ self._coef

    def save_state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._coef is None:
            raise RuntimeError("cannot serialize an unfitted readout")
        return {"coef": self._coef}, {"kind": self.kind, "ridge": float(self.ridge)}

    @classmethod
    def from_state(cls, arrays: dict[str, np.ndarray], descriptor: dict[str, Any]) -> LinearReadout:
        readout = cls(ridge=float(descriptor.get("ridge", 0.0)))
        readout._coef = arrays["coef"]
        return readout


@dataclass(frozen=True)
class LinearReadoutSpec:
    """Immutable spec for :class:`LinearReadout` (Rung 0/1)."""

    ridge: RidgeParam = 0.0
    kind: ClassVar[str] = "linear"

    def make(self) -> LinearReadout:
        return LinearReadout(ridge=self.ridge)


#: Maps a persisted readout ``kind`` tag to its live class (each exposes a ``from_state`` classmethod).
#: P2 registers ``"rff"``; P3 registers ``"mlp"`` (lazily, behind the ``[torch]`` extra).
READOUT_REGISTRY: dict[str, Any] = {LinearReadout.kind: LinearReadout}


def build_readout_from_state(arrays: dict[str, np.ndarray], descriptor: dict[str, Any]) -> Readout:
    """Reconstruct a fitted readout from persisted arrays + its JSON descriptor (the ``kind`` tag)."""
    kind = descriptor.get("kind")
    try:
        readout_cls = READOUT_REGISTRY[kind]
    except KeyError:
        raise ValueError(f"unknown readout kind {kind!r}; registered kinds: {sorted(READOUT_REGISTRY)}") from None
    return readout_cls.from_state(arrays, descriptor)
