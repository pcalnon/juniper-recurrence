"""Construct an ``LMURegressor`` from the HTTP / CLI readout selection (DP-3 P2c).

The DP-3 readout spectrum is exposed as a **tagged enum at the edge** (``readout="linear"|"rff"``)
even though the Python model is configured by an immutable readout *spec* (design §6 / Appendix A).
This module is the single translation point — shared by the ``/v1/train`` and ``/v1/crossval`` routes
and the ``train`` CLI — so the readout is wired identically everywhere and the edge never has to know
the spec classes.

* ``readout`` ``None`` / ``"linear"`` → the byte-identical ``LMURegressor(d, theta, ridge=…)`` call
  (the model wraps it in a ``LinearReadoutSpec`` internally), with ``ridge`` falling back to the
  service's ``default_ridge``.
* ``readout="rff"`` → ``LMURegressor(d, theta, readout=RFFReadoutSpec(…))`` with the ridge penalty
  living **inside** the spec — the model rejects passing both a ``ridge=`` and a non-linear
  ``readout=`` (one source of truth), so the RFF ridge is carried by the spec and defaults to
  closed-form GCV (regularisation is mandatory for this rung).
"""

from __future__ import annotations

from typing import Literal

from juniper_recurrence_model import LMURegressor, RFFReadoutSpec

__all__ = ["ReadoutKind", "build_lmu_regressor"]

#: The readout rungs reachable over the HTTP / CLI edge (Rung 2b/MLP arrives with the P3 [torch] extra).
ReadoutKind = Literal["linear", "rff"]

_RFF_DEFAULT_FEATURES = 256
_RFF_DEFAULT_GAMMA: float | Literal["median"] = "median"
#: RFF ridge default: closed-form GCV. Regularisation is mandatory for the high-variance RFF rung.
_RFF_DEFAULT_RIDGE: float | Literal["gcv"] = "gcv"


def build_lmu_regressor(
    *,
    d: int,
    theta: float | None,
    readout: str | None,
    ridge: float | Literal["gcv"] | None,
    rff_features: int | None,
    rff_gamma: float | Literal["median"] | None,
    default_ridge: float | Literal["gcv"],
) -> LMURegressor:
    """Build a fresh ``LMURegressor`` for the resolved readout selection.

    Returns a new instance on every call, so it is safe to use directly (``/v1/train``) or inside a
    per-fold factory closure (``/v1/crossval``): the RFF projection is sampled in ``fit`` from the
    model's ``random_seed`` and the spec itself is immutable, so no fitted state leaks across folds.

    Args:
        d: LMU memory order.
        theta: LMU window length (``None`` ⇒ the model resolves it data-drivenly).
        readout: ``None`` / ``"linear"`` for the linear readout, ``"rff"`` for the RFF readout.
        ridge: explicit ridge penalty (float or ``"gcv"``); ``None`` ⇒ the rung default
            (``default_ridge`` for linear, GCV for RFF).
        rff_features: RFF feature count ``D`` (``None`` ⇒ 256); ignored for the linear readout.
        rff_gamma: RFF bandwidth γ (float or ``"median"``; ``None`` ⇒ ``"median"``); ignored for linear.
        default_ridge: the service's configured linear-readout ridge default.

    Raises:
        ValueError: if ``readout`` is not one of ``None`` / ``"linear"`` / ``"rff"``.
    """
    kind = readout or "linear"
    if kind == "linear":
        resolved_ridge = ridge if ridge is not None else default_ridge
        return LMURegressor(d=d, theta=theta, ridge=resolved_ridge)
    if kind == "rff":
        resolved_ridge = ridge if ridge is not None else _RFF_DEFAULT_RIDGE
        spec = RFFReadoutSpec(
            n_features_out=rff_features if rff_features is not None else _RFF_DEFAULT_FEATURES,
            gamma=rff_gamma if rff_gamma is not None else _RFF_DEFAULT_GAMMA,
            ridge=resolved_ridge,
        )
        return LMURegressor(d=d, theta=theta, readout=spec)
    raise ValueError(f"unknown readout {kind!r}; expected 'linear' or 'rff'")
