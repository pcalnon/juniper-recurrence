"""Construct an ``LMURegressor`` from the HTTP / CLI readout selection (DP-3 P2c).

The DP-3 readout spectrum is exposed as a **tagged enum at the edge** (``readout="linear"|"rff"|"mlp"``)
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
* ``readout="mlp"`` → ``LMURegressor(d, theta, readout=MLPReadoutSpec(…))`` (Rung 2b, DP-3 P3).
  Requires the optional ``[torch]`` extra **at runtime**; this module probes for it (``find_spec``)
  and raises a clear error when it is absent, rather than crashing mid-fit. The MLP regularises via
  weight decay (``mlp_weight_decay``), so ``ridge`` is rejected for this rung.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

from juniper_recurrence_model import LMURegressor, MLPReadoutSpec, RFFReadoutSpec

__all__ = ["ReadoutKind", "build_lmu_regressor"]

#: The readout rungs reachable over the HTTP / CLI edge. ``"mlp"`` (Rung 2b) needs the optional
#: ``[torch]`` extra **at runtime** — the spec class imports fine torch-free (torch is lazily imported
#: inside the readout), so the enum always validates; a torch-less deployment fails with a clear error
#: only when ``"mlp"`` is actually selected (see the ``find_spec`` check below).
ReadoutKind = Literal["linear", "rff", "mlp"]

_RFF_DEFAULT_FEATURES = 256
_RFF_DEFAULT_GAMMA: float | Literal["median"] = "median"
#: RFF ridge default: closed-form GCV. Regularisation is mandatory for the high-variance RFF rung.
_RFF_DEFAULT_RIDGE: float | Literal["gcv"] = "gcv"

# MLP (Rung 2b) edge defaults — mirror ``MLPReadoutSpec``'s defaults so an omitted knob is byte-identical
# to the spec default. All are optional over the edge; ``readout="mlp"`` alone uses every default.
_MLP_DEFAULT_HIDDEN = 128
_MLP_DEFAULT_WEIGHT_DECAY = 1e-4
_MLP_DEFAULT_LR = 1e-3
_MLP_DEFAULT_MAX_EPOCHS = 200
_MLP_DEFAULT_PATIENCE = 20


def build_lmu_regressor(
    *,
    d: int,
    theta: float | None,
    readout: str | None,
    ridge: float | Literal["gcv"] | None,
    rff_features: int | None = None,
    rff_gamma: float | Literal["median"] | None = None,
    mlp_hidden: int | None = None,
    mlp_weight_decay: float | None = None,
    mlp_lr: float | None = None,
    mlp_max_epochs: int | None = None,
    mlp_patience: int | None = None,
    default_ridge: float | Literal["gcv"],
) -> LMURegressor:
    """Build a fresh ``LMURegressor`` for the resolved readout selection.

    Returns a new instance on every call, so it is safe to use directly (``/v1/train``) or inside a
    per-fold factory closure (``/v1/crossval``): the RFF projection / MLP weights are (re)created in
    ``fit`` and the spec itself is immutable, so no fitted state leaks across folds.

    Args:
        d: LMU memory order.
        theta: LMU window length (``None`` ⇒ the model resolves it data-drivenly).
        readout: ``None`` / ``"linear"`` (linear), ``"rff"`` (random Fourier features, Rung 2a), or
            ``"mlp"`` (torch MLP, Rung 2b — requires the ``[torch]`` extra at runtime).
        ridge: explicit ridge penalty (float or ``"gcv"``); ``None`` ⇒ the rung default
            (``default_ridge`` for linear, GCV for RFF). Not applicable to ``"mlp"`` (which regularises
            via weight decay, exposed as ``mlp_weight_decay``).
        rff_features: RFF feature count ``D`` (``None`` ⇒ 256); rejected unless ``readout="rff"``.
        rff_gamma: RFF bandwidth γ (float or ``"median"``; ``None`` ⇒ ``"median"``); rejected unless ``readout="rff"``.
        mlp_hidden / mlp_weight_decay / mlp_lr / mlp_max_epochs / mlp_patience: MLP hyperparameters
            (each ``None`` ⇒ the ``MLPReadoutSpec`` default); each rejected unless ``readout="mlp"``.
        default_ridge: the service's configured linear-readout ridge default.

    Raises:
        ValueError: if ``readout`` is unknown; if a rung's knobs are supplied for a different rung; if
            ``ridge`` is supplied with ``readout="mlp"``; or if ``readout="mlp"`` is requested on a
            deployment without the ``[torch]`` extra installed.
    """
    kind = readout or "linear"
    # No rung's knobs may be silently dropped on another rung. The HTTP edge also rejects this at the
    # schema layer; enforcing it here (the single shared translation point) covers the CLI and every
    # other caller, so all surfaces behave identically.
    rff_knobs_set = rff_features is not None or rff_gamma is not None
    mlp_knobs_set = any(v is not None for v in (mlp_hidden, mlp_weight_decay, mlp_lr, mlp_max_epochs, mlp_patience))
    if kind != "rff" and rff_knobs_set:
        raise ValueError("rff_features / rff_gamma are only valid when readout='rff'")
    if kind != "mlp" and mlp_knobs_set:
        raise ValueError("mlp_hidden / mlp_weight_decay / mlp_lr / mlp_max_epochs / mlp_patience are only valid when readout='mlp'")
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
    if kind == "mlp":
        if ridge is not None:
            raise ValueError("ridge is not applicable to readout='mlp' (the MLP regularises via weight decay; set mlp_weight_decay)")
        # The spec imports torch-free, but fitting needs the [torch] extra. Probe here so a torch-less
        # deployment returns a clear, immediate error instead of crashing mid-fit (``find_spec`` does
        # not import torch, so this stays cheap on the request path).
        if importlib.util.find_spec("torch") is None:
            raise ValueError("readout='mlp' (DP-3 Rung 2b) requires torch; install the extra: pip install 'juniper-recurrence[torch]'")
        spec_mlp = MLPReadoutSpec(
            hidden=mlp_hidden if mlp_hidden is not None else _MLP_DEFAULT_HIDDEN,
            weight_decay=mlp_weight_decay if mlp_weight_decay is not None else _MLP_DEFAULT_WEIGHT_DECAY,
            lr=mlp_lr if mlp_lr is not None else _MLP_DEFAULT_LR,
            max_epochs=mlp_max_epochs if mlp_max_epochs is not None else _MLP_DEFAULT_MAX_EPOCHS,
            patience=mlp_patience if mlp_patience is not None else _MLP_DEFAULT_PATIENCE,
        )
        return LMURegressor(d=d, theta=theta, readout=spec_mlp)
    raise ValueError(f"unknown readout {kind!r}; expected 'linear', 'rff', or 'mlp'")
