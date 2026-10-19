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

The reverse translation lives here too: :func:`readout_diagnostics` reads a *fitted* model's
resolved hyperparameters (theta, the ridge mode and the penalty the solve used, the RFF bandwidth,
the GCV edge / null-model flags) into the :class:`~juniper_recurrence.schemas.ReadoutDiagnostics`
block the ``/v1/train`` and ``/v1/crossval`` responses and the ``train`` CLI carry (W5.10).
"""

from __future__ import annotations

import importlib.util
import math
from typing import Literal

from juniper_recurrence_model import LMURegressor, MLPReadoutSpec, RFFReadoutSpec
from juniper_recurrence_model import readouts as _model_readouts

from juniper_recurrence.schemas import ReadoutDiagnostics

__all__ = ["ReadoutKind", "build_lmu_regressor", "readout_diagnostics"]

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


def _gcv_grid_bounds() -> tuple[float, float] | None:
    """``(floor, ceiling)`` of the model's GCV grid, or ``None`` when the model does not expose it."""
    grid = getattr(_model_readouts, "_GCV_GRID", None)
    if grid is None or len(grid) == 0:
        return None
    return float(grid[0]), float(grid[-1])


def readout_diagnostics(model: LMURegressor) -> ReadoutDiagnostics:
    """Read the resolved readout hyperparameters off a **fitted** ``LMURegressor`` (W5.10).

    The request names *modes* -- ``theta=None``, ``ridge="gcv"``, ``rff_gamma="median"`` -- and the
    model resolves each at ``fit``: theta from the data, lambda from the GCV grid, gamma from the
    median heuristic. The model writes the resolved values back onto itself (``model.theta``; the
    live readout's ``ridge`` / ``gamma``) and keeps the *requested* mode on the immutable spec it was
    built from, so both ends of the translation are readable after the fit. This is the one place
    that reads them, for the two routes and the CLI alike.

    ``gcv_at_edge`` is derived here from the selected lambda and the grid's floor / ceiling when the
    penalty was GCV-selected. ``null_model`` needs the GCV curve beyond the ceiling, which only the
    model can evaluate: it is taken from the fitted readout's ``null_model_`` attribute when present
    (the W5.9 contract, juniper-recurrence-model 0.5.0; the trailing-underscore fit-diagnostic
    convention the MLP rung already uses for ``n_epochs_`` / ``stopped_reason_``) and is ``None``
    otherwise. A readout that reports its own ``gcv_at_edge_`` wins over the derivation here.

    Raises:
        RuntimeError: if ``model`` has not been fitted (there is nothing resolved to read).
    """
    readout = getattr(model, "_readout", None)
    if readout is None or not readout.is_fitted:
        raise RuntimeError("readout diagnostics need a fitted model")
    if model.theta is None or not (model.theta > 0):
        raise RuntimeError(f"fitted model has no resolved theta (theta={model.theta!r})")

    kind = str(readout.kind)
    requested_ridge = getattr(model._readout_spec, "ridge", None)
    selected = getattr(readout, "ridge", None)
    ridge_selected = None if selected is None or isinstance(selected, str) else float(selected)
    ridge_mode: Literal["fixed", "gcv"] | None
    if ridge_selected is None:
        ridge_mode = None
    else:
        ridge_mode = "gcv" if requested_ridge == "gcv" else "fixed"

    gamma_raw = getattr(readout, "gamma", None)
    gamma = None if gamma_raw is None or isinstance(gamma_raw, str) else float(gamma_raw)

    at_edge = getattr(readout, "gcv_at_edge_", None)
    if at_edge is None and ridge_mode == "gcv" and ridge_selected is not None:
        bounds = _gcv_grid_bounds()
        if bounds is not None:
            floor, ceiling = bounds
            at_floor = ridge_selected <= floor or math.isclose(ridge_selected, floor, rel_tol=1e-9)
            at_ceiling = ridge_selected >= ceiling or math.isclose(ridge_selected, ceiling, rel_tol=1e-9)
            at_edge = at_floor or at_ceiling
    null_model = getattr(readout, "null_model_", None)

    return ReadoutDiagnostics(
        readout=kind,
        theta=float(model.theta),
        ridge_mode=ridge_mode,
        ridge_selected=ridge_selected,
        gamma=gamma,
        gcv_at_edge=None if at_edge is None else bool(at_edge),
        null_model=None if null_model is None else bool(null_model),
    )
