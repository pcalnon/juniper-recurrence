"""Fixed-order Δt-native LMU regressor — ``juniper-model-core`` ``TrainableModel``.

This is the WS-4 model layer: a standalone, fixed-order, irregular-Δt-native
Legendre Memory Unit **regressor** that satisfies ``juniper_model_core.TrainableModel``.
It wraps the fixed :class:`~juniper_recurrence_model.units.VariableStepLMUMemory` cell
with the only trained surface — a **readout** drawn from the DP-3 readout spectrum.

Design (ratified decisions D-WS4-1…3, plan
``notes/JUNIPER_RECURRENCE_WS4_MODEL_BUILD_PLAN_2026-06-15.md``; readout spectrum
``notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md``, both in juniper-ml):

* **D-WS4-1 — per-feature identity read-in.** Each of the ``F`` input features drives its
  own order-``d`` memory through the *same* fixed ``A``/``B``/θ (no trained projection); the
  per-window memory state is the concatenation ``M ∈ ℝ^{F·d}``. Only the readout is trained.
* **D-WS4-2 — ``target_dt`` as a readout feature.** When supplied, the irregular forecast
  horizon is concatenated to the readout design matrix as a **linear side-channel** — appended
  *after* any readout nonlinearity; the readout itself only ever sees the memory block ``M``.
* **D-WS4-3 — standalone.** No cascor cascade head; this regressor has its own readout.

The readout is configured by an **immutable spec** (:mod:`juniper_recurrence_model.readouts`) and
materialised into a fresh fitted instance inside :meth:`LMURegressor.fit`, so a spec shared across
cross-validation folds never leaks one fold's fitted weights into another. The default
:class:`~juniper_recurrence_model.readouts.LinearReadoutSpec` (``ridge=0.0``) is the closed-form
``lstsq`` solve over the LMU-memory feature map — numpy-only, no autodiff — the structural twin of
``juniper_model_core``'s ``ReferenceLinearModel`` with its ``_flatten(X)`` feature map replaced by a
dt-aware LMU-memory rollout. (A torch-backed nonlinear readout — the point at which torch enters —
is a deferred increment, gated behind a ``[torch]`` extra.)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from juniper_model_core import ModelSerializer, TaskType, Topology, TrainableModel, TrainingEvent, TrainResult

from juniper_recurrence_model.readouts import LinearReadout, LinearReadoutSpec, ReadoutSpec, RidgeParam, build_readout_from_state
from juniper_recurrence_model.units.lmu_varstep import VariableStepLMUMemory

__all__ = ["LMURegressor", "LMUSerializer"]

#: npz key prefix under which a readout's fitted arrays are namespaced (DP-3 serializer schema 2).
_READOUT_ARRAY_PREFIX = "readout__"


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Canonical regression metrics (``REGRESSION_METRIC_KEYS``); never ``accuracy`` (RK-6)."""
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": mse**0.5, "mae": mae, "r2": r2, "loss": mse}


class LMURegressor(TrainableModel):
    """Fixed-order Δt-native LMU regressor (per-feature identity read-in + spectrum readout).

    Parameters
    ----------
    d:
        LMU memory order (Legendre coefficients per feature). Practical range ~4..64.
    theta:
        Memory window length, in the same real-time units as ``dt`` (e.g. calendar days).
        ``None`` (default) resolves it data-drivenly at ``fit``: the median per-window total
        elapsed time ``median(sum(dt, axis=1))``, falling back to the window length ``T`` when
        ``dt`` is absent or non-positive.
    readout:
        An immutable readout **spec** (see :mod:`juniper_recurrence_model.readouts`). ``None``
        (default) builds a :class:`~juniper_recurrence_model.readouts.LinearReadoutSpec` from
        ``ridge`` — so ``LMURegressor(d, theta, ridge=…)`` is byte-identical to the pre-DP-3 model.
        Passing both a non-default ``ridge`` and a ``readout`` is rejected (one source of truth).
    ridge:
        Convenience for the default linear readout's L2 penalty (the bias column is never
        penalised). ``0.0`` (default) is the plain min-norm least-squares solve. A positive float
        is the regularised normal-equation solve. ``"gcv"`` selects the penalty by closed-form
        generalised cross-validation at ``fit`` (the selected λ is written back to ``self.ridge``).
        Ignored when an explicit ``readout`` spec is supplied.
    time_unit:
        Declared real-time unit of ``dt`` / ``theta`` (carried in the topology meta).
    random_seed:
        Carried for the contract and used to seed any data-independent readout randomness (e.g.
        the RFF projection). The closed-form linear fit is deterministic regardless.
    """

    def __init__(self, d: int = 16, theta: float | None = None, *, readout: ReadoutSpec | None = None, ridge: RidgeParam = 0.0, time_unit: str = "steps", random_seed: int | None = 0) -> None:
        if readout is not None and ridge != 0.0:
            raise ValueError("pass either a `readout` spec or a `ridge` value, not both; `ridge` configures the default linear readout")
        self.task_type: TaskType = "regression"
        self.random_seed = random_seed
        self.d = int(d)
        self.theta: float | None = None if theta is None else float(theta)
        self.ridge: RidgeParam = ridge
        self.time_unit = str(time_unit)
        # The readout is configured by an immutable spec and materialised fresh in fit().
        self._readout_spec: ReadoutSpec = readout if readout is not None else LinearReadoutSpec(ridge=ridge)
        self._readout: Any = None
        # When theta is data-driven (None) the fixed memory is built in fit(); see fit().
        self._memory = None if self.theta is None else VariableStepLMUMemory(self.d, self.theta)
        self._in_shape: tuple[int, ...] = ()
        self._out_shape: tuple[int, ...] = ()
        self._n_features: int | None = None
        self._uses_target_dt: bool = False
        self._metrics: dict[str, float] = {}

    @property
    def _coef(self) -> np.ndarray | None:
        """The linear readout's coefficients (``[M | target_dt? | 1]`` layout), or ``None``.

        Read-only forwarding property kept for backward compatibility (the serializer's
        unfitted check now consults ``readout.is_fitted``; this surfaces the coefficient vector
        that ``test_lmu_model`` and downstream callers historically read as ``model._coef``).
        ``None`` before fit and for nonlinear readouts (whose trained weights are not a single
        linear coefficient vector).
        """
        return None if self._readout is None else self._readout.coef

    # ----- feature map (shared by fit and predict) -----------------------------------
    @staticmethod
    def _readout_index(n: int, n_steps: int, readout_mask: np.ndarray | None, seq_lengths: np.ndarray | None) -> np.ndarray:
        """Per-sample index of the readout step (the last valid step, many-to-one)."""
        if seq_lengths is not None:
            return np.clip(np.asarray(seq_lengths, dtype=int) - 1, 0, n_steps - 1)
        if readout_mask is not None:
            mask = np.asarray(readout_mask, dtype=bool)
            reversed_mask = mask[:, ::-1]
            has_true = reversed_mask.any(axis=1)
            last_true = n_steps - 1 - np.argmax(reversed_mask, axis=1)
            return np.where(has_true, last_true, n_steps - 1)
        return np.full(n, n_steps - 1, dtype=int)

    def _memory_block(self, X: np.ndarray, dt: np.ndarray | None, readout_mask: np.ndarray | None, seq_lengths: np.ndarray | None) -> np.ndarray:
        """Roll the fixed LMU memory over ``X`` and gather the readout-step state ``M`` (n, F·d)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (n, T, F); got shape {X.shape}")
        n, n_steps, n_features = X.shape
        if self._n_features is not None and n_features != self._n_features:
            raise ValueError(f"expected F={self._n_features} features, got {n_features}")
        if dt is None:
            dt = np.zeros((n, n_steps))
            dt[:, 1:] = 1.0  # uniform unit-spacing fallback (bare predict(X) — no timing supplied)
        trajectory = self._memory.rollout_batch(X, dt)  # (n, T, F, d)
        idx = self._readout_index(n, n_steps, readout_mask, seq_lengths)
        return trajectory[np.arange(n), idx].reshape(n, n_features * self.d)  # (n, F*d)

    def _side_channel(self, target_dt: np.ndarray | None, n: int) -> np.ndarray:
        """The linear side-channel appended to the readout design: the ``target_dt`` column, or ``(n, 0)``."""
        if not self._uses_target_dt:
            return np.empty((n, 0))
        horizon = np.zeros(n) if target_dt is None else np.asarray(target_dt, dtype=float).reshape(n)
        return horizon[:, None]

    # ----- TrainableModel contract ---------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, *, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, on_event: Any = None, **kw: Any) -> TrainResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (n, T, F); got shape {X.shape}")
        if y.ndim == 3:
            raise NotImplementedError("dense many-to-many readout is a deferred WS-4 increment; supply one target per window (y of shape (n,) or (n, output_dim))")
        if y.ndim == 1:
            y = y[:, None]
        n, n_steps, n_features = X.shape
        self._in_shape = (n_steps, n_features)
        self._n_features = n_features
        self._out_shape = (int(y.shape[1]),)
        self._uses_target_dt = kw.get("target_dt") is not None
        # Resolve a data-driven theta (median per-window elapsed time) when not pinned,
        # then build the fixed LMU memory. A pinned theta is used as-is.
        if self.theta is None:
            window_dt = kw.get("dt")
            theta = float(np.median(np.sum(np.asarray(window_dt, dtype=float), axis=1))) if window_dt is not None else float(n_steps)
            self.theta = theta if theta > 0 else float(n_steps)
        if self._memory is None:
            self._memory = VariableStepLMUMemory(self.d, self.theta)

        seq = 0
        if on_event is not None:
            on_event(TrainingEvent("training_start", {"n_samples": int(n)}, seq))
            seq += 1

        memory_block = self._memory_block(X, kw.get("dt"), kw.get("readout_mask"), kw.get("seq_lengths"))
        side_channel = self._side_channel(kw.get("target_dt"), n)
        self._readout = self._readout_spec.make()
        self._readout.fit(memory_block, side_channel, y, random_seed=self.random_seed)
        if self._readout.kind == "linear":
            # Propagate a GCV-selected λ (or the fixed penalty) to the envelope so meta["ridge"]
            # records it for retraining fidelity (the lossless test can't catch its omission).
            self.ridge = self._readout.ridge
        self._metrics = _regression_metrics(y, self._readout.predict(memory_block, side_channel))

        if on_event is not None:
            on_event(TrainingEvent("epoch_end", {"epoch": 0, "metrics": dict(self._metrics)}, seq))
            seq += 1
            on_event(TrainingEvent("training_end", {"metrics": dict(self._metrics)}, seq))
        return TrainResult(final_metrics=dict(self._metrics), n_epochs=1, history=[dict(self._metrics)], stopped_reason="converged")

    def predict(self, X: np.ndarray, *, dt: np.ndarray | None = None, target_dt: np.ndarray | None = None, readout_mask: np.ndarray | None = None, seq_lengths: np.ndarray | None = None) -> np.ndarray:
        """Continuous predictions for ``X``.

        The signature widens the ``TrainableModel.predict(X)`` contract with *optional*
        sequence keywords (the ABC checks the method name, not the signature). When ``dt``
        is omitted — as the conformance kit calls it — a uniform unit grid is assumed; real
        callers pass ``dt`` (and ``target_dt`` when the model was fit with one) to engage the
        Δt path. Never returns an ``argmax`` (RK-6 — collapsing to labels is classification-only).
        """
        if self._readout is None or not self._readout.is_fitted:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        memory_block = self._memory_block(X, dt, readout_mask, seq_lengths)
        side_channel = self._side_channel(target_dt, n)
        return self._readout.predict(memory_block, side_channel).reshape((n, *self._out_shape))

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def describe_topology(self) -> Topology:
        readout_kind = self._readout.kind if self._readout is not None else self._readout_spec.kind
        return {
            "model_type": "lmu",
            "nodes": [
                {"id": "input", "kind": "input", "frozen": True},
                {"id": "memory", "kind": "memory", "frozen": True},
                {"id": "output", "kind": "output", "frozen": False},
            ],
            "edges": [
                {"src": "input", "dst": "memory", "recurrent": False},
                {"src": "memory", "dst": "memory", "recurrent": True},
                {"src": "memory", "dst": "output", "recurrent": False},
            ],
            "meta": {
                "n_units": 0,
                "task_type": self.task_type,
                "theta": self.theta,
                "d": self.d,
                "time_unit": self.time_unit,
                "n_features": self._n_features,
                # DP-3: nested readout descriptor (the LMU envelope keys above stay frozen — esp.
                # meta["d"] = memory order, asserted ==4 by the recurrence app's test_routes).
                "readout": {"kind": readout_kind},
            },
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._out_shape


class LMUSerializer(ModelSerializer):
    """Lossless ``.npz`` + JSON serializer for :class:`LMURegressor`.

    Persists the LMU envelope (``d``/θ/etc.) plus the readout's own fitted state (via the
    readout's ``save_state``), namespaced under ``readout__*`` arrays with a JSON descriptor in
    ``meta["readout"]``. The fixed memory eigendecomposition is recomputed from ``d``/θ on load
    (deterministic), so a linear readout's reloaded predictions are bit-identical (the conformance
    kit's lossless-round-trip assertion).

    Backward compatibility: a pre-DP-3 file has a top-level ``coef`` array and no
    ``meta["readout"]``; :meth:`load` detects that and reconstructs a linear readout from
    ``meta["ridge"]`` + ``coef``.
    """

    def save(self, model: TrainableModel, path: str | os.PathLike[str]) -> None:
        if not isinstance(model, LMURegressor):
            raise TypeError("LMUSerializer only serializes LMURegressor")
        if model._readout is None or not model._readout.is_fitted:
            raise RuntimeError("cannot serialize an unfitted model")
        arrays, readout_descriptor = model._readout.save_state()
        meta = {
            "schema": 2,
            "d": model.d,
            "theta": model.theta,
            "ridge": model.ridge if not isinstance(model.ridge, str) else 0.0,
            "time_unit": model.time_unit,
            "random_seed": model.random_seed,
            "task_type": model.task_type,
            "in_shape": list(model._in_shape),
            "out_shape": list(model._out_shape),
            "n_features": model._n_features,
            "uses_target_dt": model._uses_target_dt,
            "metrics": model._metrics,
            "readout": readout_descriptor,
        }
        namespaced = {f"{_READOUT_ARRAY_PREFIX}{key}": value for key, value in arrays.items()}
        np.savez(os.fspath(path), meta=json.dumps(meta), **namespaced)

    def load(self, path: str | os.PathLike[str]) -> LMURegressor:
        resolved = os.fspath(path)
        if not resolved.endswith(".npz"):
            resolved = resolved + ".npz"
        with np.load(resolved, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            readout_descriptor = meta.get("readout")
            if readout_descriptor is None:
                # Pre-DP-3 format: a top-level "coef" array, ridge in meta, implicit linear readout.
                readout = LinearReadout.from_state({"coef": data["coef"]}, {"kind": "linear", "ridge": meta.get("ridge", 0.0)})
            else:
                arrays = {key[len(_READOUT_ARRAY_PREFIX) :]: data[key] for key in data.files if key.startswith(_READOUT_ARRAY_PREFIX)}
                readout = build_readout_from_state(arrays, readout_descriptor)
        model = LMURegressor(d=meta["d"], theta=meta["theta"], time_unit=meta["time_unit"], random_seed=meta["random_seed"])
        model._readout = readout
        if isinstance(readout, LinearReadout):
            model.ridge = readout.ridge
            model._readout_spec = LinearReadoutSpec(ridge=readout.ridge)
        model._in_shape = tuple(meta["in_shape"])
        model._out_shape = tuple(meta["out_shape"])
        model._n_features = meta["n_features"]
        model._uses_target_dt = bool(meta["uses_target_dt"])
        model._metrics = {key: float(value) for key, value in meta["metrics"].items()}
        return model
