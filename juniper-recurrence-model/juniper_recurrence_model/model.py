"""Fixed-order Δt-native LMU regressor — ``juniper-model-core`` ``TrainableModel``.

This is the WS-4 model layer: a standalone, fixed-order, irregular-Δt-native
Legendre Memory Unit **regressor** that satisfies ``juniper_model_core.TrainableModel``.
It wraps the fixed :class:`~juniper_recurrence_model.units.VariableStepLMUMemory` cell
with the only trained surface — a closed-form least-squares **readout**.

Design (ratified decisions D-WS4-1…3, plan
``notes/JUNIPER_RECURRENCE_WS4_MODEL_BUILD_PLAN_2026-06-15.md`` in juniper-ml):

* **D-WS4-1 — per-feature identity read-in.** Each of the ``F`` input features drives its
  own order-``d`` memory through the *same* fixed ``A``/``B``/θ (no trained projection); the
  per-window memory state is the concatenation ``M ∈ ℝ^{F·d}``. Only the readout is trained.
* **D-WS4-2 — ``target_dt`` as a readout feature.** When supplied, the irregular forecast
  horizon is concatenated to the readout design matrix.
* **D-WS4-3 — standalone.** No cascor cascade head; this regressor has its own readout.

Because the memory matrices are fixed (never differentiated) and the readout is linear, the
whole model is a closed-form ``lstsq`` solve over an LMU-memory feature map — **numpy-only,
no autodiff framework**. This is the structural twin of ``juniper_model_core``'s
``ReferenceLinearModel`` with its ``_flatten(X)`` feature map replaced by a dt-aware
LMU-memory rollout. (A trained projection read-in / nonlinear readout — the point at which
torch would enter — is a deferred increment.)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from juniper_model_core import ModelSerializer, TaskType, Topology, TrainableModel, TrainingEvent, TrainResult

from juniper_recurrence_model.units.lmu_varstep import VariableStepLMUMemory

__all__ = ["LMURegressor", "LMUSerializer"]


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
    """Fixed-order Δt-native LMU regressor (per-feature identity read-in + linear readout).

    Parameters
    ----------
    d:
        LMU memory order (Legendre coefficients per feature). Practical range ~4..64.
    theta:
        Memory window length, in the same real-time units as ``dt`` (e.g. calendar days).
        ``None`` (default) resolves it data-drivenly at ``fit``: the median per-window total
        elapsed time ``median(sum(dt, axis=1))``, falling back to the window length ``T`` when
        ``dt`` is absent or non-positive.
    ridge:
        L2 penalty on the readout (the bias column is never penalised). ``0.0`` (default)
        uses a plain min-norm least-squares solve — which lets the readout memorise a tiny
        set exactly (the overfit-tiny guarantee) and mirrors the reference model.
    time_unit:
        Declared real-time unit of ``dt`` / ``theta`` (carried in the topology meta).
    random_seed:
        Stored for the contract; the closed-form fit is deterministic regardless.
    """

    def __init__(self, d: int = 16, theta: float | None = None, *, ridge: float = 0.0, time_unit: str = "steps", random_seed: int | None = 0) -> None:
        self.task_type: TaskType = "regression"
        self.random_seed = random_seed
        self.d = int(d)
        self.theta: float | None = None if theta is None else float(theta)
        self.ridge = float(ridge)
        self.time_unit = str(time_unit)
        # When theta is data-driven (None) the fixed memory is built in fit(); see fit().
        self._memory = None if self.theta is None else VariableStepLMUMemory(self.d, self.theta)
        self._coef: np.ndarray | None = None
        self._in_shape: tuple[int, ...] = ()
        self._out_shape: tuple[int, ...] = ()
        self._n_features: int | None = None
        self._uses_target_dt: bool = False
        self._metrics: dict[str, float] = {}

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

    def _features(self, X: np.ndarray, dt: np.ndarray | None, target_dt: np.ndarray | None, readout_mask: np.ndarray | None, seq_lengths: np.ndarray | None) -> np.ndarray:
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
        memory_state = trajectory[np.arange(n), idx].reshape(n, n_features * self.d)  # (n, F*d)
        columns = [memory_state]
        if self._uses_target_dt:
            horizon = np.zeros(n) if target_dt is None else np.asarray(target_dt, dtype=float).reshape(n)
            columns.append(horizon[:, None])
        columns.append(np.ones((n, 1)))  # bias
        return np.concatenate(columns, axis=1)

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

        design = self._features(X, kw.get("dt"), kw.get("target_dt"), kw.get("readout_mask"), kw.get("seq_lengths"))
        if self.ridge > 0.0:
            gram = design.T @ design
            penalty = self.ridge * np.eye(gram.shape[0])
            penalty[-1, -1] = 0.0  # never regularise the bias column
            coef = np.linalg.solve(gram + penalty, design.T @ y)
        else:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self._coef = coef
        self._metrics = _regression_metrics(y, design @ coef)

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
        if self._coef is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=float)
        design = self._features(X, dt, target_dt, readout_mask, seq_lengths)
        return (design @ self._coef).reshape((X.shape[0], *self._out_shape))

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def describe_topology(self) -> Topology:
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

    Persists the trained readout coefficients plus the hyperparameters; the fixed memory
    eigendecomposition is recomputed from ``d``/θ on load (deterministic), so reloaded
    predictions are bit-identical (the conformance kit's lossless-round-trip assertion).
    """

    def save(self, model: TrainableModel, path: str | os.PathLike[str]) -> None:
        if not isinstance(model, LMURegressor):
            raise TypeError("LMUSerializer only serializes LMURegressor")
        if model._coef is None:
            raise RuntimeError("cannot serialize an unfitted model")
        meta = {
            "d": model.d,
            "theta": model.theta,
            "ridge": model.ridge,
            "time_unit": model.time_unit,
            "random_seed": model.random_seed,
            "task_type": model.task_type,
            "in_shape": list(model._in_shape),
            "out_shape": list(model._out_shape),
            "n_features": model._n_features,
            "uses_target_dt": model._uses_target_dt,
            "metrics": model._metrics,
        }
        np.savez(os.fspath(path), coef=model._coef, meta=json.dumps(meta))

    def load(self, path: str | os.PathLike[str]) -> LMURegressor:
        resolved = os.fspath(path)
        if not resolved.endswith(".npz"):
            resolved = resolved + ".npz"
        with np.load(resolved, allow_pickle=False) as data:
            coef = data["coef"]
            meta = json.loads(str(data["meta"]))
        model = LMURegressor(d=meta["d"], theta=meta["theta"], ridge=meta["ridge"], time_unit=meta["time_unit"], random_seed=meta["random_seed"])
        model._coef = coef
        model._in_shape = tuple(meta["in_shape"])
        model._out_shape = tuple(meta["out_shape"])
        model._n_features = meta["n_features"]
        model._uses_target_dt = bool(meta["uses_target_dt"])
        model._metrics = {key: float(value) for key, value in meta["metrics"].items()}
        return model
