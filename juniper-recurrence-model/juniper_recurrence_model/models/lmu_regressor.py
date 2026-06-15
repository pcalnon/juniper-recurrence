"""Fixed-order Δt-native LMU regressor (P3-C / Approach-C) as a ``TrainableModel``.

The first *real recurrence* model in juniper-recurrence wired to the shared
``juniper-model-core`` contract. It is the WS-4 refactor template: a concrete
:class:`~juniper_model_core.interfaces.TrainableModel` that passes model-core's
conformance kit unchanged, proving a non-cascor model plugs into the same seam.

Design
------
* **Fixed memory, trained readout only.** The LMU memory
  (:class:`~juniper_recurrence_model.units.lmu_varstep.VariableStepLMUMemory`) uses the
  closed-form, never-learned LegT ``A``/``B`` matrices (C1-clean). Only the linear readout
  ``W`` is fit, in **closed form** via ``numpy.linalg.lstsq`` — no BPTT, no autodiff, fully
  deterministic (so ``random_seed`` has no stochastic effect; it is carried for contract
  symmetry and round-tripped through serialization).
* **Δt-native.** Each sample is a ``(T, F)`` window with optional per-step real gaps ``dt``
  ``(n, T)``; the memory is discretised at the *actual* gaps (the irregular-Δt win
  Approach-C exists to deliver). When no ``dt`` is supplied the gaps default to uniform ones,
  so the bare ABC ``predict(X)`` still works.
* **Readout row.** One target per window (the equities_seq shape): the readout reads the
  memory state at the last masked step (or the final step ``T-1`` when no ``readout_mask`` is
  given), concatenated across features, plus a bias term.

See the design of record (in juniper-ml):
``notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from juniper_model_core.interfaces import TaskType, TrainableModel, TrainResult
from juniper_model_core.serialization import ModelSerializer

from juniper_recurrence_model.units import VariableStepLMUMemory

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

    from juniper_model_core.events import TrainingEvent
    from juniper_model_core.topology import Topology

__all__ = ["FixedOrderLMURegressor", "LMURegressorSerializer"]


class FixedOrderLMURegressor(TrainableModel):
    """A fixed-order, Δt-native LMU regressor with a closed-form least-squares readout.

    Parameters
    ----------
    d:
        LMU memory order (Legendre coefficients per feature); see
        :func:`~juniper_recurrence_model.units.lmu_varstep.lmu_matrices`.
    theta:
        Memory window length in the same real-time units as ``dt``. If ``None`` (default), it
        is resolved data-driven at :meth:`fit` time: the median per-sample window time-span
        (``median(sum(dt, axis=1))``) when ``dt`` is given, else the step count ``T``.
    task_type:
        Must be ``"regression"`` — this model has no classification head; any other value
        raises ``ValueError``.
    random_seed:
        Carried for contract symmetry and round-tripped through serialization. Training is
        deterministic (closed-form least squares), so the seed has no stochastic effect.
    """

    def __init__(
        self,
        d: int = 16,
        theta: float | None = None,
        task_type: TaskType = "regression",
        random_seed: int | None = 0,
    ) -> None:
        if task_type != "regression":
            raise ValueError(f"FixedOrderLMURegressor only supports task_type='regression', got {task_type!r}")
        self.d = int(d)
        self.task_type: TaskType = task_type
        self.random_seed = random_seed
        self._theta = theta
        self._mem: VariableStepLMUMemory | None = None
        self._W: np.ndarray | None = None
        self._metrics: dict[str, float] = {}
        self._in_shape: tuple[int, int] | None = None
        self._out_shape: tuple[int] | None = None

    # ----- feature construction -------------------------------------------------------
    def _features(self, X: np.ndarray, dt: np.ndarray | None, readout_mask: np.ndarray | None = None) -> np.ndarray:
        """Project each ``(T, F)`` window onto the fixed LMU memory and read out one row.

        For sample ``i`` and feature ``f``, rolls the memory over the scalar drive
        ``X[i, :, f]`` with per-step gaps ``dt[i]`` (uniform ones when ``dt is None``) to get a
        ``(T, d)`` trajectory, then selects the readout row — the last truthy index of
        ``readout_mask[i]`` if a mask is given, else ``T - 1``. The ``F`` per-feature ``(d,)``
        memory vectors are concatenated to ``(F*d,)``; stacking over samples and appending a
        bias column of ones yields ``(n, F*d + 1)``.
        """
        X = np.asarray(X, dtype=float)
        n, timesteps, n_features = X.shape
        if dt is None:
            dt = np.ones((n, timesteps), dtype=float)
        else:
            dt = np.asarray(dt, dtype=float)
        if self._mem is None:
            raise RuntimeError("model memory is not initialized; call fit() before predict()")

        rows = np.empty((n, n_features * self.d + 1), dtype=float)
        for i in range(n):
            if readout_mask is not None:
                truthy = np.flatnonzero(np.asarray(readout_mask[i]))
                read_idx = int(truthy[-1]) if truthy.size else timesteps - 1
            else:
                read_idx = timesteps - 1
            per_feature = np.empty(n_features * self.d, dtype=float)
            for f in range(n_features):
                traj = self._mem.rollout(X[i, :, f], dt[i])  # (T, d)
                per_feature[f * self.d : (f + 1) * self.d] = traj[read_idx]
            rows[i, :-1] = per_feature
            rows[i, -1] = 1.0  # bias
        return rows

    # ----- TrainableModel contract ----------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        on_event: Callable[[TrainingEvent], None] | None = None,
        dt: np.ndarray | None = None,
        readout_mask: np.ndarray | None = None,
        **kw: Any,
    ) -> TrainResult:
        """Fit the linear readout in closed form (one deterministic least-squares solve)."""
        # Local import keeps the top-level module import free of the events dependency until
        # an event is actually emitted (the contract references it only in type annotations).
        from juniper_model_core.events import TrainingEvent

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if on_event is not None:
            on_event(TrainingEvent("training_start", {"n_samples": int(X.shape[0])}, 0))

        # Resolve theta (data-driven when not pinned) and build the fixed LMU memory.
        if self._theta is None:
            if dt is not None:
                theta = float(np.median(np.sum(np.asarray(dt, dtype=float), axis=1)))
            else:
                theta = float(X.shape[1])
            if not theta > 0:
                theta = float(X.shape[1])
            self._theta = theta
        self._mem = VariableStepLMUMemory(self.d, self._theta)

        phi = self._features(X, dt, readout_mask)
        self._W, *_ = np.linalg.lstsq(phi, y, rcond=None)

        self._in_shape = (int(X.shape[1]), int(X.shape[2]))
        self._out_shape = (int(y.shape[1]),)

        self._metrics = self._reg_metrics(y, phi @ self._W)
        if X_val is not None and y_val is not None:
            y_val_arr = np.asarray(y_val, dtype=float)
            if y_val_arr.ndim == 1:
                y_val_arr = y_val_arr.reshape(-1, 1)
            phi_val = self._features(np.asarray(X_val, dtype=float), None, None)
            val_metrics = self._reg_metrics(y_val_arr, phi_val @ self._W)
            # val_* keys are not classification-only, so validate_metrics accepts them.
            self._metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        if on_event is not None:
            on_event(TrainingEvent("epoch_end", {"epoch": 0, "metrics": dict(self._metrics)}, 1))
            on_event(TrainingEvent("training_end", {"metrics": dict(self._metrics)}, 2))

        return TrainResult(final_metrics=dict(self._metrics), n_epochs=1, history=[dict(self._metrics)])

    def predict(self, X: np.ndarray, dt: np.ndarray | None = None, readout_mask: np.ndarray | None = None) -> np.ndarray:
        """Return continuous predictions ``(n, output_dim)``.

        ``dt`` / ``readout_mask`` are optional extensions that keep the model Δt-native when a
        caller supplies them. The base ``TrainableModel.predict(X)`` signature takes only
        ``X``; the auxiliary-array gap in the ABC's ``predict`` (no ``**kw``) was surfaced by
        the second implementer (RK-4). With the uniform ``dt`` default, the bare ``predict(X)``
        still works — which is exactly what the conformance kit exercises.
        """
        if self._W is None:
            raise RuntimeError("model is not fitted; call fit() before predict()")
        phi = self._features(np.asarray(X, dtype=float), dt, readout_mask)
        return phi @ self._W

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def describe_topology(self) -> Topology:
        return {
            "model_type": "lmu",
            "nodes": [
                {"id": "input", "kind": "input", "frozen": True},
                {"id": "memory", "kind": "memory", "frozen": True},
                {"id": "readout", "kind": "output", "frozen": False},
            ],
            "edges": [
                {"src": "input", "dst": "memory", "recurrent": False},
                {"src": "memory", "dst": "memory", "recurrent": True},
                {"src": "memory", "dst": "readout", "recurrent": False},
            ],
            "meta": {"n_units": 0, "task_type": self.task_type, "d": self.d, "theta": self._theta},
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._in_shape or (0, 0)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._out_shape or (0,)

    # ----- metrics --------------------------------------------------------------------
    def _reg_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Canonical regression metrics (all valid ``REGRESSION_METRIC_KEYS``)."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        resid = y_true - y_pred
        mse = float(np.mean(resid**2))
        mae = float(np.mean(np.abs(resid)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "r2": float(r2)}


class LMURegressorSerializer(ModelSerializer):
    """Persist / restore a :class:`FixedOrderLMURegressor`.

    The model is fully described by its hyperparameters plus the trained readout ``W``; the
    fixed LMU memory is rebuilt from ``(d, theta)`` on load. Saved as a single ``.npz``.
    Round-trip is lossless for :meth:`~FixedOrderLMURegressor.predict`: an identical ``W`` and
    an identically-rebuilt memory yield byte-identical predictions.
    """

    def save(self, model: TrainableModel, path: str | os.PathLike[str]) -> None:
        if not isinstance(model, FixedOrderLMURegressor):
            raise TypeError(f"LMURegressorSerializer can only save FixedOrderLMURegressor, got {type(model).__name__}")
        if model._W is None or model._theta is None:
            raise ValueError("cannot save an unfitted FixedOrderLMURegressor (call fit() first)")
        np.savez(
            str(path) + ".npz",
            d=np.int64(model.d),
            theta=np.float64(model._theta),
            W=np.asarray(model._W, dtype=float),
            in_shape=np.asarray(model._in_shape if model._in_shape is not None else (0, 0), dtype=np.int64),
            out_shape=np.asarray(model._out_shape if model._out_shape is not None else (0,), dtype=np.int64),
            task_type=np.str_(model.task_type),
            random_seed=np.int64(model.random_seed if model.random_seed is not None else -1),
            random_seed_is_none=np.bool_(model.random_seed is None),
        )

    def load(self, path: str | os.PathLike[str]) -> FixedOrderLMURegressor:
        with np.load(str(path) + ".npz", allow_pickle=False) as data:
            d = int(data["d"])
            theta = float(data["theta"])
            seed = None if bool(data["random_seed_is_none"]) else int(data["random_seed"])
            task_type = str(data["task_type"])
            model = FixedOrderLMURegressor(d=d, theta=theta, task_type=task_type, random_seed=seed)
            model._theta = theta
            model._mem = VariableStepLMUMemory(d, theta)
            model._W = np.asarray(data["W"], dtype=float)
            model._in_shape = (int(data["in_shape"][0]), int(data["in_shape"][1]))
            model._out_shape = (int(data["out_shape"][0]),)
            model._metrics = {}
        return model
