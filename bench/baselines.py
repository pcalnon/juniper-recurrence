"""Baselines + the fixed-Δt control for the recurrence benchmark.

- :func:`uniform_dt` — replace the real per-step gaps with a uniform unit grid; feeding this to the
  *same* ``LMURegressor`` (same ``theta``) is the **fixed-Δt negative control** (the §9.1a foil:
  timing information removed, everything else held constant), isolating the Δt contribution.
- :class:`NaivePersistence` — predict the last observed signal value (the forecast floor).
- :class:`LinearRidge` — closed-form ridge on the last-step features (+ ``target_dt``); an honest
  non-temporal baseline.

Both baselines satisfy ``juniper_model_core.TrainableModel`` so ``cross_validate`` drives them on the
exact same folds as the LMU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from juniper_model_core import TaskType, Topology, TrainableModel, TrainResult

__all__ = ["uniform_dt", "NaivePersistence", "LinearRidge"]


def uniform_dt(dt: np.ndarray) -> np.ndarray:
    """A uniform unit-spacing grid the shape of ``dt`` (``[:, 0] = 0``, ``[:, 1:] = 1``).

    Feeding this in place of the real gaps is the fixed-Δt control: the LMU rolls its memory on a
    regular grid, exactly as if it had no timing information.
    """
    u = np.zeros_like(np.asarray(dt, dtype=float))
    if u.shape[1] > 1:
        u[:, 1:] = 1.0
    return u


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    err = np.asarray(y_pred, dtype=float) - yt
    mse = float(np.mean(err**2))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": float(np.mean(np.abs(err))),
        "r2": r2,
        "loss": mse,
    }


class _BaselineModel(TrainableModel):
    """Shared ``TrainableModel`` boilerplate for the regression baselines."""

    _model_type = "baseline"

    def __init__(self) -> None:
        self.task_type: TaskType = "regression"
        self.random_seed: int | None = 0
        self._in: tuple[int, ...] = ()
        self._out: tuple[int, ...] = ()
        self._metrics: dict[str, float] = {}

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def describe_topology(self) -> Topology:
        return {
            "model_type": self._model_type,
            "nodes": [
                {"id": "input", "kind": "input", "frozen": True},
                {"id": "output", "kind": "output", "frozen": False},
            ],
            "edges": [{"src": "input", "dst": "output", "recurrent": False}],
            "meta": {"n_units": 0, "task_type": self.task_type},
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._in

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._out

    @staticmethod
    def _last_step(X: np.ndarray) -> np.ndarray:
        """Last-step feature vector per window: ``(n, F)``."""
        return np.asarray(X, dtype=float)[:, -1, :]


class NaivePersistence(_BaselineModel):
    """Predict the last observed value of feature 0 (the signal) — the persistence floor."""

    _model_type = "naive_persistence"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Any = None,
        y_val: Any = None,
        on_event: Any = None,
        **kw: Any,
    ) -> TrainResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._in = X.shape[1:]
        self._out = y.shape[1:] if y.ndim > 1 else (1,)
        self._metrics = _regression_metrics(y, self.predict(X))
        return TrainResult(
            final_metrics=self._metrics,
            n_epochs=1,
            history=[],
            stopped_reason="converged",
        )

    def predict(self, X: np.ndarray, **kw: Any) -> np.ndarray:
        return self._last_step(X)[:, 0:1]  # (n, 1) — last observed signal value


class LinearRidge(_BaselineModel):
    """Closed-form ridge on the last-step features (+ ``target_dt``) — non-temporal baseline."""

    _model_type = "linear_ridge"

    def __init__(self, ridge: float = 1e-3) -> None:
        super().__init__()
        self.ridge = float(ridge)
        self._coef: np.ndarray | None = None

    def _design(self, X: np.ndarray, target_dt: np.ndarray | None) -> np.ndarray:
        feats = self._last_step(X)
        n = feats.shape[0]
        cols = [feats]
        if target_dt is not None:
            cols.append(np.asarray(target_dt, dtype=float).reshape(n, 1))
        cols.append(np.ones((n, 1)))  # bias
        return np.concatenate(cols, axis=1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Any = None,
        y_val: Any = None,
        on_event: Any = None,
        **kw: Any,
    ) -> TrainResult:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(y, dtype=float)
        Y2 = Y if Y.ndim > 1 else Y.reshape(-1, 1)
        self._in = X.shape[1:]
        self._out = Y2.shape[1:]
        design = self._design(X, kw.get("target_dt"))
        p = design.shape[1]
        reg = self.ridge * np.eye(p)
        reg[-1, -1] = 0.0  # never penalise the bias column
        self._coef = np.linalg.solve(design.T @ design + reg, design.T @ Y2)
        self._metrics = _regression_metrics(Y2, design @ self._coef)
        return TrainResult(
            final_metrics=self._metrics,
            n_epochs=1,
            history=[],
            stopped_reason="converged",
        )

    def predict(self, X: np.ndarray, **kw: Any) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("LinearRidge.predict called before fit")
        return (
            self._design(np.asarray(X, dtype=float), kw.get("target_dt")) @ self._coef
        )
