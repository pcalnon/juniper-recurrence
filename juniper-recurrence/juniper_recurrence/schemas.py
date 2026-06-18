"""Pydantic request / response models for the juniper-recurrence API (plan §6).

Regression-generic throughout (RK-6): predictions are continuous arrays, metrics are
the regression set (``mse`` / ``rmse`` / ``mae`` / ``r2`` / ``loss``) — never an
``accuracy`` key and never an ``argmax`` collapse to class labels.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "DatasetRef",
    "DatasetDescriptor",
    "TrainRequest",
    "TrainResponse",
    "EventModel",
    "StatusResponse",
    "PredictRequest",
    "PredictResponse",
    "ModelResponse",
    "CrossValRequest",
    "CrossValFoldModel",
    "CrossValResponse",
    "CrossValStatusResponse",
]


class DatasetRef(BaseModel):
    """Reference to a 3-D sequence dataset to fetch via juniper-data-client.

    Resolution precedence: ``dataset_id`` (direct) → ``name`` (latest version) →
    ``generator`` + ``params`` (create on the fly). At least one must be supplied.
    """

    dataset_id: str | None = None
    name: str | None = None
    generator: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    split: str = "train"

    @model_validator(mode="after")
    def _require_one_ref(self) -> DatasetRef:
        if not (self.dataset_id or self.name or self.generator):
            raise ValueError("dataset ref requires one of: dataset_id, name, generator")
        return self


class DatasetDescriptor(BaseModel):
    """Thin descriptor of a loaded dataset split (``GET /v1/dataset``)."""

    dataset_id: str | None = None
    name: str | None = None
    split: str
    n_windows: int
    lookback: int
    n_features: int
    output_dim: int
    has_target_dt: bool
    has_seq_lengths: bool


class TrainRequest(BaseModel):
    """Body for ``POST /v1/train``: a dataset ref plus optional LMU hyperparameters.

    Unset hyperparameters fall back to the service defaults (``default_d`` /
    ``default_theta`` / ``default_ridge``). ``theta=None`` is meaningful — it asks the
    model to resolve θ data-drivenly from the per-window elapsed time. The irregular
    forecast horizon (``target_dt``) is engaged automatically when the dataset carries
    it (it is per-window data, not a hyperparameter).
    """

    dataset: DatasetRef
    d: int | None = Field(default=None, ge=1)
    theta: float | None = Field(default=None, gt=0)
    ridge: float | None = Field(default=None, ge=0)


class TrainResponse(BaseModel):
    """``POST /v1/train`` result: the ``TrainResult`` plus the dataset descriptor."""

    final_metrics: dict[str, float]
    n_epochs: int
    stopped_reason: str | None = None
    dataset: DatasetDescriptor


class EventModel(BaseModel):
    """One serialised :class:`~juniper_model_core.TrainingEvent`."""

    type: str
    seq: int
    payload: dict[str, Any]


class StatusResponse(BaseModel):
    """``GET /v1/training/status``: synchronous, instant (no background job)."""

    state: str  # "idle" | "trained"
    final_metrics: dict[str, float] | None = None
    stopped_reason: str | None = None
    events: list[EventModel] = Field(default_factory=list)


class PredictRequest(BaseModel):
    """Body for ``POST /v1/predict``: inline arrays **or** a dataset ref.

    ``X`` is ``(n, T, F)``; ``dt`` ``(n, T)`` engages the Δt path; ``target_dt`` ``(n,)``
    supplies the irregular horizon; ``seq_lengths`` ``(n,)`` selects the many-to-one
    readout step. Exactly one of ``X`` / ``dataset`` is required.
    """

    X: list | None = None
    dt: list | None = None
    target_dt: list | None = None
    seq_lengths: list | None = None
    dataset: DatasetRef | None = None

    @model_validator(mode="after")
    def _require_x_or_dataset(self) -> PredictRequest:
        if self.X is None and self.dataset is None:
            raise ValueError("predict requires either 'X' or 'dataset'")
        return self


class PredictResponse(BaseModel):
    """Continuous predictions ``ŷ`` and their shape (never argmax — RK-6)."""

    predictions: list
    shape: list[int]


class ModelResponse(BaseModel):
    """``GET /v1/model``: current model topology + metrics."""

    topology: dict[str, Any]
    metrics: dict[str, float]


class CrossValRequest(BaseModel):
    """Body for ``POST /v1/crossval``: a dataset ref + walk-forward CV params + LMU hyperparameters.

    Cross-validation always derives folds from the dataset's ``full`` split (the chronologically
    ordered set), so the ``split`` field of the dataset ref is not used here. Unset hyperparameters
    fall back to the service defaults; ``theta=None`` asks each fold's model to resolve θ
    data-drivenly from that fold's elapsed time.
    """

    dataset: DatasetRef
    n_folds: int = Field(ge=2)
    scheme: Literal["expanding", "rolling"] = "expanding"
    embargo: int = Field(default=0, ge=0)
    min_train: int | None = Field(default=None, ge=1)
    d: int | None = Field(default=None, ge=1)
    theta: float | None = Field(default=None, gt=0)
    ridge: float | None = Field(default=None, ge=0)


class CrossValFoldModel(BaseModel):
    """One fold's outcome: the model's own train-time metrics + the held-out eval metrics."""

    fold: int
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float]
    n_epochs: int


class CrossValResponse(BaseModel):
    """``POST /v1/crossval`` result: per-fold detail + per-metric mean / std aggregates."""

    task_type: str
    n_folds: int
    folds: list[CrossValFoldModel]
    eval_aggregate: dict[str, float]
    eval_std: dict[str, float]
    dataset: DatasetDescriptor


class CrossValStatusResponse(BaseModel):
    """``GET /v1/crossval/status``: the most recent persisted CV result, or ``idle`` if none has run."""

    state: str  # "idle" | "done"
    result: CrossValResponse | None = None
