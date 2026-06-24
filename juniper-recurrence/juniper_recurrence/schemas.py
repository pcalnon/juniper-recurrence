"""Pydantic request / response models for the juniper-recurrence API (plan §6).

Regression-generic throughout (RK-6): predictions are continuous arrays, metrics are
the regression set (``mse`` / ``rmse`` / ``mae`` / ``r2`` / ``loss``) — never an
``accuracy`` key and never an ``argmax`` collapse to class labels.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

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

# DP-3 (P1): ``ridge`` accepts a non-negative float, the literal ``"gcv"`` (closed-form GCV
# selection of the readout penalty, performed in the model), or ``None`` (fall back to
# ``settings.default_ridge``). The ``ge=0`` bound applies only to the float member of the union.
RidgeField = Annotated[float, Field(ge=0)] | Literal["gcv"] | None

# DP-3: the readout *spectrum* exposed as a tagged enum at the HTTP edge. ``None`` ⇒ the back-compat
# linear readout (which uses ``ridge``); ``"rff"`` selects the nonlinear random-Fourier-feature readout
# (Rung 2a, P2c), configured by ``rff_features`` / ``rff_gamma``; ``"mlp"`` selects the torch MLP readout
# (Rung 2b, P3 — needs the ``[torch]`` extra at runtime), configured by the ``mlp_*`` knobs. The Python
# model is configured by an immutable readout *spec*; this enum is its edge surface (design §6 / App. A).
ReadoutKind = Literal["linear", "rff", "mlp"]

# RFF bandwidth γ: a positive float, or the median heuristic (``"median"``); ``None`` ⇒ the model
# default ("median"). The ``gt=0`` bound applies only to the float member of the union.
GammaField = Annotated[float, Field(gt=0)] | Literal["median"] | None


def _validate_readout_fields(
    *,
    readout: ReadoutKind | None,
    ridge: Any,
    rff_features: int | None,
    rff_gamma: Any,
    mlp_hidden: int | None,
    mlp_weight_decay: float | None,
    mlp_lr: float | None,
    mlp_max_epochs: int | None,
    mlp_patience: int | None,
) -> None:
    """Reject a rung's params when a different rung (or none) is selected — no silent no-op at the edge.

    Mirrors the guards in ``juniper_recurrence._readout.build_lmu_regressor`` so the HTTP edge rejects a
    misconfiguration with 422 up front, identically to the CLI and the shared translation point.
    """
    if readout != "rff" and (rff_features is not None or rff_gamma is not None):
        raise ValueError("rff_features / rff_gamma are only valid when readout='rff'")
    if readout != "mlp" and any(v is not None for v in (mlp_hidden, mlp_weight_decay, mlp_lr, mlp_max_epochs, mlp_patience)):
        raise ValueError("mlp_hidden / mlp_weight_decay / mlp_lr / mlp_max_epochs / mlp_patience are only valid when readout='mlp'")
    if readout == "mlp" and ridge is not None:
        raise ValueError("ridge is not applicable to readout='mlp' (the MLP regularises via weight decay; set mlp_weight_decay)")


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
    ridge: RidgeField = None
    readout: ReadoutKind | None = None
    rff_features: int | None = Field(default=None, ge=1)
    rff_gamma: GammaField = None
    mlp_hidden: int | None = Field(default=None, ge=1)
    mlp_weight_decay: float | None = Field(default=None, ge=0)
    mlp_lr: float | None = Field(default=None, gt=0)
    mlp_max_epochs: int | None = Field(default=None, ge=1)
    mlp_patience: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_readout(self) -> TrainRequest:
        _validate_readout_fields(
            readout=self.readout,
            ridge=self.ridge,
            rff_features=self.rff_features,
            rff_gamma=self.rff_gamma,
            mlp_hidden=self.mlp_hidden,
            mlp_weight_decay=self.mlp_weight_decay,
            mlp_lr=self.mlp_lr,
            mlp_max_epochs=self.mlp_max_epochs,
            mlp_patience=self.mlp_patience,
        )
        return self


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
    ridge: RidgeField = None
    readout: ReadoutKind | None = None
    rff_features: int | None = Field(default=None, ge=1)
    rff_gamma: GammaField = None
    mlp_hidden: int | None = Field(default=None, ge=1)
    mlp_weight_decay: float | None = Field(default=None, ge=0)
    mlp_lr: float | None = Field(default=None, gt=0)
    mlp_max_epochs: int | None = Field(default=None, ge=1)
    mlp_patience: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_readout(self) -> CrossValRequest:
        _validate_readout_fields(
            readout=self.readout,
            ridge=self.ridge,
            rff_features=self.rff_features,
            rff_gamma=self.rff_gamma,
            mlp_hidden=self.mlp_hidden,
            mlp_weight_decay=self.mlp_weight_decay,
            mlp_lr=self.mlp_lr,
            mlp_max_epochs=self.mlp_max_epochs,
            mlp_patience=self.mlp_patience,
        )
        return self


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
