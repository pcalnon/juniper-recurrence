"""Rung 2b — the optional torch MLP readout (DP-3 P3).

Isolated in its own module so the base package import stays **numpy-only / dependency-free**:
``torch`` is imported lazily inside the methods, never at module load. The base test job (no
``[torch]`` extra) therefore never executes this code, so it is ``omit``-ed from the base
``--cov-fail-under`` gate (see ``pyproject.toml``); a separate optional torch CI job exercises it.

Design-of-record: ``notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md`` §4 (Rung 2b),
ratified GO 2026-06-23 (``notes/JUNIPER_DECISIONS_RATIFIED_2026-06-23.md`` D5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from juniper_recurrence_model.readouts import _standardize_fit

_LAYER_NAMES = ("fc1", "fc2", "head")
_ARRAY_KEYS = tuple(f"{name}_{p}" for name in _LAYER_NAMES for p in ("weight", "bias"))


def _require_torch() -> Any:
    """Lazy-import torch (Rung 2b only); raise a helpful error if the ``[torch]`` extra is absent."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - only reachable without the extra
        raise ModuleNotFoundError("the 'mlp' readout (DP-3 Rung 2b) requires torch; install the extra: pip install 'juniper-recurrence-model[torch]'") from exc
    return torch


def _build_net(torch: Any, hidden: int, p: int, k: int, out: int) -> Any:
    """A fixed 2-hidden-layer GELU trunk over standardized ``M`` + a linear head over ``[trunk | extra]``.

    ``extra`` (the ``target_dt`` linear side-channel) and the head bias enter **after** the
    nonlinearity (design boundary D-WS4-2): ``y = head([GELU(fc2(GELU(fc1(zM)))) | extra])``.
    """
    nn = torch.nn

    class _Net(nn.Module):  # noqa: N801 - local module class
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(p, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden + k, out)
            self.act = nn.GELU()

        def forward(self, z_m: Any, ext: Any) -> Any:
            r = self.act(self.fc2(self.act(self.fc1(z_m))))
            return self.head(torch.cat([r, ext], dim=1))

    return _Net()


class MLPReadout:
    """Rung 2b — optional torch MLP readout: ``standardize(M) → GELU MLP (h→h) → linear head over [trunk | extra]``.

    Targets are standardized internally for training stability and un-standardized at predict. Training
    is **CPU-only, single-threaded, float32, deterministic** (``use_deterministic_algorithms(True)``),
    so an in-process save→load→predict round-trip is bit-exact within a machine (no cross-machine claim
    — gated by the ``MLPReadout`` conformance subclass). State persists as **named numpy arrays** (never
    ``torch.save``; the serializer loads with ``allow_pickle=False``). Early stopping uses the
    ``M_val``/``y_val`` the caller supplies (``LMURegressor`` plumbs them from its ``X_val``/``y_val``);
    with no validation data the readout trains for ``max_epochs``.
    """

    kind: ClassVar[str] = "mlp"

    def __init__(self, hidden: int = 128, *, weight_decay: float = 1e-4, lr: float = 1e-3, max_epochs: int = 200, patience: int = 20) -> None:
        self.hidden = int(hidden)
        self.weight_decay = float(weight_decay)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.n_epochs_: int = 0  # epochs actually trained (LMURegressor reads this for TrainResult.n_epochs — A2)
        self._arrays: dict[str, np.ndarray] | None = None  # the fitted layer weights/biases (float32)
        self._dims: tuple[int, int, int] | None = None  # (p, k, out)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._arrays is not None

    @property
    def coef(self) -> np.ndarray | None:
        return None  # nonlinear: no single linear coefficient vector -> model._coef is None

    @staticmethod
    def _extract(net: Any) -> dict[str, np.ndarray]:
        """Flatten the module's weights/biases to named float32 numpy arrays (the serialization unit)."""
        return {f"{name}_{p}": getattr(getattr(net, name), p).detach().cpu().numpy().astype(np.float32) for name in _LAYER_NAMES for p in ("weight", "bias")}

    def fit(self, M: np.ndarray, extra: np.ndarray, y: np.ndarray, *, M_val: np.ndarray | None = None, extra_val: np.ndarray | None = None, y_val: np.ndarray | None = None, random_seed: int | None = None) -> None:
        torch = _require_torch()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.manual_seed(0 if random_seed is None else int(random_seed))  # seed BEFORE module init (weight init)

        M = np.asarray(M, dtype=float)
        extra = np.asarray(extra, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        mean, std = _standardize_fit(M)
        y_std = y.std(axis=0)
        # Store standardization stats as float32: save_state persists them as float32 and the predict
        # path casts to float32 tensors anyway, so float64 stats would make a reloaded readout diverge
        # (breaking the bit-exact round-trip). Float32 here keeps original == reloaded exactly.
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)
        self._y_mean = y.mean(axis=0).astype(np.float32)
        self._y_std = np.where(y_std > 0.0, y_std, 1.0).astype(np.float32)  # constant-target guard (no 0/0)
        p, k, out = M.shape[1], extra.shape[1], y.shape[1]
        self._dims = (p, k, out)

        def _f32(a: np.ndarray) -> Any:
            return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float32)

        z_m = _f32((M - self._mean) / self._std)
        ext = _f32(extra)
        yt = _f32((y - self._y_mean) / self._y_std)
        has_val = M_val is not None and y_val is not None
        if has_val:
            yv = np.asarray(y_val, dtype=float)
            if yv.ndim == 1:
                yv = yv[:, None]
            exv = np.asarray(extra_val, dtype=float) if extra_val is not None else np.zeros((np.asarray(M_val).shape[0], k))
            z_mv = _f32((np.asarray(M_val, dtype=float) - self._mean) / self._std)
            extv = _f32(exv)
            yvt = _f32((yv - self._y_mean) / self._y_std)

        net = _build_net(torch, self.hidden, p, k, out)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        best_arrays: dict[str, np.ndarray] | None = None
        best_val = float("inf")
        stale = 0
        for epoch in range(self.max_epochs):
            net.train()
            opt.zero_grad()
            loss_fn(net(z_m, ext), yt).backward()
            opt.step()
            self.n_epochs_ = epoch + 1
            if not has_val:
                continue
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(net(z_mv, extv), yvt))
            if v < best_val * (1.0 - 1e-4):  # require a >0.01% relative improvement (robust min_delta)
                best_val, stale, best_arrays = v, 0, self._extract(net)
            else:
                stale += 1
                if stale >= self.patience:
                    break
        net.eval()
        self._arrays = best_arrays if best_arrays is not None else self._extract(net)

    def _load_net(self, torch: Any) -> Any:
        """Rebuild the module from ``_dims`` and copy the fitted arrays in (eval mode, deterministic)."""
        p, k, out = self._dims  # type: ignore[misc]
        net = _build_net(torch, self.hidden, p, k, out)
        with torch.no_grad():
            for name in _LAYER_NAMES:
                getattr(net, name).weight.copy_(torch.tensor(self._arrays[f"{name}_weight"]))  # type: ignore[index]
                getattr(net, name).bias.copy_(torch.tensor(self._arrays[f"{name}_bias"]))  # type: ignore[index]
        net.eval()
        return net

    def predict(self, M: np.ndarray, extra: np.ndarray) -> np.ndarray:
        if self._arrays is None:
            raise RuntimeError("readout is not fitted")
        torch = _require_torch()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        net = self._load_net(torch)
        with torch.no_grad():
            z_m = torch.tensor((np.asarray(M, dtype=float) - self._mean) / self._std, dtype=torch.float32)
            ext = torch.tensor(np.asarray(extra, dtype=float), dtype=torch.float32)
            pred = net(z_m, ext).cpu().numpy().astype(float)
        return pred * self._y_std + self._y_mean  # un-standardize

    def save_state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._arrays is None:
            raise RuntimeError("cannot serialize an unfitted readout")
        p, k, out = self._dims  # type: ignore[misc]
        arrays = {
            **self._arrays,
            "mean": self._mean.astype(np.float32),  # type: ignore[union-attr]
            "std": self._std.astype(np.float32),  # type: ignore[union-attr]
            "y_mean": self._y_mean.astype(np.float32),  # type: ignore[union-attr]
            "y_std": self._y_std.astype(np.float32),  # type: ignore[union-attr]
        }
        descriptor = {
            "kind": self.kind,
            "hidden": self.hidden,
            "p": int(p),
            "k": int(k),
            "out": int(out),
            "weight_decay": self.weight_decay,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }
        return arrays, descriptor

    @classmethod
    def from_state(cls, arrays: dict[str, np.ndarray], descriptor: dict[str, Any]) -> MLPReadout:
        ro = cls(
            hidden=int(descriptor["hidden"]),
            weight_decay=float(descriptor.get("weight_decay", 1e-4)),
            lr=float(descriptor.get("lr", 1e-3)),
            max_epochs=int(descriptor.get("max_epochs", 200)),
            patience=int(descriptor.get("patience", 20)),
        )
        ro._dims = (int(descriptor["p"]), int(descriptor["k"]), int(descriptor["out"]))
        ro._mean = arrays["mean"]
        ro._std = arrays["std"]
        ro._y_mean = arrays["y_mean"]
        ro._y_std = arrays["y_std"]
        ro._arrays = {key: arrays[key] for key in _ARRAY_KEYS}
        return ro


@dataclass(frozen=True)
class MLPReadoutSpec:
    """Immutable spec for :class:`MLPReadout` (Rung 2b; requires the ``[torch]`` extra)."""

    hidden: int = 128
    weight_decay: float = 1e-4
    lr: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    kind: ClassVar[str] = "mlp"

    def make(self) -> MLPReadout:
        return MLPReadout(hidden=self.hidden, weight_decay=self.weight_decay, lr=self.lr, max_epochs=self.max_epochs, patience=self.patience)
