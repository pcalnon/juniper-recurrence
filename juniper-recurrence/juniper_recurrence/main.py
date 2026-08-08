"""CLI entrypoint for the juniper-recurrence service (C2 dual-mode).

* ``juniper-recurrence serve`` launches the FastAPI app under uvicorn (single
  worker; in-process state).
* ``juniper-recurrence train`` is headless: load a 3-D NPZ via the shared data
  adapter, fit ``LMURegressor``, print the regression metrics, and optionally persist
  the model via ``LMUSerializer``. It reuses the exact ``data.load_sequence_data`` +
  model construction the ``/v1/train`` route uses.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from juniper_recurrence._version import __version__


def _ridge_arg(value: str) -> float | str:
    """Parse the ``--ridge`` CLI value: the literal ``"gcv"`` or a non-negative float (DP-3 P1)."""
    if value == "gcv":
        return "gcv"
    return float(value)


def _gamma_arg(value: str) -> float | str:
    """Parse the ``--rff-gamma`` CLI value: the literal ``"median"`` or a positive float (DP-3 P2c)."""
    if value == "median":
        return "median"
    return float(value)


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``juniper-recurrence`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="juniper-recurrence",
        description="FastAPI + CLI service for the Δt-native LMU recurrence model.",
    )
    parser.add_argument("--version", action="version", version=f"juniper-recurrence {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{serve,train}")

    serve = subparsers.add_parser("serve", help="Run the FastAPI service under uvicorn.")
    serve.add_argument("--host", default=None, help="Bind host (defaults to JUNIPER_RECURRENCE_HOST / settings).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (defaults to JUNIPER_RECURRENCE_PORT / settings).")
    serve.add_argument("--config", default=None, help="Experiment YAML whose service: block overrides env (sets JUNIPER_RECURRENCE_CONFIG_FILE before settings load; Wave 3.3).")

    train = subparsers.add_parser("train", help="Headless: fit the LMU on a dataset and print metrics.")
    train.add_argument("--dataset", default=None, help="Dataset id to train on.")
    train.add_argument("--name", default=None, help="Dataset name (uses the latest version).")
    train.add_argument("--generator", default=None, help="Generator to create a dataset from (e.g. equities_seq).")
    train.add_argument("--split", default="train", help="Split to train on (train/test/full; default: train).")
    train.add_argument("--d", type=int, default=None, help="LMU memory order (default: settings.default_d).")
    train.add_argument("--theta", type=float, default=None, help="LMU window length θ (default: data-driven).")
    train.add_argument("--ridge", type=_ridge_arg, default=None, help="Readout L2 penalty: a float or 'gcv' for closed-form GCV selection (default: settings.default_ridge).")
    train.add_argument("--readout", choices=("linear", "rff", "mlp"), default=None, help="Readout rung: 'linear' (default), 'rff' (nonlinear random Fourier features; DP-3 P2c), or 'mlp' (torch MLP; DP-3 P3 — needs the [torch] extra).")
    train.add_argument("--rff-features", type=int, default=None, help="RFF feature count D when --readout=rff (default: 256).")
    train.add_argument("--rff-gamma", type=_gamma_arg, default=None, help="RFF bandwidth γ when --readout=rff: a positive float or 'median' (default: 'median').")
    train.add_argument("--mlp-hidden", type=int, default=None, help="MLP hidden width when --readout=mlp (default: 128).")
    train.add_argument("--mlp-weight-decay", type=float, default=None, help="MLP Adam weight decay when --readout=mlp (default: 1e-4).")
    train.add_argument("--mlp-lr", type=float, default=None, help="MLP Adam learning rate when --readout=mlp (default: 1e-3).")
    train.add_argument("--mlp-max-epochs", type=int, default=None, help="MLP max training epochs when --readout=mlp (default: 200).")
    train.add_argument("--mlp-patience", type=int, default=None, help="MLP early-stop patience in epochs when --readout=mlp (default: 20).")
    train.add_argument("--out", default=None, help="Path to save the trained model (.npz) via LMUSerializer.")
    train.add_argument("--config", default=None, help="Experiment YAML whose service: block overrides env (sets JUNIPER_RECURRENCE_CONFIG_FILE before settings load; Wave 3.3).")

    return parser


def _serve(args: argparse.Namespace) -> int:
    """Run ``uvicorn`` against the module-level app, honoring host/port overrides."""
    import uvicorn

    from juniper_recurrence.settings import Settings

    settings = Settings()
    host = args.host or settings.host
    port = args.port or settings.port
    # Logging is configured by the app's lifespan (service-core create_app(lifespan=)) when uvicorn
    # starts the app — this CLI path and a direct `uvicorn juniper_recurrence.app:app` are both covered.
    # Import string (not the app object) so uvicorn owns process/worker lifecycle.
    uvicorn.run("juniper_recurrence.app:app", host=host, port=port)
    return 0


# W-11 (CLI experimentation plan SS11 / Wave 3.6): ``train`` seeds its argparse defaults
# from the experiment YAML's ``train:`` block. Precedence: an explicitly-passed CLI flag
# wins; an unset flag (None) falls back to the YAML value; absent both, the existing
# settings/builder defaults apply (SS5.1: CLI > YAML > env-backed settings > defaults).
# main() has already threaded --config into JUNIPER_RECURRENCE_CONFIG_FILE, and the
# Settings source fail-loud-validates the file (SS5.6) before these helpers read it.
_W11_TRAIN_KEYS = ("d", "theta", "ridge", "readout", "rff_features", "rff_gamma", "mlp_hidden", "mlp_weight_decay", "mlp_lr", "mlp_max_epochs", "mlp_patience")


def _experiment_train_overrides() -> dict:
    """Return the experiment YAML's ``train:`` block ({} when no config is threaded)."""
    config_path = os.environ.get("JUNIPER_RECURRENCE_CONFIG_FILE")
    if not config_path:
        return {}
    from pathlib import Path

    import yaml

    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    block = data.get("train") or {}
    unknown = sorted(key for key in block if key not in _W11_TRAIN_KEYS)
    if unknown:
        print(f"warning: experiment train: keys with no CLI counterpart, ignored: {', '.join(unknown)}", file=sys.stderr)
    return {key: value for key, value in block.items() if key in _W11_TRAIN_KEYS}


def _apply_train_overrides(args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    """Seed unset (None) train flags from the YAML ``train:`` block; explicit CLI wins."""
    for key in _W11_TRAIN_KEYS:
        if key in overrides and getattr(args, key, None) is None:
            setattr(args, key, overrides[key])
    return args


def _train(args: argparse.Namespace) -> int:
    """Headless train: load a 3-D NPZ, fit ``LMURegressor``, print metrics, persist."""
    from juniper_recurrence_model import LMUSerializer

    from juniper_recurrence._readout import build_lmu_regressor
    from juniper_recurrence.data import load_sequence_data
    from juniper_recurrence.settings import Settings

    if not (args.dataset or args.name or args.generator):
        print("error: train requires one of --dataset / --name / --generator", file=sys.stderr)
        return 2

    # W-11: YAML train: block seeds any flag the CLI left unset (explicit CLI wins).
    args = _apply_train_overrides(args, _experiment_train_overrides())

    settings = Settings()
    sequence, descriptor = load_sequence_data(
        base_url=settings.juniper_data_url,
        api_key=settings.juniper_data_api_key,
        dataset_id=args.dataset,
        name=args.name,
        generator=args.generator,
        split=args.split,
    )

    d = args.d if args.d is not None else settings.default_d
    theta = args.theta if args.theta is not None else settings.default_theta

    try:
        model = build_lmu_regressor(
            d=d,
            theta=theta,
            readout=args.readout,
            ridge=args.ridge,
            rff_features=args.rff_features,
            rff_gamma=args.rff_gamma,
            mlp_hidden=args.mlp_hidden,
            mlp_weight_decay=args.mlp_weight_decay,
            mlp_lr=args.mlp_lr,
            mlp_max_epochs=args.mlp_max_epochs,
            mlp_patience=args.mlp_patience,
            default_ridge=settings.default_ridge,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = model.fit(sequence.X, sequence.y, **sequence.fit_kwargs())

    print(f"Trained LMURegressor on dataset {descriptor['dataset_id']} (split={descriptor['split']}, windows={descriptor['n_windows']}, F={descriptor['n_features']}).")
    print("Metrics:")
    for key, value in result.final_metrics.items():
        print(f"  {key}: {value:.6f}")

    if args.out:
        LMUSerializer().save(model, args.out)
        print(f"Saved model to {args.out}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI dispatch entrypoint (``[project.scripts] juniper-recurrence``)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "config", None):
        # Wave 3.3: must land before the first Settings() construction (plan SS5.2) --
        # both _serve and _train build Settings() inside their handlers.
        os.environ["JUNIPER_RECURRENCE_CONFIG_FILE"] = args.config

    if args.command == "serve":
        return _serve(args)
    if args.command == "train":
        return _train(args)

    # ``required=True`` on the subparser makes this unreachable; kept as a guard.
    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
