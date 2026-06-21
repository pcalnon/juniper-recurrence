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
import sys
from collections.abc import Sequence

from juniper_recurrence._version import __version__


def _ridge_arg(value: str) -> float | str:
    """Parse the ``--ridge`` CLI value: the literal ``"gcv"`` or a non-negative float (DP-3 P1)."""
    if value == "gcv":
        return "gcv"
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

    train = subparsers.add_parser("train", help="Headless: fit the LMU on a dataset and print metrics.")
    train.add_argument("--dataset", default=None, help="Dataset id to train on.")
    train.add_argument("--name", default=None, help="Dataset name (uses the latest version).")
    train.add_argument("--generator", default=None, help="Generator to create a dataset from (e.g. equities_seq).")
    train.add_argument("--split", default="train", help="Split to train on (train/test/full; default: train).")
    train.add_argument("--d", type=int, default=None, help="LMU memory order (default: settings.default_d).")
    train.add_argument("--theta", type=float, default=None, help="LMU window length θ (default: data-driven).")
    train.add_argument("--ridge", type=_ridge_arg, default=None, help="Readout L2 penalty: a float or 'gcv' for closed-form GCV selection (default: settings.default_ridge).")
    train.add_argument("--out", default=None, help="Path to save the trained model (.npz) via LMUSerializer.")

    return parser


def _serve(args: argparse.Namespace) -> int:
    """Run ``uvicorn`` against the module-level app, honoring host/port overrides."""
    import uvicorn

    from juniper_recurrence.settings import Settings

    settings = Settings()
    host = args.host or settings.host
    port = args.port or settings.port
    # Import string (not the app object) so uvicorn owns process/worker lifecycle.
    uvicorn.run("juniper_recurrence.app:app", host=host, port=port)
    return 0


def _train(args: argparse.Namespace) -> int:
    """Headless train: load a 3-D NPZ, fit ``LMURegressor``, print metrics, persist."""
    from juniper_recurrence_model import LMURegressor, LMUSerializer

    from juniper_recurrence.data import load_sequence_data
    from juniper_recurrence.settings import Settings

    if not (args.dataset or args.name or args.generator):
        print("error: train requires one of --dataset / --name / --generator", file=sys.stderr)
        return 2

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
    ridge = args.ridge if args.ridge is not None else settings.default_ridge

    model = LMURegressor(d=d, theta=theta, ridge=ridge)
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

    if args.command == "serve":
        return _serve(args)
    if args.command == "train":
        return _train(args)

    # ``required=True`` on the subparser makes this unreachable; kept as a guard.
    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
