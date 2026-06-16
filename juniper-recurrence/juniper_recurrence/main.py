"""CLI entrypoint for the juniper-recurrence service (C2 dual-mode).

``juniper-recurrence serve`` launches the FastAPI app under uvicorn (single
worker; in-process state). The headless ``train`` subcommand — load a 3-D NPZ via
the data adapter, fit ``LMURegressor``, print metrics, persist via
``LMUSerializer`` — is added in WS-4b PR-2 alongside the data path; this skeleton
ships ``serve`` only and leaves a clean extension point for it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from juniper_recurrence._version import __version__


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``juniper-recurrence`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="juniper-recurrence",
        description="FastAPI + CLI service for the Δt-native LMU recurrence model.",
    )
    parser.add_argument("--version", action="version", version=f"juniper-recurrence {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{serve}")

    serve = subparsers.add_parser("serve", help="Run the FastAPI service under uvicorn.")
    serve.add_argument("--host", default=None, help="Bind host (defaults to JUNIPER_RECURRENCE_HOST / settings).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (defaults to JUNIPER_RECURRENCE_PORT / settings).")

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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI dispatch entrypoint (``[project.scripts] juniper-recurrence``)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args)

    # ``required=True`` on the subparser makes this unreachable; kept as a guard.
    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
