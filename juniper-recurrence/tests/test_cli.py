"""CLI tests for ``juniper-recurrence`` (WS-4b PR-1, plan §11 / §12).

Exercises the ``serve`` wiring without binding a socket: ``uvicorn.run`` is
monkeypatched so the test asserts the import string + resolved host/port. The
headless ``train`` subcommand arrives in PR-2.
"""

from __future__ import annotations

import uvicorn

from juniper_recurrence import main as cli


def test_build_parser_has_serve_subcommand():
    parser = cli._build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.host is None
    assert args.port is None


def test_serve_invokes_uvicorn_with_default_host_port(monkeypatch):
    captured = {}

    def fake_run(target, **kwargs):
        captured["target"] = target
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    rc = cli.main(["serve"])

    assert rc == 0
    assert captured["target"] == "juniper_recurrence.app:app"
    assert captured["host"] == "0.0.0.0"  # default (env cleared by conftest)
    assert captured["port"] == 8210


def test_serve_honors_cli_overrides(monkeypatch):
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: captured.update({"target": target, **kw}))

    rc = cli.main(["serve", "--host", "127.0.0.1", "--port", "9000"])

    assert rc == 0
    assert captured == {"target": "juniper_recurrence.app:app", "host": "127.0.0.1", "port": 9000}


def test_serve_honors_env_host_port(monkeypatch):
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: captured.update(kw))
    monkeypatch.setenv("JUNIPER_RECURRENCE_HOST", "10.0.0.5")
    monkeypatch.setenv("JUNIPER_RECURRENCE_PORT", "8299")

    cli.main(["serve"])

    assert captured["host"] == "10.0.0.5"
    assert captured["port"] == 8299


def test_version_flag_exits_zero(capsys):
    try:
        cli.main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover
        raise AssertionError("--version should raise SystemExit")
    assert "juniper-recurrence" in capsys.readouterr().out


def test_no_command_errors(capsys):
    try:
        cli.main([])
    except SystemExit as exc:
        assert exc.code == 2  # argparse: missing required subcommand
    else:  # pragma: no cover
        raise AssertionError("missing subcommand should raise SystemExit")
