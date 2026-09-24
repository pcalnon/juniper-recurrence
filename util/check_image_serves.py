#!/usr/bin/env python3
"""Assert that a built image SERVES, and that every version it reports is the one it was built as.

Runs on the RUNNER, not inside the image: it drives ``docker``. Stdlib only.

Why this exists
---------------
The publish path already asserts what an image CONTAINS (``check_image_cpu_only.py``,
``check_image_no_secrets.py``) and that the application IMPORTS. Two defects shipped anyway,
because nothing asserted what the running image DOES or what it SAYS it is:

* **An image that imports but reports the wrong version.** ``juniper-cascor-worker:0.6.0``
  imported fine and reported ``__version__ == "0.4.0"``: the literal in ``__init__.py`` was never
  bumped, and two releases shipped it (fixed in 0.6.1). ``juniper-cascor:0.11.0`` answers
  ``/v1/health`` with ``0.11.0`` and stamps ``meta.version: "0.6.0"`` on every enveloped response,
  because the envelope read a hard-coded ``_API_VERSION`` (juniper-cascor#668, fixed by #672).
  An import check passes both.
* **An image that builds, starts and serves nothing.** juniper-deploy's test runner built,
  started and ran zero tests for six months. "It imports" is not "it serves".

So this starts the image exactly as ``docker run IMAGE`` would -- its own entrypoint and
command, no overrides -- and requires, within ``--timeout`` seconds:

1. the installed distribution's metadata version equals ``--expect-version`` (on a release, the
   version in the release TAG, so an image tagged ``X.Y.Z`` cannot carry a package that says
   anything else);
2. ``<--module>.__version__`` equals that metadata version. A module with NO ``__version__`` is a
   failure, not a pass: that is the vacuous pass the class-2 sweep scored for cascor;
3. the LIVENESS endpoint (``--health-path``, default ``/v1/health``) answers 200. Readiness is
   deliberately not asked: a standalone container has no backing services, so a readiness 503 is
   the probe working, not the image failing;
4. the liveness body's ``version`` field equals the metadata version. ``--health-version optional``
   tolerates an ABSENT field (juniper-recurrence's body is ``{"status": "ok"}``); a PRESENT field
   that disagrees always fails;
5. each ``--enveloped-path`` answers with a ``meta.version`` equal to the metadata version.

The endpoints are probed from INSIDE the container (``docker exec … python``), so a service bound
to 127.0.0.1 -- the worker's health server -- is reachable, and no host port can collide.

Exit codes: 0 every check passed; 1 a check failed; 2 a usage or environment error (no docker,
an image that cannot run python). The container is always removed.

Canonical copy: juniper-ml ``notes/JUNIPER_2026-09-05_JUNIPER-ECOSYSTEM_CONTAINER-REGISTRY-PUBLISHING-PLAN.md``
§5.2 records where each image repo's copy lives; keep them identical.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - drives the docker CLI with fixed argv, never a shell
import sys
import time

SEMVER = re.compile(r"^\d+\.\d+\.\d+([.+-][0-9A-Za-z.+-]+)?$")

# Printed by the in-image probe as one JSON line: the metadata version, and the module's
# ``__version__`` (null when the module has none, "<import error>" text when it cannot import).
_VERSION_PROBE = """
import importlib, importlib.metadata as md, json, sys
dist, module = sys.argv[1], sys.argv[2]
out = {"metadata": None, "module_version": None, "module_error": None}
try:
    out["metadata"] = md.version(dist)
except md.PackageNotFoundError as exc:
    out["module_error"] = "no distribution %r: %s" % (dist, exc)
if module:
    try:
        out["module_version"] = getattr(importlib.import_module(module), "__version__", None)
    except Exception as exc:
        out["module_error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""

# GETs one path on 127.0.0.1 inside the container; prints the status, then the body.
_GET_PROBE = """
import sys, urllib.request, urllib.error
url = "http://127.0.0.1:%s%s" % (sys.argv[1], sys.argv[2])
try:
    r = urllib.request.urlopen(url, timeout=4)
    status, body = r.status, r.read()
except urllib.error.HTTPError as e:
    status, body = e.code, e.read()
print(status)
print(body.decode("utf-8", "replace"))
"""


def evaluate(expect: str, metadata: "str | None", module: "str | None", module_version: "str | None", module_error: "str | None", health_status: "int | None", health_body, health_version_required: bool, enveloped: "dict[str, tuple]") -> list:
    """Every failure, as text. Pure: the tests drive it with synthetic observations.

    ``enveloped`` maps each probed path to ``(status, parsed_body)``.
    """
    failures: list = []
    if metadata is None:
        failures.append(f"no installed distribution metadata ({module_error or 'unknown'})")
    elif metadata != expect:
        failures.append(f"installed metadata version {metadata} != expected {expect}")
    if module:
        if module_error and module_version is None:
            failures.append(f"{module} does not import: {module_error}")
        elif module_version is None:
            failures.append(f"{module} has no __version__ -- an absent version is not a match")
        elif module_version != metadata:
            failures.append(f"{module}.__version__ {module_version} != metadata {metadata}")
    if health_status != 200:
        failures.append(f"liveness answered {health_status}, not 200")
    else:
        served = health_body.get("version") if isinstance(health_body, dict) else None
        if served is None:
            if health_version_required:
                failures.append("the liveness body carries no version field")
        elif served != metadata:
            failures.append(f"the liveness body reports version {served} != metadata {metadata}")
    for path, (status, body) in sorted(enveloped.items()):
        meta_version = (body.get("meta") or {}).get("version") if isinstance(body, dict) else None
        if status != 200:
            failures.append(f"{path} answered {status}, not 200")
        elif meta_version is None:
            failures.append(f"{path} carries no meta.version")
        elif meta_version != metadata:
            failures.append(f"{path} meta.version {meta_version} != metadata {metadata}")
    return failures


def _docker(args: list, timeout: int = 60) -> "tuple[int, str]":
    try:
        # docker is resolved from PATH on purpose: CI runners and dev boxes install it in different
        # places. The argv is fixed by this script and never passes through a shell.
        proc = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
    except FileNotFoundError:
        return 127, "docker not found"
    except subprocess.TimeoutExpired:
        return 124, f"docker {args[0]} timed out after {timeout}s"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _get(cid: str, port: int, path: str) -> "tuple[int | None, object]":
    code, out = _docker(["exec", cid, "python", "-c", _GET_PROBE, str(port), path], timeout=20)
    if code != 0 or not out:
        return None, out
    status_line, _, body = out.partition("\n")
    try:
        status = int(status_line.strip())
    except ValueError:
        return None, out
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def main(argv: "list | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Assert an image serves liveness and reports the version it was built as.")
    parser.add_argument("--image", required=True, help="image reference, e.g. ghcr.io/pcalnon/juniper-data@sha256:...")
    parser.add_argument("--dist", required=True, help="distribution whose metadata is the truth, e.g. juniper-data")
    parser.add_argument("--module", default="", help="package whose __version__ must match; omit when the image ships none")
    parser.add_argument("--port", required=True, type=int, help="the liveness port INSIDE the container")
    parser.add_argument("--expect-version", required=True, help="X.Y.Z the image must be: the release tag's version on a release")
    parser.add_argument("--health-path", default="/v1/health")
    parser.add_argument("--health-version", choices=("required", "optional"), default="required", help="whether the liveness body must carry a version field")
    parser.add_argument("--enveloped-path", action="append", default=[], help="a path whose response carries meta.version (repeatable)")
    parser.add_argument("--timeout", type=int, default=120, help="seconds to wait for liveness")
    args = parser.parse_args(argv)

    if not SEMVER.match(args.expect_version):
        print(f"::error::--expect-version {args.expect_version!r} is not X.Y.Z", file=sys.stderr)
        return 2

    code, out = _docker(["run", "--rm", "--entrypoint", "python", args.image, "-c", _VERSION_PROBE, args.dist, args.module], timeout=180)
    if code != 0:
        print(f"::error::could not run python in {args.image}: {out[-400:]}", file=sys.stderr)
        return 2
    try:
        versions = json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        print(f"::error::the version probe printed no JSON: {out[-400:]}", file=sys.stderr)
        return 2
    print(f"  metadata {args.dist} = {versions['metadata']}; {args.module or '(no module)'}.__version__ = {versions['module_version']}")

    code, out = _docker(["run", "-d", args.image], timeout=120)
    if code != 0:
        print(f"::error::the image did not start: {out[-400:]}", file=sys.stderr)
        return 1
    cid = out.splitlines()[-1].strip()
    health_status, health_body, enveloped = None, None, {}
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            running = _docker(["inspect", "-f", "{{.State.Running}}", cid], timeout=20)[1]
            if running != "true":
                print(f"::error::the container exited before liveness answered; last log lines:\n{_docker(['logs', '--tail', '40', cid])[1]}", file=sys.stderr)
                return 1
            health_status, health_body = _get(cid, args.port, args.health_path)
            if health_status == 200:
                break
            time.sleep(3)
        print(f"  {args.health_path} -> {health_status}: {json.dumps(health_body)[:240] if not isinstance(health_body, str) else health_body[:240]}")
        for path in args.enveloped_path:
            enveloped[path] = _get(cid, args.port, path)
            status, body = enveloped[path]
            print(f"  {path} -> {status}: {json.dumps(body)[:240] if not isinstance(body, str) else body[:240]}")
        if health_status != 200:
            print(f"  last log lines:\n{_docker(['logs', '--tail', '40', cid])[1]}")
    finally:
        _docker(["rm", "-f", cid], timeout=60)

    failures = evaluate(args.expect_version, versions["metadata"], args.module or None, versions["module_version"], versions["module_error"], health_status, health_body, args.health_version == "required", enveloped)
    for failure in failures:
        print(f"::error::{args.image}: {failure}", file=sys.stderr)
    if failures:
        return 1
    print(f"  serves {args.health_path} and reports {args.expect_version} on every surface checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
