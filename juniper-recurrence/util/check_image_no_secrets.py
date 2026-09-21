#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-recurrence
Application: util
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Assert that a container image carries **no credential-shaped file**, anywhere in the code it
actually ships.

This runs INSIDE the image -- the question is about the image's own filesystem -- so pipe it to
the image's interpreter on stdin (``--entrypoint python`` is needed for images whose Dockerfile
sets an ENTRYPOINT)::

    docker run --rm -i --entrypoint python IMAGE - < util/check_image_no_secrets.py

WHY THIS EXISTS
---------------
Docker's build context does **not** honour ``.gitignore``, only ``.dockerignore``. juniper-deploy's
context held ``secrets/`` with eight live credential files and had no ``.dockerignore`` at all
(2026-09-17). Two defences were added there -- an explicit COPY allowlist and a ``.dockerignore`` --
plus this assertion, on the grounds that **an unasserted defence is a comment**.

The 2026-09-21 sweep then found the defences are weaker than they look:

* A ``COPY src/ ./src/`` is a **DIRECTORY** allowlist, not a file allowlist. It ships everything
  tracked beneath it.
* ``.dockerignore`` patterns are matched by Go ``filepath.Match`` **relative to the context root**,
  so ``cascor_snapshots/`` never matches ``src/cascor_snapshots/``. In juniper-cascor that left 766
  ``.h5`` files -- every one carrying a plaintext multiprocessing authkey -- unexcluded beneath a
  shipping COPY. They stayed out of the published image only because ``.gitignore`` kept them
  untracked and CI builds from ``actions/checkout``.

So the only defence that does not depend on someone getting a pattern right is a check against the
built artifact. That is this file.

WHY IT DOES NOT JUST LIST ``/app``
----------------------------------
Because on three of the five Juniper images that would pass **vacuously**. ``/app`` holds only
runtime directories -- ``['data']`` on juniper-data, ``['logs']`` on juniper-cascor-worker and
juniper-recurrence -- while the actual code lives in ``site-packages``. A top-level ``/app``
iterdir finds nothing to complain about and reports success, having inspected none of the shipped
code. This walks every tree the image actually ships, recursively.

AND WHY IT REFUSES AN EMPTY SCAN
--------------------------------
A scan that finds no roots, or walks zero files, "passes" -- which is the same vacuous-success
class. This exits non-zero if it cannot find at least one Juniper package tree, so a future
refactor that moves the code cannot silently turn the check into a no-op.

Exit codes: 0 clean, 1 credential-shaped file found, 2 the scan itself was not valid.
"""

from __future__ import annotations

import fnmatch
import os
import sys
import sysconfig
from pathlib import Path

# Directory names that should never appear inside an image.
BAD_DIRS = {"secrets", ".git", ".ssh", ".aws", ".gnupg", "private", ".gnupg"}

# Filename globs that should never appear inside an image. `.env.example` and friends are
# templates and are explicitly permitted.
BAD_FILE_GLOBS = (
    ".env", ".env.*", "*.key", "*.pem", "*.p12", "*.pfx", "*.jks", "*.keystore",
    "id_rsa", "id_rsa.*", "id_ecdsa", "id_ed25519", ".netrc", ".npmrc", ".pypirc",
    "credentials", "credentials.*", "*.kdbx",
)
ALLOWED_SUFFIXES = (".example", ".sample", ".template", ".dist")

# Walking these buys nothing and costs a lot; none is shipped Juniper code.
PRUNE_DIRS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"}


def scan_roots() -> list[Path]:
    """Every tree this image actually ships: /app plus each installed Juniper package."""
    roots: list[Path] = []
    app = Path("/app")
    if app.is_dir():
        roots.append(app)

    site = sysconfig.get_paths().get("purelib")
    if site and Path(site).is_dir():
        for child in sorted(Path(site).iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("juniper") or name in {"cascade_correlation", "candidate_unit"}:
                roots.append(child)
    return roots


def is_bad_file(name: str) -> bool:
    if name.endswith(ALLOWED_SUFFIXES):
        return False
    return any(fnmatch.fnmatchcase(name, g) for g in BAD_FILE_GLOBS)


def main() -> int:
    roots = scan_roots()
    if not roots:
        print("::error::no scan root found -- neither /app nor any juniper* package in site-packages")
        print("            the check inspected NOTHING; treat this as a failure, not a pass")
        return 2

    findings: list[str] = []
    files_seen = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
            for d in list(dirnames):
                if d in BAD_DIRS:
                    findings.append(f"{Path(dirpath) / d}/  (directory)")
            for f in filenames:
                files_seen += 1
                if is_bad_file(f):
                    findings.append(str(Path(dirpath) / f))

    print(f"scanned {files_seen} files across {len(roots)} root(s):")
    for r in roots:
        print(f"    {r}")

    if files_seen == 0:
        print("::error::scan walked ZERO files -- the roots exist but are empty, so this check")
        print("            proved nothing. Refusing to report success.")
        return 2

    if findings:
        print(f"\n::error::{len(findings)} credential-shaped path(s) in the image:")
        for f in sorted(findings):
            print(f"    {f}")
        return 1

    print("\n    no credential-shaped file in any shipped tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
