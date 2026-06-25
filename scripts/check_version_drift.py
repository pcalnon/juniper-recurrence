#!/usr/bin/env python3
"""Version-drift lint for the juniper-recurrence monorepo (audit CI-06).

Catches the failure class behind audit findings DOC-05 / CI-01 / DOC-01: a package's
``_version.py`` disagreeing with its CHANGELOG and the root ``AGENTS.md`` version table.

For each published package (app / model / client) it enforces, as **hard** invariants
(always true at commit time, in-repo only -- no network, no install):

  1. ``_version.py`` ``__version__``  ==  the top released ``## [X.Y.Z]`` CHANGELOG heading
     (the ``## [Unreleased]`` section is skipped).
  2. ``_version.py``                  ==  the package's version cell in the root AGENTS.md
     sub-package table.

Plus one root invariant:

  3. Root ``AGENTS.md`` ``**Version**:`` header  ==  the **app** package version (the flagship
     published artifact; the header drifted to 0.1.1 while the app shipped 0.2.0 -- DOC-01).

The git tag is checked **directionally and gracefully**: the latest ``<prefix>vX.Y.Z`` tag must
not be *ahead* of ``_version.py`` (a shipped tag the code fell behind = a regression). It is
**skipped** when no tags are visible (a shallow CI checkout has none) and a tag *behind*
``_version.py`` is fine -- that is the normal "version bumped, release not yet cut" window, so
this never makes a release-prep PR flake.

Exit 0 = all agree (tag check may be skipped/informational); exit 1 = a hard mismatch.
Pure stdlib; safe to run from anywhere (locates the repo root from ``__file__``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# (repo subdirectory, import-package dir, git tag prefix)
PACKAGES: list[tuple[str, str, str]] = [
    ("juniper-recurrence", "juniper_recurrence", "juniper-recurrence-v"),
    ("juniper-recurrence-model", "juniper_recurrence_model", "juniper-recurrence-model-v"),
    ("juniper-recurrence-client", "juniper_recurrence_client", "juniper-recurrence-client-v"),
]
APP_DIR = "juniper-recurrence"  # the flagship package the root AGENTS.md **Version** tracks

_SEMVER = r"\d+\.\d+\.\d+"
_VERSION_RE = re.compile(r'__version__\s*=\s*["\'](' + _SEMVER + r')["\']')
_CHANGELOG_RE = re.compile(r"^##\s*\[(" + _SEMVER + r")\]", re.MULTILINE)
_AGENTS_HEADER_RE = re.compile(r"^\*\*Version\*\*:\s*(" + _SEMVER + r")", re.MULTILINE)


def _repo_root() -> Path:
    """Repo root = parent of the ``scripts/`` dir holding this file."""
    return Path(__file__).resolve().parent.parent


def _read_version_py(root: Path, pkg_dir: str, import_dir: str) -> str | None:
    path = root / pkg_dir / import_dir / "_version.py"
    if not path.is_file():
        return None
    m = _VERSION_RE.search(path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _read_changelog_top(root: Path, pkg_dir: str) -> str | None:
    path = root / pkg_dir / "CHANGELOG.md"
    if not path.is_file():
        return None
    # First ``## [X.Y.Z]`` heading -- ``## [Unreleased]`` is not semver so it is skipped naturally.
    m = _CHANGELOG_RE.search(path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _read_agents(root: Path) -> str:
    path = root / "AGENTS.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _agents_header_version(agents: str) -> str | None:
    m = _AGENTS_HEADER_RE.search(agents)
    return m.group(1) if m else None


def _agents_table_version(agents: str, pkg_name: str) -> str | None:
    """Version cell from the AGENTS.md sub-package table row naming ``pkg_name`` in backticks.

    Matches the backtick-delimited package name exactly so ``juniper-recurrence`` does not also
    match the ``juniper-recurrence-model`` row.
    """
    needle = "`" + pkg_name + "`"
    for line in agents.splitlines():
        if not line.lstrip().startswith("|") or needle not in line:
            continue
        versions = re.findall(_SEMVER, line)
        if versions:
            return versions[-1]  # the version cell is the last semver-looking token in the row
    return None


def _latest_tag_version(root: Path, prefix: str) -> str | None:
    """Newest ``<prefix>X.Y.Z`` tag version, or ``None`` if no tags are visible."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "tag", "--list", f"{prefix}*", "--sort=-v:refname"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        m = re.search(_SEMVER, line)
        if m:
            return m.group(0)
    return None


def _as_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def main() -> int:
    root = _repo_root()
    agents = _read_agents(root)
    errors: list[str] = []
    notes: list[str] = []

    app_version: str | None = None

    for pkg_dir, import_dir, tag_prefix in PACKAGES:
        version = _read_version_py(root, pkg_dir, import_dir)
        changelog = _read_changelog_top(root, pkg_dir)
        table = _agents_table_version(agents, pkg_dir)

        if pkg_dir == APP_DIR:
            app_version = version

        if version is None:
            errors.append(f"{pkg_dir}: could not read __version__ from _version.py")
            continue

        # (1) _version.py == top CHANGELOG heading
        if changelog is None:
            errors.append(f"{pkg_dir}: no released '## [X.Y.Z]' heading found in CHANGELOG.md")
        elif changelog != version:
            errors.append(f"{pkg_dir}: _version.py {version} != top CHANGELOG heading {changelog}")

        # (2) _version.py == AGENTS.md sub-package table row
        if table is None:
            errors.append(f"{pkg_dir}: no version cell for `{pkg_dir}` in the AGENTS.md table")
        elif table != version:
            errors.append(f"{pkg_dir}: _version.py {version} != AGENTS.md table version {table}")

        # (tag) directional + graceful
        tag = _latest_tag_version(root, tag_prefix)
        if tag is None:
            notes.append(f"{pkg_dir}: no '{tag_prefix}*' tag visible -- tag check skipped (shallow checkout?)")
        elif _as_tuple(tag) > _as_tuple(version):
            errors.append(f"{pkg_dir}: latest tag {tag_prefix}{tag} is AHEAD of _version.py {version}")
        elif _as_tuple(tag) < _as_tuple(version):
            notes.append(f"{pkg_dir}: _version.py {version} leads latest tag {tag} (release not yet cut) -- OK")
        else:
            notes.append(f"{pkg_dir}: {version} == _version.py == CHANGELOG == AGENTS == tag")

    # (3) root AGENTS.md **Version** header == app version
    header = _agents_header_version(agents)
    if header is None:
        errors.append("AGENTS.md: no '**Version**:' header found")
    elif app_version is not None and header != app_version:
        errors.append(f"AGENTS.md **Version** {header} != app _version.py {app_version}")

    for n in notes:
        print(f"  ok   {n}")
    if errors:
        print("\nVersion drift detected:", file=sys.stderr)
        for e in errors:
            print(f"  FAIL {e}", file=sys.stderr)
        return 1
    print("\nVersion consistency: OK (no drift).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
