"""The TestPyPI import smoke installs exactly the package's own runtime ranges.

``publish-recurrence-model.yml`` verifies a TestPyPI upload in two steps. It installs the wheel with ``--no-deps``
from TestPyPI alone, so a squatted package can never take part in that resolution, and then installs the runtime
dependencies by name from the default index. Those by-name pins are a second copy of ``dependencies``, and nothing
kept the two in step: the smoke installed ``juniper-model-core>=0.1.0,<0.2.0`` while ``dependencies`` required
``>=0.2.0,<0.4.0``, so every publish after the floor rose smoked the wheel against a model-core its own metadata
excludes. These tests fail when the smoke installs another set of packages, or another range, extra or marker for
any of them, than ``dependencies`` declares. ``ci-recurrence-model.yml``'s change filter names the publish
workflow, so a PR that edits only the smoke line runs them too.
"""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_PACKAGE = Path(__file__).resolve().parents[1]
_WORKFLOW = _PACKAGE.parent / ".github" / "workflows" / "publish-recurrence-model.yml"


def _by_name(requirements: list[str], where: str) -> dict[str, tuple[str, frozenset[str], str]]:
    """Each requirement keyed by its canonical name: (specifier, extras, marker), compared as text."""
    out: dict[str, tuple[str, frozenset[str], str]] = {}
    for text in requirements:
        req = Requirement(text)
        name = canonicalize_name(req.name)
        assert name not in out, f"{where}: {name} is named twice in {requirements}"
        out[name] = (str(req.specifier), frozenset(req.extras), str(req.marker or ""))
    return out


def _smoke_requirements() -> list[str]:
    """The by-name runtime installs: the one ``pip install`` line that passes requirements and no options."""
    found: list[list[str]] = []
    for line in _WORKFLOW.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("pip install ") or stripped.endswith("\\"):
            continue
        args = shlex.split(stripped)[2:]
        if args and not any(arg.startswith("-") for arg in args):
            found.append(args)
    assert len(found) == 1, f"{_WORKFLOW.name}: expected one option-free 'pip install' line (the import smoke's runtime deps), found {found}"
    return found[0]


def _runtime_dependencies() -> list[str]:
    return tomllib.loads((_PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]


def test_import_smoke_installs_the_declared_runtime_dependencies():
    smoke = _by_name(_smoke_requirements(), _WORKFLOW.name)
    declared = _by_name(_runtime_dependencies(), "pyproject.toml [project] dependencies")
    assert smoke == declared, f"{_WORKFLOW.name}'s import smoke installs {smoke}, but pyproject.toml declares {declared}; keep the smoke's by-name pins equal to dependencies"


def test_import_smoke_names_model_core():
    """Guard the guard: the comparison above is vacuous if the smoke line is not the one that installs model-core."""
    assert "juniper-model-core" in _by_name(_smoke_requirements(), _WORKFLOW.name)
