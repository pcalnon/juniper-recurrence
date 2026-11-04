"""The ``juniper-recurrence-model`` range is one range: ``dependencies``, ``[torch]``, ``[bench-torch]`` and the image lock agree.

0.7.0 shipped ``[torch]`` at ``>=0.4.0,<0.5.0`` against a base of ``>=0.5.0,<0.6.0`` -- disjoint, so
``pip install "juniper-recurrence[torch]==0.7.0"`` could not resolve, and no lane installs the extra at a
pinned version. A floor raise edits four sites; this fails when one is missed.

The ``juniper-model-core`` floor is pinned here too. ``_folds.build_folds`` passes ``groups=`` to
``walk_forward_folds`` on every ``POST /v1/crossval``, and ``groups=`` first shipped in model-core 0.3.0.
0.8.0 declared ``>=0.2.0``, so an env holding 0.2.0 passed ``pip check`` while every crossval request
answered 500. CI resolves the newest model-core, so no suite runs against the floor; these tests fail if
it admits a release without ``groups=`` again, or if the image lock leaves the range.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = re.compile(r"^juniper-recurrence-model(?:\[(?P<extras>[^\]]*)\])?\s*(?P<spec>[^;]*?)\s*(?:;.*)?$")
_LOCK = re.compile(r"^juniper-recurrence-model==(?P<version>\S+)\s*$", re.MULTILINE)
_MODEL_CORE = re.compile(r"^juniper-model-core(?:\[(?P<extras>[^\]]*)\])?\s*(?P<spec>[^;]*?)\s*(?:;.*)?$")
_MODEL_CORE_LOCK = re.compile(r"^juniper-model-core==(?P<version>\S+)\s*$", re.MULTILINE)
#: Every published juniper-model-core release without ``walk_forward_folds(groups=)`` (PyPI, 2026-11-04): 0.1.0 has no
#: crossval layer at all, and 0.2.0's ``walk_forward_folds`` has neither ``groups`` nor ``**kwargs``.
_MODEL_CORE_WITHOUT_GROUPS = ("0.1.0", "0.2.0")


def _model_spec(requirements: list[str], where: str) -> SpecifierSet:
    found = [m.group("spec") for r in requirements if (m := _MODEL.match(r.strip()))]
    assert len(found) == 1, f"{where}: expected one juniper-recurrence-model requirement, found {found}"
    return SpecifierSet(found[0])


def _project() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_torch_extras_carry_the_base_model_range():
    project = _project()
    base = _model_spec(project["dependencies"], "dependencies")
    extras = project["optional-dependencies"]
    assert _model_spec(extras["torch"], "[torch]") == base
    assert _model_spec(extras["bench-torch"], "[bench-torch]") == base


def test_image_lock_pins_a_model_inside_the_base_range():
    base = _model_spec(_project()["dependencies"], "dependencies")
    lock = (_ROOT / "requirements.lock").read_text(encoding="utf-8")
    pinned = _LOCK.findall(lock)
    assert len(pinned) == 1, f"requirements.lock: expected one juniper-recurrence-model pin, found {pinned}"
    assert Version(pinned[0]) in base, f"requirements.lock pins juniper-recurrence-model=={pinned[0]}, outside {base}"


def _model_core_spec(requirements: list[str], where: str) -> SpecifierSet:
    found = [m.group("spec") for r in requirements if (m := _MODEL_CORE.match(r.strip()))]
    assert len(found) == 1, f"{where}: expected one juniper-model-core requirement, found {found}"
    return SpecifierSet(found[0])


def test_model_core_floor_admits_no_release_without_walk_forward_folds_groups():
    base = _model_core_spec(_project()["dependencies"], "dependencies")
    admitted = [v for v in _MODEL_CORE_WITHOUT_GROUPS if Version(v) in base]
    assert admitted == [], f"dependencies admit juniper-model-core {admitted}, whose walk_forward_folds takes no groups= ({base}); _folds.build_folds passes it on every crossval"


def test_image_lock_pins_a_model_core_inside_the_base_range():
    base = _model_core_spec(_project()["dependencies"], "dependencies")
    lock = (_ROOT / "requirements.lock").read_text(encoding="utf-8")
    pinned = _MODEL_CORE_LOCK.findall(lock)
    assert len(pinned) == 1, f"requirements.lock: expected one juniper-model-core pin, found {pinned}"
    assert Version(pinned[0]) in base, f"requirements.lock pins juniper-model-core=={pinned[0]}, outside {base}"
