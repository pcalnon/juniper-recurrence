"""The ``juniper-recurrence-model`` range is one range: ``dependencies``, ``[torch]``, ``[bench-torch]`` and the image lock agree.

0.7.0 shipped ``[torch]`` at ``>=0.4.0,<0.5.0`` against a base of ``>=0.5.0,<0.6.0`` -- disjoint, so
``pip install "juniper-recurrence[torch]==0.7.0"`` could not resolve, and no lane installs the extra at a
pinned version. A floor raise edits four sites; this fails when one is missed.
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
