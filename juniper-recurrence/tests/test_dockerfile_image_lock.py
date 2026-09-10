"""
Pin the container image's dependency-lock contract.

Before ``requirements.lock`` existed the Dockerfile ran a bare ``pip install ".[observability]"``,
so the image's contents were whatever PyPI served at build time: two builds of the same commit
could ship different dependency versions, and a bad upstream release reached the image with
nothing pinned to stop it. Each test below fails on one way back to that state.

None of them build an image; ``publish-image.yml``'s pull_request arm does that on both arches and
runs ``util/check_image_cpu_only.py`` inside the result with ``EXPECT_TORCH=absent``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
DOCKERFILE = APP / "Dockerfile"
LOCK = APP / "requirements.lock"
PYPROJECT = APP / "pyproject.toml"
PUBLISH_WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
LOCK_REL = "juniper-recurrence/requirements.lock"
# The image is CPU-only by design (plan D-5) and the LMU memory is numpy-only, so torch is
# never in scope for the resolution -- which is why this repo cannot exhibit the
# CUDA-by-re-resolution shape that cost juniper-cascor-worker its 2026-09-07 image.
CUDA_STACK_PREFIXES = ("nvidia-", "triton==", "cuda-")


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _run_instructions() -> list[str]:
    """Every RUN instruction with continuation lines joined and shell quotes dropped."""
    joined = re.sub(r"\\\n\s*", " ", _dockerfile_text())
    return [line.replace('"', "").replace("'", "") for line in joined.splitlines() if line.startswith("RUN ")]


def _pins() -> list[str]:
    return [line for line in LOCK.read_text(encoding="utf-8").splitlines() if line and not line.startswith((" ", "\t", "#"))]


def _pinned_names() -> set[str]:
    return {p.split("==", 1)[0].strip().lower().replace("_", "-") for p in _pins() if "==" in p}


class TestImageInstallsTheLock:
    def test_lock_exists_and_has_pins(self):
        assert LOCK.is_file(), f"{LOCK_REL} is missing -- the image would resolve from PyPI at build time"
        assert _pins(), f"{LOCK_REL} contains no pins"

    def test_dockerfile_installs_the_lock(self):
        installs = [r for r in _run_instructions() if "-r requirements.lock" in r]
        assert len(installs) == 1, f"expected exactly one `pip install -r requirements.lock`, found {installs}"

    def test_dockerfile_does_not_resolve_the_app_extras_from_pypi(self):
        """The regression: a bare ``pip install ".[observability]"`` re-resolves everything the lock pinned."""
        offenders = [r for r in _run_instructions() if re.search(r"pip install(?!.*--no-deps).*\.\[", r)]
        assert offenders == [], f"the app must be installed with --no-deps against the lock, not by resolving its extras: {offenders}"

    def test_app_is_installed_without_deps_and_checked(self):
        installs = [r for r in _run_instructions() if "--no-deps" in r and " ." in r]
        assert len(installs) == 1, f"expected exactly one `pip install --no-deps .`, found {installs}"
        assert "pip check" in installs[0], "`pip check` must gate the builder: it is what catches a lock that has drifted from pyproject.toml"

    def test_lock_is_installed_before_the_app_source_is_copied(self):
        """Layer caching: a source-only change must not re-resolve dependencies."""
        text = _dockerfile_text()
        lock_install = text.index("-r requirements.lock")
        source_copy = text.index("COPY juniper_recurrence/")
        assert lock_install < source_copy, "COPY requirements.lock + install must precede COPY of the app source"


class TestLockMatchesTheContract:
    def test_lock_pins_no_torch_and_no_cuda_stack(self):
        """``check_image_cpu_only.py`` runs against this image with EXPECT_TORCH=absent."""
        offenders = [p for p in _pins() if p.startswith("torch==") or p.startswith(CUDA_STACK_PREFIXES)]
        assert offenders == [], f"{LOCK_REL} must pin neither torch nor any CUDA-stack package: {offenders}"

    def test_lock_covers_every_declared_dependency(self):
        """A dependency added to pyproject but never re-locked would otherwise fail at import, not at build."""
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        declared = data["project"]["dependencies"] + data["project"].get("optional-dependencies", {}).get("observability", [])
        name_re = re.compile(r"^[A-Za-z0-9_.-]+")
        required = {name_re.match(d).group(0).lower().replace("_", "-") for d in declared if name_re.match(d)}
        missing = sorted(required - _pinned_names())
        assert missing == [], f"{LOCK_REL} is missing declared dependencies {missing} -- regenerate with the recipe in its header"

    def test_lock_header_carries_the_regeneration_recipe(self):
        """The lock is hand-headered because uv overwrites the header with its own command line."""
        header = LOCK.read_text(encoding="utf-8").split("\n")
        header = "\n".join(line for line in header if line.startswith("#"))
        assert "uv pip compile" in header, f"{LOCK_REL}'s header must carry the regeneration command"
        assert "--extra observability" in header
        assert "--python-version 3.13" in header, "the recipe must pin the interpreter the image runs (python:3.13-slim)"


class TestPublishWorkflowWatchesTheLock:
    def test_paths_filter_covers_the_lock(self):
        data = yaml.safe_load(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))
        # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
        data["on"] = data.pop(True, data.get("on"))
        paths = data["on"]["pull_request"]["paths"]
        assert LOCK_REL in paths, f"{LOCK_REL!r} missing from the pull_request paths filter: {paths}"
