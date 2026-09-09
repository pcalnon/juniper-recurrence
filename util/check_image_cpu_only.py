#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-recurrence
Application: util
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Assert that a container image's Python environment is CPU-only: either it carries
exactly the pinned ``+cpu`` PyTorch build and nothing from the NVIDIA/CUDA stack, or
it carries no torch at all.

This runs INSIDE the image -- the question is about the image's own site-packages --
so pipe it to the image's interpreter on stdin (``--entrypoint python`` is needed only
for images whose Dockerfile sets an ENTRYPOINT, as the worker's does)::

    docker run --rm -i -e EXPECT_TORCH=2.12.0+cpu --entrypoint python IMAGE - \\
        < util/check_image_cpu_only.py

``EXPECT_TORCH`` selects the contract:

``<major>.<minor>.<patch>+cpu``
    torch must import, ``torch.__version__`` must equal this value exactly,
    ``torch.version.cuda`` must be ``None``, and no ``nvidia-*``, ``cuda-*`` or
    ``triton`` distribution may be installed.

``absent``
    torch must not be importable, and no ``nvidia-*`` / ``cuda-*`` / ``triton``
    distribution may be installed (for images that never ship torch).

Why the distribution census is a separate check from the version string
------------------------------------------------------------------------
The 2026-09-07 worker image (``ghcr.io/pcalnon/juniper-cascor-worker:dispatch-3d81f2c``)
shipped ``torch 2.12.1+cu130`` plus ``nvidia-cublas``, ``nvidia-cudnn-cu13``,
``nvidia-nccl-cu13`` and ``triton`` -- 3 GB per Raspberry Pi node of libraries a Pi can
never use -- because the lockfile install re-resolved torch from PyPI after the CPU
wheel had been installed. A version assertion alone would have caught that. It does
NOT catch the vacuous fix: installing the CPU wheel *last* replaces ``torch`` and leaves
every orphaned ``nvidia-*`` / ``triton`` wheel in site-packages, so ``__version__`` reads
``+cpu`` while the image is still 3 GB. The census is what makes "CPU-only" mean what
it says.

Exit codes: ``0`` contract holds, ``1`` contract violated, ``2`` usage error.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
import sys

_CPU_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+\+cpu$")
# The whole CUDA stack as it actually appeared in the 2026-09-07 image: the nvidia-* runtime
# libraries, triton, AND the cuda-* packages (cuda-toolkit, cuda-bindings, cuda-pathfinder).
# The first census forbade only the first two families and would have passed the third.
_FORBIDDEN_PREFIXES = ("nvidia-", "cuda-")
_FORBIDDEN_NAMES = frozenset({"triton"})


def _normalise(name: str) -> str:
    """PEP 503 normalisation, so ``nvidia_cublas`` and ``nvidia-cublas`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distributions() -> set[str]:
    """Normalised names of every distribution visible to this interpreter."""
    names: set[str] = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            names.add(_normalise(name))
    return names


def forbidden_distributions(names: set[str]) -> list[str]:
    """The installed distributions that belong to the CUDA stack."""
    return sorted(n for n in names if n.startswith(_FORBIDDEN_PREFIXES) or n in _FORBIDDEN_NAMES)


def torch_importable() -> bool:
    """Whether ``import torch`` would resolve -- without importing it."""
    return importlib.util.find_spec("torch") is not None


def check(expect: str, names: set[str]) -> list[str]:
    """Return every violation of the ``expect`` contract; an empty list is a pass."""
    problems: list[str] = []
    offenders = forbidden_distributions(names)
    if offenders:
        problems.append("CUDA/NVIDIA distributions are installed: " + ", ".join(offenders))

    if expect == "absent":
        if torch_importable() or "torch" in names:
            problems.append("torch is installed, but EXPECT_TORCH=absent")
        return problems

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only reachable inside a broken image
        problems.append(f"torch failed to import: {exc}")
        return problems

    if torch.__version__ != expect:
        problems.append(f"torch.__version__ is {torch.__version__!r}, expected {expect!r}")
    cuda = getattr(torch.version, "cuda", None)
    if cuda is not None:
        problems.append(f"torch.version.cuda is {cuda!r}; a CPU build reports None")
    return problems


def main() -> int:
    expect = os.environ.get("EXPECT_TORCH", "").strip()
    if not expect or (expect != "absent" and not _CPU_VERSION_RE.match(expect)):
        print(
            f"::error::EXPECT_TORCH must be 'absent' or '<major>.<minor>.<patch>+cpu', got {expect!r}",
            file=sys.stderr,
        )
        return 2

    names = installed_distributions()
    problems = check(expect, names)

    torch_report = "absent"
    if expect != "absent" and torch_importable():
        import torch

        torch_report = f"{torch.__version__} (cuda={getattr(torch.version, 'cuda', None)!r})"
    print(
        f"machine={platform.machine()} python={platform.python_version()} "
        f"torch={torch_report} distributions={len(names)} "
        f"cuda_stack={len(forbidden_distributions(names))} expect={expect}"
    )

    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        print("CPU-only contract VIOLATED", file=sys.stderr)
        return 1
    print("CPU-only contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
