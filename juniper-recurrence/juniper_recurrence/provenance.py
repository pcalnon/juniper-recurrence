"""Build-provenance accessors for juniper-recurrence.

The image's ``Dockerfile`` stamps the source git SHA and an ISO-8601 build
timestamp into the ``JUNIPER_RECURRENCE_GIT_SHA`` / ``JUNIPER_RECURRENCE_BUILD_DATE``
environment variables at build time (from the ``GIT_SHA`` / ``BUILD_DATE``
build-args wired by the deploy Makefile). These accessors read them back so the
Prometheus ``juniper_recurrence_build_info`` Info metric can surface the deployed
source revision -- the foundation for ecosystem stale-image-drift detection
(see juniper-ml ``notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md``).

Both return ``None`` when the service runs outside a provenance-stamped image
(local dev / a bare ``docker build`` leaves the vars empty). Never raise.
"""

from __future__ import annotations

import os


def git_sha() -> str | None:
    """Return the source git SHA baked into the image, or ``None``.

    Read from ``JUNIPER_RECURRENCE_GIT_SHA`` (set by the Dockerfile from a
    build-arg). ``None`` outside a provenance-stamped image.
    """
    return os.environ.get("JUNIPER_RECURRENCE_GIT_SHA") or None


def build_date() -> str | None:
    """Return the ISO-8601 image build timestamp, or ``None``.

    Read from ``JUNIPER_RECURRENCE_BUILD_DATE`` (set by the Dockerfile from a
    build-arg). ``None`` outside a provenance-stamped image.
    """
    return os.environ.get("JUNIPER_RECURRENCE_BUILD_DATE") or None
