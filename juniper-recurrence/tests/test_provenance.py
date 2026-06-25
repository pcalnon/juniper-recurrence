"""Tests for ``juniper_recurrence.provenance`` (build-info accessors, OBS-03).

The accessors read ``JUNIPER_RECURRENCE_GIT_SHA`` / ``JUNIPER_RECURRENCE_BUILD_DATE``
(stamped into the image by the Dockerfile) and return ``None`` outside a
provenance-stamped build. Mirrors the cross-repo build-provenance standard
(juniper-ml ``notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md``); see the data sibling's
``juniper_data.provenance``.
"""

from __future__ import annotations

import pytest

from juniper_recurrence import provenance


@pytest.mark.parametrize(
    ("env_var", "accessor"),
    [
        ("JUNIPER_RECURRENCE_GIT_SHA", provenance.git_sha),
        ("JUNIPER_RECURRENCE_BUILD_DATE", provenance.build_date),
    ],
)
def test_returns_value_when_env_set(monkeypatch: pytest.MonkeyPatch, env_var: str, accessor) -> None:
    monkeypatch.setenv(env_var, "deadbeef")
    assert accessor() == "deadbeef"


@pytest.mark.parametrize(
    ("env_var", "accessor"),
    [
        ("JUNIPER_RECURRENCE_GIT_SHA", provenance.git_sha),
        ("JUNIPER_RECURRENCE_BUILD_DATE", provenance.build_date),
    ],
)
def test_returns_none_when_env_absent(monkeypatch: pytest.MonkeyPatch, env_var: str, accessor) -> None:
    monkeypatch.delenv(env_var, raising=False)
    assert accessor() is None


@pytest.mark.parametrize(
    ("env_var", "accessor"),
    [
        ("JUNIPER_RECURRENCE_GIT_SHA", provenance.git_sha),
        ("JUNIPER_RECURRENCE_BUILD_DATE", provenance.build_date),
    ],
)
def test_empty_string_coerced_to_none(monkeypatch: pytest.MonkeyPatch, env_var: str, accessor) -> None:
    # A bare ``docker build`` leaves the build-arg empty -> the ENV is "" -> treat as absent.
    monkeypatch.setenv(env_var, "")
    assert accessor() is None
