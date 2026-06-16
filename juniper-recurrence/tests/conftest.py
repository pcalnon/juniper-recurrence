"""Shared pytest fixtures for the juniper-recurrence app test suite.

The autouse fixture clears ambient ``JUNIPER_*`` environment variables so every
test sees the documented defaults regardless of the developer's shell or a stray
``.env`` export — settings, app-assembly, and CLI tests all rely on this.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_juniper_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("JUNIPER_"):
            monkeypatch.delenv(key, raising=False)
    yield
