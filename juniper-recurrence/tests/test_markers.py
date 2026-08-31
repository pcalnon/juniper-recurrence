"""The `performance` marker is registered and usable (G-17, CLI-experimentation plan §12.2).

Why a test for a marker declaration. An unregistered marker is a **collection error for the
entire suite** here — not a warning on the one test that used it. Two configured gates converge
on that: ``addopts`` carries ``--strict-markers``, and ``filterwarnings`` starts with ``error``;
both act by promoting ``PytestUnknownMarkWarning``, and the observed failure is
``Interrupted: 1 error during collection``. So the failure mode this guards is not "the marker is missing", it is "someone
tidies an unused-looking entry out of ``pyproject.toml`` and the next person to write
``@pytest.mark.performance`` finds the whole suite refusing to collect, with an error that points
at their new test rather than at the deletion".

Nothing carries the marker yet — the plan registers it *for* the micro-benchmarks that lane will
add. That is precisely why the registration needs pinning: an entry with no user is the kind that
gets removed as dead.
"""

from __future__ import annotations

import pytest


@pytest.mark.performance
def test_performance_marker_is_registered_and_collectible():
    """Carrying the marker is itself the assertion.

    Under ``--strict-markers`` this test cannot be collected at all unless ``performance`` is
    registered, so its presence in a passing run proves the registration. The body only has to
    not fail.
    """
    assert True


def test_performance_marker_is_declared_with_a_description(pytestconfig):
    """The registration exists and explains itself.

    ``markers`` entries are ``"name: description"``; a bare name registers fine but tells a
    reader nothing about when to apply it, which is how a marker drifts into meaning whatever
    each author assumed.
    """
    declared = pytestconfig.getini("markers")
    entry = next((m for m in declared if m.split(":")[0].strip() == "performance"), None)
    assert entry is not None, f"`performance` not registered; markers={declared}"
    assert ":" in entry and entry.split(":", 1)[1].strip(), f"`performance` registered without a description: {entry!r}"
