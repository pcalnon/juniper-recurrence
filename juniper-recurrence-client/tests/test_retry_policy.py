"""Regression tests for the HTTP retry policy.

Guards APD-ECO-002: the client must pass ``backoff_jitter`` to urllib3's
``Retry`` so concurrent clients do not retry a failing service in lockstep.
"""

from __future__ import annotations

from juniper_recurrence_client import constants
from juniper_recurrence_client.client import JuniperRecurrenceClient


class TestRetryBackoffJitter:
    """APD-ECO-002: retry schedules must be decorrelated across client instances.

    urllib3 applies jitter as an ABSOLUTE additive term --
    ``backoff_value += random.random() * backoff_jitter`` -- so without it every
    client that trips the same transient outage retries on an identical
    schedule, and a service that is already failing is hit by a synchronised
    herd. The parameter arrived in urllib3 2.0.0, which is the floor this
    package already pins.
    """

    def test_jitter_constant_is_positive(self) -> None:
        # Pin the VALUE, not merely the kwarg's presence: setting it to 0.0
        # leaves the call site looking correct while silently restoring the herd.
        assert constants.DEFAULT_BACKOFF_JITTER > 0

    def test_retry_adapter_carries_the_jitter(self) -> None:
        with JuniperRecurrenceClient(base_url="http://localhost:8211") as client:
            adapter = client.session.get_adapter("http://localhost:8211/")
            assert adapter.max_retries.backoff_jitter == constants.DEFAULT_BACKOFF_JITTER

    def test_backoff_schedule_actually_varies(self) -> None:
        """The decisive arm -- a stored constant proves nothing if urllib3 ignores it."""
        with JuniperRecurrenceClient(base_url="http://localhost:8211", retries=5) as client:
            retry = client.session.get_adapter("http://localhost:8211/").max_retries

        # get_backoff_time() returns 0 until at least two consecutive errors.
        for _ in range(2):
            retry = retry.increment(method="GET", url="/x", error=Exception("transient"))

        observed = {retry.get_backoff_time() for _ in range(200)}
        assert len(observed) > 1, "backoff is constant across 200 samples -- jitter is not being applied"

        # Bounds follow urllib3's documented formula for two consecutive errors:
        # backoff_factor * 2 ** (n - 1), then + uniform(0, backoff_jitter).
        base = constants.DEFAULT_BACKOFF_FACTOR * 2
        assert min(observed) >= base
        assert max(observed) <= base + constants.DEFAULT_BACKOFF_JITTER
