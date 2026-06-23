"""TST-11: client configuration validation (``JuniperRecurrenceConfigurationError``).

``JuniperRecurrenceConfigurationError`` was an exported-but-never-raised exception. It is now
raised by ``_normalize_url`` when ``base_url`` carries no host, so a hostless/empty URL fails
fast at construction instead of normalizing to a broken URL and erroring opaquely on the first
request.
"""

from __future__ import annotations

import pytest

from juniper_recurrence_client import (
    JuniperRecurrenceClient,
    JuniperRecurrenceConfigurationError,
)


@pytest.mark.parametrize("bad", ["", "   ", "http://", "https://"])
def test_hostless_base_url_raises_configuration_error(bad: str) -> None:
    with pytest.raises(JuniperRecurrenceConfigurationError):
        JuniperRecurrenceClient(base_url=bad)


def test_configuration_error_is_a_client_error() -> None:
    # It subclasses the client base, so callers catching the base still catch it.
    from juniper_recurrence_client import JuniperRecurrenceClientError

    assert issubclass(JuniperRecurrenceConfigurationError, JuniperRecurrenceClientError)


def test_valid_hostful_base_url_constructs() -> None:
    client = JuniperRecurrenceClient(base_url="recurrence.test:8211")
    assert client.base_url == "http://recurrence.test:8211"
