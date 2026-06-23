"""TST-6: run the bench app_e2e Δt proof (``bench/app_e2e.py``) inside the bench CI lane.

``app_e2e.main()`` drives ``POST /v1/train`` -> ``/v1/predict`` -> ``/v1/crossval`` through
the deployed FastAPI app (via ``TestClient``, with the juniper-data adapter mocked) and
asserts the app trains + predicts well on irregular-Δt data. It ships as a
``python -m bench.app_e2e`` entrypoint with no ``test_`` function, so ``pytest bench/``
never collected it. This thin wrapper runs it in the bench lane, which installs the app's
``[test,bench]`` extras (the FastAPI app + the juniper-data generators).
"""

from __future__ import annotations

import pytest

pytest.importorskip("juniper_data")  # the irregular_sine generator
pytest.importorskip("juniper_recurrence")  # the deployed FastAPI app


def test_app_e2e_trains_and_predicts_on_irregular_dt() -> None:
    # Imported inside the test so the importorskips above gate collection cleanly
    # (app_e2e pulls fastapi + the app only when the bench extras are installed).
    from bench import app_e2e

    # main() asserts internally (train r2 >= 0.9, predict + crossval HTTP 200);
    # a regression surfaces as an AssertionError raised through here.
    app_e2e.main()
