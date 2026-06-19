"""Recurrence evaluation / Δt-proof benchmark harness.

Design: juniper-ml ``notes/JUNIPER_RECURRENCE_EVALUATION_DESIGN_2026-06-18.md`` (Wave-2 C2/I2).

Standalone harness (not part of either published dist). Run from the repo root:

    python -m bench.run_benchmark        # the C2 benchmark -> bench/results/
    python -m bench.app_e2e              # the I2 end-to-end app proof

Requires the ``[bench]`` extra (juniper-data, for the generators) plus the model + crossval
already pinned by the app. juniper-data's synthetic generators (#187/#188) are on main but not
yet in a PyPI release; until juniper-data publishes them, install it editable from the sibling
clone (see the design doc §7 note).
"""
