"""Recurrence evaluation / Δt-proof benchmark harness.

Design: juniper-ml ``notes/JUNIPER_RECURRENCE_EVALUATION_DESIGN_2026-06-18.md`` (Wave-2 C2/I2).

Standalone harness (not part of either published dist). Run from the repo root:

    python -m bench.run_benchmark        # the C2 benchmark -> bench/results/
    python -m bench.app_e2e              # the I2 end-to-end app proof

Requires the ``[bench]`` extra (juniper-data>=0.7.0, for the generators) plus the model +
crossval already pinned by the app. juniper-data 0.7.0 ships the synthetic Δt generators
(#187/#188) + scaling meta (#189) on PyPI, so ``pip install -e '.[bench]'`` resolves cleanly
from PyPI — no editable-sibling clone needed.
"""
