# AGENTS.md

**Project**: juniper-recurrence — Recurrent / Continuous-Time Neural-Network Application for the Juniper ML Research Platform
**Repository**: pcalnon/juniper-recurrence
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.1.0
**Last Updated**: 2026-06-17

---

This file guides Claude Code (and other agents) working in this repository. `CLAUDE.md` is a symlink to this file.

## What this is

`juniper-recurrence` is the recurrent / continuous-time neural-network application for the Juniper platform — the structural sibling of `juniper-cascor`. It adds time-axis memory for **time-series regression**, with the selected model being **P3-C (LMU + Approach-C)**: a closed-form, variable-Δt Legendre Memory Unit discretization (the only C1-clean, irregular-Δt-native option; see the design doc).

The canonical design of record lives in juniper-ml:
- [`notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md) — application **and** model-package design (may later split).
- [`notes/JUNIPER_RECURSE_OQ4_CASCOR_3D_INGESTION_GATE_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURSE_OQ4_CASCOR_3D_INGESTION_GATE_2026-06-14.md) — the 3-D dataset-ingestion build-side scoping.

## Repository layout

Follows the Juniper "model family" pattern (precedent: `juniper-cascor/juniper-cascor-protocol/`): each independently-publishable, model-specific package lives in a same-named subdirectory.

```
juniper-recurrence/
├── LICENSE                       # MIT (repo-level)
├── README.md
├── AGENTS.md                     # this file (CLAUDE.md -> AGENTS.md)
├── .gitignore
└── juniper-recurrence-model/     # the model-specific core package (PyPI: juniper-recurrence-model)
    ├── pyproject.toml
    ├── README.md
    ├── CHANGELOG.md
    ├── juniper_recurrence_model/ # import package
    │   ├── __init__.py
    │   ├── _version.py
    │   └── units/
    │       ├── __init__.py
    │       └── lmu_varstep.py    # the Δt-native LMU (Approach-C) memory unit
    └── tests/
        └── test_lmu_grid_invariance.py
```

The future app package (`juniper_recurrence/`, the FastAPI service + CLI) is **not yet scaffolded** — it is gated on the shared `juniper-service-core` / `juniper-model-core` packages (which live as juniper-ml subdirectories), per the model/middleware refactor (WS-0/WS-2/WS-3).

## Conventions (inherited from the Juniper ecosystem)

- Python >= 3.12; line length 512 (ruff); pytest + ≥80% coverage target.
- Package naming: `-core` = genuinely shared abstraction (homed in juniper-ml); `juniper-<model>-model` = model-specific core (homed here). This repo's core is therefore `juniper-recurrence-model`.
- Dataset capability belongs to `juniper-data`; this repo never generates or vendors datasets (the `data/` path is gitignored).
- Observability via `juniper-observability` (`>=0.3.1`) once the app shell exists.
- Independent publish per package on a `juniper-recurrence-model-v*` tag (to be wired, mirroring `juniper-cascor-protocol`'s `publish-protocol.yml` / `ci-protocol.yml`).

## Status

Pre-implementation scaffold; WS-0 not ratified. The Δt-native LMU memory unit (the C1-clean Approach-C core, verified) has landed in `juniper-recurrence-model`; `TrainableModel` interface wiring follows when `juniper-model-core` is defined.
