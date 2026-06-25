# AGENTS.md

**Project**: juniper-recurrence — Recurrent / Continuous-Time Neural-Network Application for the Juniper ML Research Platform
**Repository**: pcalnon/juniper-recurrence
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.2.0
**Last Updated**: 2026-06-25

---

This file guides Claude Code (and other agents) working in this repository. `CLAUDE.md` is a symlink to this file.

## What this is

`juniper-recurrence` is the recurrent / continuous-time neural-network application for the Juniper platform — the structural sibling of `juniper-cascor`. It adds time-axis memory for **time-series regression**, with the selected model being **P3-C (LMU + Approach-C)**: a closed-form, variable-Δt Legendre Memory Unit discretization (the only C1-clean, irregular-Δt-native option; see the design doc).

It is a live **4-sub-project monorepo** — a FastAPI + CLI application, its model-specific core, an HTTP client, and a benchmark harness — with **three packages published to PyPI**:

| Sub-project | Directory | PyPI package | Version |
|---|---|---|---|
| Application (FastAPI + CLI service) | `juniper-recurrence/` | `juniper-recurrence` | 0.2.0 |
| Model core (Δt-native LMU + `LMURegressor`) | `juniper-recurrence-model/` | `juniper-recurrence-model` | 0.1.5 |
| HTTP client | `juniper-recurrence-client/` | `juniper-recurrence-client` | 0.2.0 |
| Benchmark / evaluation harness | `bench/` | _(not a package)_ | n/a |

The application is the first real consumer of the shared `juniper-service-core` framework (`create_app` + `TrainingLifecycle`), and the model passes the shared `juniper-model-core` `TrainableModel` conformance kit unchanged. The model, the data foundation, and the service framework all ship as separate PyPI packages; the app is the glue + the HTTP/CLI surface.

The canonical design of record lives in juniper-ml:

- [`notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md) — application **and** model-package design.
- [`notes/JUNIPER_RECURRENCE_WS4B_APP_BUILD_PLAN_2026-06-15.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_WS4B_APP_BUILD_PLAN_2026-06-15.md) — the WS-4b application build plan.
- [`notes/JUNIPER_RECURSE_OQ4_CASCOR_3D_INGESTION_GATE_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURSE_OQ4_CASCOR_3D_INGESTION_GATE_2026-06-14.md) — the 3-D dataset-ingestion build-side scoping.

## Repository layout

Follows the Juniper "model family" pattern (precedent: `juniper-cascor/juniper-cascor-protocol/`): each independently-publishable package lives in a same-named subdirectory, alongside the `bench/` harness.

```text
juniper-recurrence/
├── LICENSE                          # MIT (repo-level)
├── README.md
├── AGENTS.md                        # this file (CLAUDE.md -> AGENTS.md)
├── .gitignore
├── .github/
│   ├── CODEOWNERS
│   └── workflows/                   # per-package CI + publish, path-scoped
│       ├── ci-recurrence-app.yml
│       ├── ci-recurrence-model.yml
│       ├── ci-recurrence-client.yml
│       ├── publish-recurrence-app.yml
│       ├── publish-recurrence-model.yml
│       ├── publish-recurrence-client.yml
│       └── pr-base-branch-guard.yml
├── notes/                           # repo-local notes
├── juniper-recurrence/              # the FastAPI + CLI application (PyPI: juniper-recurrence)
│   ├── pyproject.toml
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── Dockerfile
│   ├── juniper_recurrence/          # import package (app, routers, settings, CLI, …)
│   └── tests/
├── juniper-recurrence-model/        # the model-specific core (PyPI: juniper-recurrence-model)
│   ├── pyproject.toml
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── juniper_recurrence_model/    # import package (LMU memory unit, LMURegressor, readouts, data)
│   └── tests/
├── juniper-recurrence-client/       # the HTTP client (PyPI: juniper-recurrence-client)
│   ├── pyproject.toml
│   ├── README.md
│   ├── AGENTS.md
│   ├── juniper_recurrence_client/   # import package
│   └── tests/
└── bench/                           # benchmark / evaluation harness (not published)
    ├── datasets.py
    ├── baselines.py
    ├── run_benchmark.py
    ├── app_e2e.py
    ├── test_bench_smoke.py
    └── results/
```

## Conventions (inherited from the Juniper ecosystem)

- Python >= 3.12; line length 512 (ruff); pytest + ≥80% coverage target (CI gates each package at 90%).
- Package naming: `-core` = genuinely shared abstraction (homed in juniper-ml); `juniper-<model>-model` = model-specific core (homed here). This repo's core is therefore `juniper-recurrence-model`.
- Dataset capability belongs to `juniper-data`; this repo never generates or vendors datasets (the `data/` path is gitignored). The `bench/` harness pulls datasets through `juniper-data` / `juniper-data-client`.
- Observability via `juniper-observability` (`[prometheus]>=0.4.0` for the app's `/metrics`; `>=0.3.1` for the client). Imports are guarded — the app and client run without the extra installed.
- Independent publish per package: the app on `juniper-recurrence-v*` tags, the model on `juniper-recurrence-model-v*` tags, the client on `juniper-recurrence-client-v*` tags (path-scoped so they never cross-fire). TestPyPI-first, then PyPI, via OIDC trusted publishing.

## Testing

No dedicated on-host conda env carries the app's deps; install the package + test extras into your active env first, then run each package's suite **from its own subdirectory** (each `pyproject.toml` sets `testpaths=["tests"]`):

```bash
# Application (needs the observability extra for the /metrics route tests)
cd juniper-recurrence && pip install -e ".[test,observability]" && python -m pytest

# Model core
cd juniper-recurrence-model && pip install -e ".[test]" && python -m pytest

# HTTP client
cd juniper-recurrence-client && pip install -e ".[test]" && python -m pytest

# Benchmark harness — run from the REPO ROOT so `import bench` resolves
pip install -e "juniper-recurrence/.[test,bench]" && python -m pytest bench/
```

CI mirrors these per-package invocations across the Python 3.12 / 3.13 / 3.14 matrix and enforces `--cov-fail-under=90`. The pytest `addopts` carry the ecosystem-standard `-p no:dash -p no:playwright` autoload-SIGSEGV guard.

## Status

Live monorepo: the application (`juniper-recurrence` 0.2.0), the model core (`juniper-recurrence-model` 0.1.5), and the HTTP client (`juniper-recurrence-client` 0.2.0) are all published to PyPI, plus the `bench/` evaluation harness. The Δt-native LMU memory unit (the C1-clean Approach-C core) and `LMURegressor` pass `juniper-model-core`'s conformance kit; the app exposes the train / predict / model / dataset / cross-validation HTTP surface on the shared `juniper-service-core` framework.
