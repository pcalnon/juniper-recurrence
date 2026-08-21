# AGENTS.md

**Project**: juniper-recurrence — Recurrent / Continuous-Time Neural-Network Application for the Juniper ML Research Platform
**Repository**: pcalnon/juniper-recurrence
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.4.0
**Last Updated**: 2026-08-21

---

This file guides Claude Code (and other agents) working in this repository. `CLAUDE.md` is a symlink to this file.

## What this is

`juniper-recurrence` is the recurrent / continuous-time neural-network application for the Juniper platform — the structural sibling of `juniper-cascor`. It adds time-axis memory for **time-series regression**, with the selected model being **P3-C (LMU + Approach-C)**: a closed-form, variable-Δt Legendre Memory Unit discretization (the only C1-clean, irregular-Δt-native option; see the design doc).

It is a live **4-sub-project monorepo** — a FastAPI + CLI application, its model-specific core, an HTTP client, and a benchmark harness — with **three packages published to PyPI**:

| Sub-project | Directory | PyPI package | Version |
|---|---|---|---|
| Application (FastAPI + CLI service) | `juniper-recurrence/` | `juniper-recurrence` | 0.4.0 |
| Model core (Δt-native LMU + `LMURegressor`) | `juniper-recurrence-model/` | `juniper-recurrence-model` | 0.2.0 |
| HTTP client | `juniper-recurrence-client/` | `juniper-recurrence-client` | 0.2.0 |
| Benchmark / evaluation harness | `bench/` | _(not a package)_ | n/a |

The application is the first real consumer of the shared `juniper-service-core` framework (`create_app` + `TrainingLifecycle`), and the model passes the shared `juniper-model-core` `TrainableModel` conformance kit unchanged. The model, the data foundation, and the service framework all ship as separate PyPI packages; the app is the glue + the HTTP/CLI surface.

The canonical design of record lives in juniper-ml:

- [`notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-14_JUNIPER-RECURRENCE_MODEL-DETAILED-DESIGN.md) — application **and** model-package design.
- [`notes/JUNIPER_RECURRENCE_WS4B_APP_BUILD_PLAN_2026-06-15.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-15_JUNIPER-RECURRENCE_WS4B-APP-BUILD-PLAN.md) — the WS-4b application build plan.
- [`notes/JUNIPER_RECURSE_OQ4_CASCOR_3D_INGESTION_GATE_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-14_JUNIPER-RECURRENCE_RECURSE-OQ4-CASCOR-3D-INGESTION-GATE.md) — the 3-D dataset-ingestion build-side scoping.

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
│   └── workflows/                   # per-package CI + publish (path-scoped) + repo-wide gates and nets
│       ├── ci-recurrence-app.yml
│       ├── ci-recurrence-model.yml
│       ├── ci-recurrence-client.yml
│       ├── publish-recurrence-app.yml
│       ├── publish-recurrence-model.yml
│       ├── publish-recurrence-client.yml
│       ├── pr-base-branch-guard.yml
│       ├── ci-recurrence-bench.yml
│       ├── ci-pre-commit.yml
│       ├── ci-docs.yml
│       ├── security-scan.yml
│       ├── sequence-safety.yml      # per-PR advisory sequence-safety net (rollout extension, 2026-08-09)
│       └── main-verify.yml          # post-merge bypass-proof compositional-loss net
├── notes/                           # repo-local notes
├── scripts/                         # repo-level tooling
│   └── check_version_drift.py       # CI-06 version-drift lint (run by the version-drift pre-commit hook)
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

A repo-wide **version-drift** gate (`scripts/check_version_drift.py`, audit CI-06) runs as a `version-drift` pre-commit hook (and so via the `CI — pre-commit` gate): each package's `_version.py` must agree with its CHANGELOG top heading and the root AGENTS.md version table, and the root `**Version**` header must match the app. Pure stdlib; the git-tag check degrades gracefully on a shallow checkout.

## Sequence-safety nets (required CI)

The ecosystem sequence-safety rollout ([the juniper-ml rollout plan](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-08-07_JUNIPER-ECOSYSTEM_SEQUENCE-SAFETY-ROLLOUT-PLAN.md)) was extended to this monorepo on 2026-08-09 (the original Wave-2 repo set predated / omitted it). Both workflows consume the published `juniper-ci-tools>=0.8.0,<0.9.0` console scripts (`juniper-symbol-loss-check` / `juniper-docs-additions-check`); neither is a required check.

- `.github/workflows/sequence-safety.yml` — per-PR **advisory** screens over base..HEAD: AST symbol-loss + docs deletion-magnitude. Symbol scope: five monorepo trees (`juniper-recurrence/**`, `juniper-recurrence-model/**`, `juniper-recurrence-client/**`, `bench/**`, `scripts/**`; tests/ live inside each tree). Docs screen: the universal default cluster (AGENTS.md, docs/, notes/). `allow-symbol-loss` / `docs-rewrite` labels demote the screen to WARN-only; JSON reports upload as `sequence-safety-report`.
- `.github/workflows/main-verify.yml` — post-merge, bypass-proof net on `push: main` (per-SHA concurrency, no cancel, so a merge storm never drops a verification): the same two screens over the catch-up BASE..merge (screens-only — no battery; the per-package CI lanes gate pre-merge). On failure it upserts a stable-title tracking issue (one per red streak) and posts a non-blocking Slack summary when a `SLACK_WEBHOOK_URL` secret exists (none is currently provisioned, so that step self-skips).

An intentional symbol removal / docs rewrite is waived with the enumerated `Allow-Symbol-Loss: <qualified.symbol>` / `Allow-Docs-Rewrite: <path>` commit trailers, which travel in git history and clear both nets — carry them into the squash-merge commit message.

### PR base-branch guard (required check)

`.github/workflows/pr-base-branch-guard.yml` fails any PR whose base branch is not the
default branch. Its job name -- **`Guard PR base branch`** -- is a **required status check**
in this repo's ruleset, so renaming the job or deleting the file makes `main` unmergeable
until the context is un-required first.

**What it protects against.** A PR based on another feature branch can squash-merge into
that branch, stranding its content off `main` behind a green **MERGED** badge. It has
happened three times in this ecosystem (`juniper-recurrence#7`/`#8`, `juniper-canopy#365`).

**Why it matters more than it looks.** Both rulesets here are scoped to `~DEFAULT_BRANCH`, so
a PR whose base is a feature branch is governed by **no ruleset at all** -- it has zero
required status checks and merges clean with nothing having run:

```bash
gh api repos/pcalnon/<repo>/rules/branches/feature%2Fanything --jq length   # -> 0
gh api repos/pcalnon/<repo>/rules/branches/main               --jq length   # -> 9
```

This workflow carries no `branches:` filter, so it is the **only** check that runs on such a
PR. It cannot block the merge there -- no ruleset applies -- but it turns a silent merge into
a visibly red one.

**If it fails.** Re-open the work against the default branch. The house practice is
**close and re-open** a fresh PR titled `[retarget #NNN]`. Retargeting in place is *not*
sufficient on its own: every `ci*.yml` here uses the default `pull_request` types
`[opened, synchronize, reopened]`, which exclude `edited`, so a retarget re-runs this guard
and nothing else -- the PR stays blocked on its other required contexts until a push or a
close/re-open.

**`stacked-pr` label.** Silences this guard for a deliberate stack. It does **not** make the
PR mergeable into `main`, and it does **not** re-land the stack -- do that separately.

Rollout and rationale: [juniper-ml#434](https://github.com/pcalnon/juniper-ml/issues/434).

## Status

Live monorepo: the application (`juniper-recurrence` 0.3.0), the model core (`juniper-recurrence-model` 0.2.0), and the HTTP client (`juniper-recurrence-client` 0.2.0) are all published to PyPI, plus the `bench/` evaluation harness. The Δt-native LMU memory unit (the C1-clean Approach-C core) and `LMURegressor` pass `juniper-model-core`'s conformance kit; the app exposes the train / predict / model / dataset / cross-validation HTTP surface on the shared `juniper-service-core` framework.
