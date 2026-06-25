# AGENTS.md

**Project**: juniper-recurrence-client — HTTP Client for the juniper-recurrence service
**Repository**: pcalnon/juniper-recurrence
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.2.0
**Last Updated**: 2026-06-25

---

This file guides Claude Code (and other agents) working in the `juniper-recurrence-client/`
package. `CLAUDE.md` is a symlink to this file.

## What this is

The HTTP client library for the **juniper-recurrence** FastAPI service (train / predict /
cross-validate / inspect). The 3rd distribution in the `pcalnon/juniper-recurrence` repo,
alongside `juniper-recurrence-model/` (the Δt-native LMU model) and `juniper-recurrence/` (the
FastAPI/CLI app). Mirrors `juniper-data-client` / `juniper-cascor-client`; its primary consumer is
juniper-canopy's recurrence `BackendProtocol` adapter.

## Commands

```bash
pip install -e ".[test]"        # install with test deps (requests, responses, pytest)
python -m pytest -q             # run the suite
ruff check .                    # lint (line-length 512; E/F/W/B/I/N)
python -m build                 # build sdist + wheel
```

## Conventions

- Python >= 3.12. Version is single-sourced in `juniper_recurrence_client/_version.py`
  (setuptools dynamic-attr).
- `requests` + `urllib3.Retry` transport (NOT httpx); tests mock HTTP with `responses`.
- **Auth asymmetry:** the *client* sends one key under the singular `JUNIPER_RECURRENCE_API_KEY`
  (its `_FILE` Docker-secret form resolved first) as the `X-API-Key` header. The *server* reads
  the *plural* `JUNIPER_RECURRENCE_API_KEYS` (CSV/JSON — its accepted set). Keep the
  singular/plural distinction in mind.
- Retry is idempotent-only (GET/HEAD): the train/predict/crossval POSTs carry server-side state
  (train & crossval are lock-guarded → 409), so they must never auto-retry on a transient 5xx.
- `X` is the design-matrix argument name (ML convention); `client.py` carries a `per-file-ignore`
  for ruff N803 (mirrors the model package).

## Release

Publishing is tag-triggered (`juniper-recurrence-client-v*`) via
`.github/workflows/publish-recurrence-client.yml` (OIDC trusted publishing, TestPyPI → PyPI),
mirroring the model and app publish workflows. Bump `_version.py` + cut the `[Unreleased]`
CHANGELOG section in the release PR.
