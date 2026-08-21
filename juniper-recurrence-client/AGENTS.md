# AGENTS.md

**Project**: juniper-recurrence-client — HTTP Client for the juniper-recurrence service
**Repository**: pcalnon/juniper-recurrence
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.2.0
**Last Updated**: 2026-08-21

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
- **Exception context (do not remove).** Every exception carries `message`, `status_code`,
  `detail` and `response`, set by the base `__init__`. `status_code` is the *only* thing
  separating a 400 from a 422 — both raise `JuniperRecurrenceValidationError` — and before it
  existed, telling them apart meant substring-matching the message (defect-register
  `APD-RCLIENT-001`). Three constraints a refactor must not break:
  - The extra parameters are **keyword-only**, so every single-positional-message call site keeps
    working; making any of them positional is a breaking change for consumers.
  - `detail` holds the server's payload **exactly as decoded** — a `str` for most handlers, a
    `list[dict]` for FastAPI's 422. The message renders that list as `body.seed: Field required`
    via `client._render_error_detail`; the structure itself stays on the attribute. Interpolating
    it into the message was the defect: the result was an unparseable Python repr.
  - `__reduce__` must stay. `BaseException.__reduce__` rebuilds from `args`, which holds only the
    message, so without it a pickle/copy round-trip returns an exception that looks right and has
    silently lost the context. That is what ruff/flake8-bugbear `B042` warns about; the `noqa` on
    `__init__` is paired with `__reduce__`, not a dismissal of it.
- **The three-client convention.** This package's exception surface deliberately mirrors
  `juniper-data-client` and `juniper-cascor-client`. They are separately released packages with
  **no shared code**, so no drift check can span them — the alignment is a convention, kept by
  each package's tests and these notes. juniper-data-client#158 is the reference implementation.

## Release

Publishing is tag-triggered (`juniper-recurrence-client-v*`) via
`.github/workflows/publish-recurrence-client.yml` (OIDC trusted publishing, TestPyPI → PyPI),
mirroring the model and app publish workflows. Bump `_version.py` + cut the `[Unreleased]`
CHANGELOG section in the release PR.
