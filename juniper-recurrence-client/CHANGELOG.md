# Changelog

All notable changes to the `juniper-recurrence-client` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with [PEP 440](https://peps.python.org/pep-0440/) pre-release identifiers.

## [Unreleased]

### Added

- **Per-call `timeout` override on the synchronous `train` / `crossval` calls** (defect-register
  `APD-RCLIENT-002`). Both endpoints run their compute *inside* the request — `train` fits the
  model, `crossval` runs `n_folds` sequential fits — so the client-wide 30 s default is routinely
  wrong for them and the only prior remedy was constructing a whole client with a bigger scalar.
  The new keyword-only `timeout` parameter governs that one request; `None` (the default) keeps
  the client-wide value and never means "no timeout". The timeout error message now reports the
  *effective* timeout — the override when one was passed — instead of unconditionally
  interpolating the client-wide `self.timeout`.

### Fixed

- **The README now documents the monorepo layout and the three-way name split** (defect-register
  `APD-RCLIENT-005`). This package is published from the `juniper-recurrence-client/` subdirectory of
  `pcalnon/juniper-recurrence`, so its repository, distribution (`juniper-recurrence-client`) and import
  (`juniper_recurrence_client`) names are three different strings. The README — which is what PyPI
  renders, and the only page a consumer of the published wheel sees — said none of this, while linking
  its two sibling clients to their own standalone repositories, reinforcing the expectation that this one
  has a standalone repository too. The filed defect is a **documentation** problem, not a metadata error:
  `[project.urls]` pointing at the monorepo is correct and is deliberately unchanged, and the
  hyphen-to-underscore import transform is conventional rather than drift. A reader who hits
  `ModuleNotFoundError` on the import name can now find the distribution to install, and knows which
  repository takes issues and PRs for this client.
- **`_normalize_url` treats the scheme case-insensitively and validates `hostname`, not `netloc`.** Two flaws
  found by a confirmed review on the cascor-client port of this client's own normalisation (this package is
  the reference implementation, so its flaws were being copied): a case-sensitive `startswith` re-prefixed
  `HTTPS://host` into `http://HTTPS://host` — a silent TLS downgrade sending the API key over HTTP to
  hostname `https` (RFC 3986 makes schemes case-insensitive) — and the hostless guard read `netloc`, which
  accepts a userinfo-only `http://user:secret@` as truthy while `hostname` is `None` for it. Uppercase and
  mixed-case schemes are now canonicalised, and the userinfo-only form joins the hostless-rejection tests.
- **`mypy --strict` now actually runs against the surface this package advertises** (defect-register
  `APD-RCLIENT-003`). The package ships `py.typed` and the `Typing :: Typed` classifier, but no mypy
  configuration existed anywhere in the monorepo and no lane ran mypy at all — a checked surface
  nothing checks. The package now carries `[tool.mypy]` (strict) in its own `pyproject.toml`, and the
  repo pre-commit gate gains a mypy hook scoped to this package, so the advertisement is enforced on
  every push. Making strict true surfaced one real looseness: `_parse_json` returned `Any`, silently
  laundering every public method's declared `dict[str, Any]` — it now validates the body is a JSON
  object and raises the typed client error on a syntactically valid non-object body, which previously
  surfaced as a downstream `AttributeError` in the caller.
- **Exceptions now carry `status_code`, `detail` and `response`** (defect-register
  `APD-RCLIENT-001`). Every exception subclassed `Exception` with nothing on it, so a 400 and a
  422 raised the same type with the same text and the only way to tell them apart was
  substring-matching the message. The base `JuniperRecurrenceClientError.__init__` now accepts
  keyword-only `status_code` / `detail` / `response`, and every mapped branch (404, 409,
  400/422, and the generic fallback) passes them. **Backward compatible**: the new parameters
  are keyword-only, so existing single-positional-message call sites are unchanged, and locally
  raised errors (configuration, connection, timeout) simply report `status_code=None`.
- **A FastAPI 422 `detail` list is no longer f-string-interpolated into an unparseable repr.**
  FastAPI answers a validation failure with a *list* of error objects; that list went straight
  into the message, producing `Validation error: [{'type': 'missing', ...}]`. The structure is
  now attached to `exc.detail` **unmodified** while the message renders it as
  `body.seed: Field required` via a new `_render_error_detail` helper. This is the same defect
  juniper-data-client tracks as `APD-DCLIENT-003`; it had never been recorded against this
  client.
- **Exception context survives `pickle` and `copy`.** `BaseException.__reduce__` returns
  `(cls, args, self.__dict__)` whenever the instance dict is non-empty, so the keyword-only
  context is restored automatically — but only while `args` holds exactly the constructor's
  positional message, which is the invariant `test_context_survives_pickle_and_copy` pins (the
  failure mode `B042` warns about). An interim `__reduce__` override that reproduced this
  default byte-for-byte was removed; its stated rationale — that the default rebuilds from
  `args` alone — was wrong, and the same correction has landed in juniper-service-core,
  juniper-data-client and juniper-cascor-client.

  Port of the convention established in
  [juniper-data-client#158](https://github.com/pcalnon/juniper-data-client/pull/158) and
  [juniper-cascor-client#123](https://github.com/pcalnon/juniper-cascor-client/pull/123). The
  three Juniper clients are separately released packages with no shared code, so nothing
  mechanical keeps them aligned; the alignment is a convention carried by each package's tests
  and AGENTS.md.

### Changed

- **Retry backoff is jittered — `backoff_jitter` is passed to urllib3's `Retry`** (defect-register
  `APD-ECO-002`). Without it, every client instance that tripped the same transient outage retried on
  an *identical* schedule, so a service that was already failing took a synchronised herd on each
  backoff step. urllib3 applies jitter as an **absolute additive term**
  (`backoff_value += random.random() * backoff_jitter`), not a proportional one, so the new
  `DEFAULT_BACKOFF_JITTER` is matched to `DEFAULT_BACKOFF_FACTOR` (0.5) — a full window of spread on
  the first retry, the step that carries the most callers. **No dependency floor moves**:
  `backoff_jitter` arrived in urllib3 2.0.0 and this package already pins `urllib3>=2.0.0`. Retry
  counts, allowed methods and the status forcelist are untouched, so retry *behaviour* is unchanged —
  only its timing is decorrelated. `tests/test_retry_policy.py` pins the constant's presence, its
  positivity (a `0.0` would silently restore the herd while leaving the call site looking correct),
  and — the decisive arm — that 200 sampled backoffs actually differ.

- **Per-file coverage lifted to the ratified bars + a blocking gate wired into CI (per-file
  coverage rollout C-5, juniper-ml
  `notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md`).** Every source file
  now measures ≥90% statement coverage and the package's statement-weighted `pooled` coverage is
  ≥95% (baseline `client.py` 92.73% / package pooled 93.96% → 100% / 100%), enforced on every PR
  by `juniper-coverage-gap-map --enforce` (`juniper-ci-tools>=0.6.0,<0.7.0`). Added 6 `responses`-based
  tests (403/501 error-status `else` arm, non-JSON error-body text fallback, `on_request` hook
  exception suppression, `crossval` MLP-regularization knobs, `DatasetRef` `params` forwarding,
  `is_ready` typed-error path). The `[test]` extra now pulls `juniper-observability` so the three
  guarded `X-Request-ID` propagation tests run instead of skipping in CI. Tests / CI / packaging-extra
  only — no runtime change, no version bump.

## [0.2.0] - 2026-06-24

### Added

- **`readout` selection forwarded by `train()` / `crossval()` (DP-3 P2c).** Both methods gain
  `readout: Optional[Literal["linear", "rff"]]`, `rff_features: Optional[int]`, and
  `rff_gamma: Optional[Union[float, Literal["median"]]]`, forwarded verbatim in the request body so
  callers can select the service's nonlinear RFF readout (Rung 2a). Backward compatible — all optional;
  unset ⇒ an unchanged request body.

- **`readout="mlp"` + MLP knobs forwarded by `train()` / `crossval()` (DP-3 P3).** Both methods widen
  `readout` to `Optional[Literal["linear", "rff", "mlp"]]` and gain `mlp_hidden` / `mlp_weight_decay` /
  `mlp_lr` / `mlp_max_epochs` / `mlp_patience` (all `Optional`), forwarded verbatim in the request body
  so callers can select the service's torch MLP readout (Rung 2b). Backward compatible — all optional;
  unset ⇒ an unchanged request body. (The service needs its own `[torch]` extra to fulfil `readout="mlp"`.)

- **`ridge="gcv"` accepted by `train()` / `crossval()` (DP-3 P1).** The `ridge` parameter widens
  from `Optional[float]` to `Optional[Union[float, Literal["gcv"]]]`, so callers can request the
  service's closed-form GCV selection of the readout penalty. The value is forwarded verbatim in
  the request body (no client-side validation change).

## [0.1.0] - 2026-06-18

### Added

- **Initial `juniper-recurrence-client` package** — a lean `requests`-based HTTP client for the
  juniper-recurrence service, the 3rd distribution in the `pcalnon/juniper-recurrence` repo
  alongside `juniper-recurrence-model` and `juniper-recurrence`. Mirrors `juniper-data-client` /
  `juniper-cascor-client` so juniper-canopy's recurrence `BackendProtocol` adapter drives every
  Juniper backend the same way.
- **`JuniperRecurrenceClient`** wrapping the app's REST surface: `train` / `training_status`,
  `predict`, `crossval` / `crossval_status`, `get_model`, `get_dataset`, and
  `health_check` / `is_ready` / `wait_for_ready`. Idempotent-only retry policy (GET/HEAD only — the
  train/predict/crossval POSTs carry server-side state), a pooled `requests.Session`, `X-API-Key`
  auth with `_FILE` Docker-secret indirection, the optional `on_request` instrumentation hook, and
  best-effort `X-Request-ID` propagation via `juniper-observability` (guarded — never required).
- **Typed exception hierarchy** (`JuniperRecurrenceClientError` + connection / timeout /
  not-found / **conflict (409)** / validation / configuration leaves). The 409 path is unique to
  the recurrence app (lock-guarded train/crossval; "no trained model yet").
- **33 unit tests** (`responses`-mocked) covering URL normalization, every client method, auth
  resolution (explicit / env / `_FILE` precedence + empty-file fallback), and the full
  error-mapping matrix (404/409/422/500/connection/malformed-JSON).
