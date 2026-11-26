# juniper-recurrence

**Project**: Juniper — Cascade Correlation Neural Network Research Platform
**Application**: juniper-recurrence (FastAPI + CLI service)
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.2.0

FastAPI + CLI service that wraps the Δt-native Legendre Memory Unit regressor
([`juniper-recurrence-model`](https://github.com/pcalnon/juniper-recurrence)) on the
shared [`juniper-service-core`](https://pypi.org/project/juniper-service-core/)
framework. It loads 3-D windowed sequences (`equities_seq`, the WS-1 irregular-Δt
contract) through [`juniper-data-client`](https://pypi.org/project/juniper-data-client/)
and trains / serves the LMU over HTTP.

This is the **application layer** (WS-4b): the first real consumer of
service-core's `create_app` + `TrainingLifecycle`. The model, the data foundation,
and the service framework ship separately; this package is the glue + the HTTP/CLI
surface.

## Install

```bash
pip install juniper-recurrence
pip install "juniper-recurrence[torch]"   # adds the torch MLP readout (DP-3 Rung 2b)
```

All upstreams resolve from PyPI: `juniper-service-core`, `juniper-model-core`,
`juniper-recurrence-model`, `juniper-data-client`, plus `fastapi` / `uvicorn`.

## Run

```bash
# Serve the API (single worker, in-process state). Binds 0.0.0.0:8210 by default;
# set JUNIPER_RECURRENCE_HOST=127.0.0.1 for local-only.
juniper-recurrence serve
juniper-recurrence serve --host 127.0.0.1 --port 8210
```

Once running, the API exposes (every `/v1/*` route below requires `X-API-Key` when API
keys are configured; health + docs are always exempt):

| Route | Method | Behavior |
|---|---|---|
| `/v1/health`, `/v1/health/ready` | GET | Liveness / readiness (exempt). |
| `/v1/train` | POST | Train the LMU on a dataset (synchronous); returns the `TrainResult`, with `metrics_scope: "in_sample"` labelling its `final_metrics`, and the fit's `operation_id`. |
| `/v1/training/status` | GET | `idle` / `training` / `restoring` / `trained` / `restored` / `failed`, the operation it describes (`operation_id`, `busy_since`, `dataset_id`, `requested_by`), `model_operation_id`, a `failure` under `failed`, + last metrics + training events. |
| `/v1/crossval` | POST | Walk-forward cross-validation over the `trainval` pool (train \| val, rebuilt entity-major; `test` is never in a fold). On an artifact with `ticker_code` / `window_end_date` the folds are cut on the global date with a per-entity embargo (default: one look-back) and a label-availability rule; `embargo_days` adds a duration gap. The response's `folding` block says what was done. Synchronous; returns aggregated per-fold metrics. |
| `/v1/crossval/status` | GET | Most recent cross-validation result (aggregate + per-fold). |
| `/v1/predict` | POST | Continuous predictions for inline `X` (+ `dt`) or a dataset ref; optional `expect_operation_id` (`409` when it does not name the model's operation). |
| `/v1/model/snapshots` | GET / POST | List snapshots / save the current model; the save takes an optional `expect_operation_id`, like `/v1/predict`. |
| `/v1/model/snapshots/{id}/restore` | POST | Load a snapshot as the current model; the restore mints its own `operation_id`. |
| `/v1/model` | GET | Current model topology + regression metrics. |
| `/v1/dataset` | GET | Descriptor of the trained-on dataset. |
| `/v1/metrics` | GET | Prometheus metrics — IP-allowlist gated; needs the `[observability]` extra. |
| `/docs` | GET | OpenAPI / Swagger UI (exempt). |

Both `POST /v1/train` and `POST /v1/crossval` run **inline** (closed-form solves) and return
their result in the response — no background jobs or WebSocket streams in v1. A second
cross-validation run while one is in progress returns `409`.

`POST /v1/train`'s `final_metrics` are **in-sample**: they are scored on the split the fit was
trained on (`dataset.split`), and the response says so with `metrics_scope: "in_sample"`. They
measure fit, not generalisation; the route reports no held-out metric.

Both responses carry a **`diagnostics`** block per fit -- once on `POST /v1/train`, once per fold
on `POST /v1/crossval` -- with the hyperparameters the fit *resolved*: `theta` (the data-driven
value a `theta=None` request became, per fold), `ridge_mode` / `ridge_selected` (the lambda
`ridge="gcv"` selected), `gamma` (the median-heuristic RFF bandwidth), and the `gcv_at_edge` /
`null_model` flags (booleans on a GCV-selected fit, which juniper-recurrence-model 0.5.0 reports;
`null` means *not determined*, as on a fixed-ridge fit). The `train` CLI prints the same block under `Readout:`.

`POST /v1/train` and `POST /v1/crossval` accept a **`readout`** selector (DP-3): `"linear"`
(default, closed-form least squares), `"rff"` (nonlinear random-Fourier-features readout, with
`rff_features` / `rff_gamma`), or `"mlp"` (a torch MLP with the `mlp_*` knobs — needs the `[torch]`
extra; returns `503` if torch is not installed). `ridge` accepts a float or `"gcv"` (closed-form
generalized-cross-validation selection). A linear-readout request that omits `ridge` gets the
service's `default_ridge`, **`"gcv"` since 0.7.0** (plan W5.8 (B)): the penalty is GCV-selected on
the train-fold-standardised memory block, so on a drifting input the fit abstains (lambda at the grid
ceiling, `null_model: true` in `diagnostics`) rather than extrapolate as the former `0.0` min-norm
default did. An explicit `ridge: 0.0`, or `JUNIPER_RECURRENCE_DEFAULT_RIDGE=0.0` on the service,
restores the old behaviour. See the `juniper-recurrence-model` README for the spectrum.

```bash
# Train on a juniper-data dataset, then inspect the model.
curl -sX POST localhost:8210/v1/train \
  -H 'Content-Type: application/json' \
  -d '{"dataset": {"dataset_id": "<id>"}, "d": 16}'
curl -s localhost:8210/v1/model
```

### Train (headless CLI)

```bash
# Fit the LMU on a dataset and persist it — no server.
juniper-recurrence train --dataset <id> --d 16 --out model.npz
juniper-recurrence train --name equities_seq_v1 --split train

# Create the dataset from a generator with explicit generator params, then fit it.
juniper-recurrence train --generator equities_seq --params '{"symbols": ["AAPL"], "start_date": "2015-01-01", "end_date": "2022-01-01", "lookback": 64, "fundamentals_fill": "drop", "regression_target": "log_return"}'
juniper-recurrence train --generator equities_seq --params-file equities_seq.json
```

`--params` (a JSON object) and `--params-file` (a file holding one) carry the generator params that
`POST /v1/train` takes as `dataset.params`. They are mutually exclusive and need `--generator`. Pass
them explicitly for `equities_seq`: the generator's bare defaults are not a trainable configuration.
Either flag without `--generator`, malformed JSON, or JSON that is not an object exits `2`. The
printed metrics are in-sample, as on the route.

## One caller per service

A recurrence service is **exclusively owned by one caller at a time**: it has one process-wide
`train_lock`, one in-memory model and one snapshot directory. Two callers sharing a listener -- canopy
and a CLI experiment suite, say -- get each other's `409`s and silently score each other's models.

- **The operation id is how a caller proves whose model it is scoring.** Each `POST /v1/train` mints
  an `operation_id` when it takes the lock and returns it; a restore mints its own, which the restored
  model then carries. Pass it as `expect_operation_id` to `/v1/predict` and `POST /v1/model/snapshots`:
  a model produced by any other operation is refused with a `409` that names both ids.
- **Who holds it, and what happened.** A refused `POST /v1/train` gets a `409` whose `detail` names the
  holder (`operation_id`, `busy_since`, `requested_by`). `GET /v1/training/status` reports `training`
  while a fit runs, then `trained` or `failed` (with `failure`), plus `model_operation_id` for the model
  `/v1/predict` scores. Send an `X-Request-ID` with your train request; it comes back as `requested_by`.
- **A fit is not cancellable.** A caller that times out leaves the fit running under the lock; poll the
  status for its outcome rather than retrying.
- **Keep callers apart by port.** The juniper-ml experiment launcher (`util/experiment_stack.bash`)
  starts a fresh service per run on ports 8260-8289, and that is what keeps a CLI suite and canopy
  apart. Canopy's service URL is outside that range (unset by default; Compose sets
  `http://juniper-recurrence:8210`, host port 8211; a native service listens on 8210), so canopy and a
  suite pointed at the same native or Compose listener recreate the collision, and neither side warns.

## Configuration

All settings read the `JUNIPER_RECURRENCE_` environment namespace (e.g.
`JUNIPER_RECURRENCE_PORT`). Secrets honor the Docker `_FILE` indirection
(`JUNIPER_RECURRENCE_API_KEYS_FILE`, `JUNIPER_DATA_API_KEY_FILE`). When no API keys
are configured, authentication is disabled (open access — development default).

| Variable | Default | Purpose |
|---|---|---|
| `JUNIPER_RECURRENCE_HOST` | `0.0.0.0` | Bind host (container default; `127.0.0.1` locally). |
| `JUNIPER_RECURRENCE_PORT` | `8210` | Bind port (deploy maps host `8211` → container `8210`). |
| `JUNIPER_RECURRENCE_API_KEYS` | _(unset)_ | CSV or JSON-array of valid `X-API-Key` values. |
| `JUNIPER_DATA_URL` | `http://localhost:8100` | Upstream juniper-data base URL. |
| `JUNIPER_DATA_API_KEY` | _(unset)_ | Outbound `X-API-Key` to juniper-data. |
| `JUNIPER_RECURRENCE_JUNIPER_DATA_TIMEOUT_SECONDS` | `120` | Per-request timeout of the service's juniper-data client -- dataset creation, where a cold `equities_seq` fetch happens, included. Keep it below your callers' own timeouts. |

## Development

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Publishing

Releases are published to PyPI via GitHub Actions
(`.github/workflows/publish-recurrence-app.yml`) on a `juniper-recurrence-v*` tag —
TestPyPI first (with a `--no-deps` install verification), then PyPI, via OIDC trusted
publishing (no API tokens). The model package (`juniper-recurrence-model`) publishes
separately on `juniper-recurrence-model-v*` tags.

```bash
# Cut a GitHub Release on a juniper-recurrence-v* tag (the ecosystem convention — not a
# bare `git push <tag>`); the Release drives publish-recurrence-app.yml (TestPyPI → PyPI).
gh release create juniper-recurrence-v0.2.0 --generate-notes
```

The client (`juniper-recurrence-client`) publishes the same way on `juniper-recurrence-client-v*` tags, each package
from its own workflow. Each run builds once, and both indexes get those same files: the build records their sha256 in
`SHA256SUMS`, each publish job checks the files against it before uploading, and TestPyPI's JSON must list exactly
those sha256 after the TestPyPI upload and again before the PyPI upload, and PyPI's after it. The steps are a verbatim
copy of juniper-cascor's template:
[One Artifact, Checked Against TestPyPI](https://github.com/pcalnon/juniper-cascor/blob/main/docs/ci_cd/MANUAL.md#one-artifact-checked-against-testpypi).

Each workflow has five jobs, and a release runs the first four. `Build and Validate` builds the one artifact, kept 90
days. `Publish to TestPyPI` uploads it and checks TestPyPI's JSON; it holds `id-token: write` alone and checks nothing
out. `Verify the TestPyPI install`, a job of its own after it (ruling 2 of 2026-11-16), installs the wheel from TestPyPI
alone (`--no-deps`, with the version read from a sparse checkout of the package's `_version.py`) and imports the
package, from `$RUNNER_TEMP`, outside the checkout; the model's and the client's runtime dependencies come from PyPI by
name. It holds `contents: read` alone, with no environment and no `id-token`, so nothing the package runs can mint a
credential, and its name holds "TestPyPI", so juniper-ml's release-train ceremony reads its failure as `HALT_TESTPYPI`.
`Publish to PyPI` needs both TestPyPI jobs, waits at the `pypi` gate, checks TestPyPI's JSON again, uploads, and checks
PyPI's. The fifth, `Republish the tested files (dispatch only)`, runs alone on a dispatch with
`republish_from_testpypi`: it is the first rung of the recovery ladder, below. The workflow's `env` holds the template's
three parameters: `PUBLISH_PACKAGE`, `PUBLISH_DIST_DIR` and `PUBLISH_TAG_PREFIX`, the package's tag prefix
(`juniper-recurrence-v`, `juniper-recurrence-model-v`, `juniper-recurrence-client-v`); the build job's guard repeats the
prefix, since a job-level `if:` cannot read `env`. Every Release fires all three workflows, and each build job skips a
sibling's tag; a plain `workflow_dispatch` of a workflow builds only from its own package's tag.

### A release that did not reach PyPI

The Overtoad's recovery ladder (2026-11-15), in this order. Every rung publishes the files TestPyPI took, or nothing
([juniper-cascor's account](https://github.com/pcalnon/juniper-cascor/blob/main/docs/ci_cd/MANUAL.md#recovering-a-release-that-did-not-reach-pypi)).

1. **Republish TestPyPI's files.** Dispatch the package's workflow from its release tag. Only the job
   `Republish the tested files (dispatch only)` runs: it waits at the `pypi` gate (an approval, then the 5-minute
   timer) and builds nothing. It refuses unless the tag's newest release run of that workflow exists and both its
   TestPyPI jobs, `Publish to TestPyPI` and `Verify the TestPyPI install`, concluded success in their latest attempt
   (ruling 1 of 2026-11-16). Then it copies TestPyPI's files for that version to PyPI, each checked by sha256 against
   TestPyPI's JSON and, while the release run's artifact exists, its `SHA256SUMS`, and checks PyPI's JSON.

   ```bash
   gh workflow run publish-recurrence-app.yml --repo pcalnon/juniper-recurrence --ref juniper-recurrence-v<version> -f republish_from_testpypi=true
   gh workflow run publish-recurrence-model.yml --repo pcalnon/juniper-recurrence --ref juniper-recurrence-model-v<version> -f republish_from_testpypi=true
   gh workflow run publish-recurrence-client.yml --repo pcalnon/juniper-recurrence --ref juniper-recurrence-client-v<version> -f republish_from_testpypi=true
   ```

2. **If that fails, within 30 days of the release run: Re-run failed jobs** on that run. The re-run reuses the run's
   artifact, kept 90 days, and waits at the gate again.
3. **Otherwise, cut a new patch version.**

The republish takes the version from the tag, after its `PUBLISH_TAG_PREFIX`: it refuses another package's tag before
any request, and a tag that is the prefix followed by anything but the exact version cannot be republished either: the
job fails later, without publishing.

When rung 1 refuses, or a release run failed:

- **A TestPyPI job failed on index lag** (its JSON check gave up waiting, or the install check stopped with an error
  that names index lag): within 30 days, **Re-run failed jobs** on the release run. A re-run `Publish to TestPyPI`
  uploads the same bytes, from the run's artifact, which the index accepts, and both checks run again; then the PyPI
  job waits at the gate, and rung 1 has a release run to accept.
- **The install check failed for real** (the package does not install or import, and a re-run fails the same way): cut a
  new patch version. Neither a re-run nor a same-version re-cut can fix it: the index refuses new bytes under a file
  name it already holds.
- **A Release that was re-cut**: never delete a Release and publish it again to retry. The re-cut's run builds again,
  TestPyPI refuses its files, and that failed run is then the tag's newest release run, so rung 1 refuses the tag for
  good. Rung 2 still applies to the original release run, within 30 days of it; then rung 3.
- **A version a plain dispatch published** (below): no release run of its tag passed its TestPyPI jobs, so rung 1
  refuses it. Rung 2 applies to that dispatch run, within 30 days of it; then rung 3.

A failed `Verify PyPI serves exactly these files` comes after the upload, so the run's files are already on PyPI. The
step runs in two jobs, `Publish to PyPI` (in the release run, or a plain dispatch's re-fire) and the republish job (in
rung 1's dispatch run), and what it needs depends on why it failed:

- **PyPI lists a file this run did not build, or one with another sha256**: another upload put it there. Compare PyPI's
  JSON for the version with TestPyPI's to find which one; only a new patch version ships tested files under a clean
  version.
- **The check gave up waiting** (PyPI's JSON did not list the files in time): index lag, not a wrong file. **Re-run
  failed jobs** on the run whose check gave up (the release run, a plain dispatch's re-fire, or rung 1's dispatch run),
  within 30 days of that run: the job uploads the same bytes again, which the index accepts, and repeats the check.
  Re-running it needs a fresh `pypi` approval.

**Never "Re-run all jobs"** on a publish run, nor `gh run rerun <id>` without `--failed`. Dispatch from the tag: both
publish environments admit tags only.

**A dispatch without `republish_from_testpypi` is only for a release that never reached TestPyPI.** It re-fires the
whole publish and builds again, from the package's own tag alone. If that re-fire puts its own files on TestPyPI and
its PyPI job then fails, the ladder above targets the wrong run: rung 1 refuses the version, since no release run of
its tag passed its TestPyPI jobs, and rung 2 on the release run re-uploads a build TestPyPI rejects. Use **Re-run failed
jobs** on the dispatch run instead, within 30 days of it.

**Each rung acts on the workflow file as of the tag.** A tag cut before the one-artifact publish change merged
(juniper-recurrence#236, 2026-11-15), every release so far included, has the old workflow: no republish job,
`skip-existing` on both uploads, no sha256 check, a rebuild on every dispatch, and a 7-day artifact. Never dispatch
such a tag: a plain dispatch would rebuild, skip the TestPyPI upload, test TestPyPI's original files and then push the
rebuild to PyPI (the juniper-ci-tools 0.9.1 defect, which "Re-run all jobs" caused there). For such a tag, use **Re-run
failed jobs** on its release run within 7 days of it; otherwise cut a new patch version. A tag cut after #236 and before
this change runs rung 1 without the check of the release run's TestPyPI jobs.

## Ecosystem

Part of the [Juniper](https://github.com/pcalnon) ML research platform. See the
WS-4b build plan (`notes/JUNIPER_RECURRENCE_WS4B_APP_BUILD_PLAN_2026-06-15.md` in
`juniper-ml`) for the design of record.
