# juniper-recurrence

[![PyPI — juniper-recurrence](https://img.shields.io/pypi/v/juniper-recurrence?label=juniper-recurrence)](https://pypi.org/project/juniper-recurrence/)
[![PyPI — juniper-recurrence-model](https://img.shields.io/pypi/v/juniper-recurrence-model?label=juniper-recurrence-model)](https://pypi.org/project/juniper-recurrence-model/)
[![PyPI — juniper-recurrence-client](https://img.shields.io/pypi/v/juniper-recurrence-client?label=juniper-recurrence-client)](https://pypi.org/project/juniper-recurrence-client/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**Time-series regression with a recurrent neural network that handles irregular sampling natively.**

`juniper-recurrence` fits and serves a [**Legendre Memory Unit (LMU)**](#the-model-in-one-minute)
whose memory is discretised *in closed form at each step's real elapsed time* `Δt`. Series sampled
on an uneven time grid — financial ticks, sensor logs, clinical events — are consumed as-is, with no
resampling onto a uniform grid. The recurrent memory is **fixed**; only a readout is trained, **in
closed form, with no backpropagation-through-time** — so training is fast, deterministic, and exactly
reproducible.

The repository ships this capability at three layers you can adopt independently — a **model
library**, an **HTTP service**, and a **Python client** — plus a benchmark harness. All three are
published to PyPI.

---

## What it's for

Most recurrent models (RNN, LSTM, GRU, Transformer) assume samples arrive at a fixed cadence — one
step per unit of time. Real time series often don't: markets trade in bursts, sensors drop packets,
clinical measurements happen when a patient is seen. The usual workarounds (binning, forward-filling,
interpolating onto a grid) distort the signal and discard the timing information itself.

`juniper-recurrence` treats the **time gap between samples as a first-class input**. It targets
**regression over time** — predicting a continuous value from a window of irregularly-spaced
observations — and is designed for workloads where you want:

- **Native irregular `Δt`** — the elapsed time per step drives the recurrence directly.
- **Deterministic, reproducible training** — a closed-form least-squares solve, not stochastic
  gradient descent. Same data in, identical model out. No GPU required for the default path.
- **A choice of layer** — call the model in-process, run it as a microservice, or talk to that
  service over HTTP from another application.

> **New to Juniper?** Juniper is a multi-package ML research platform built around constructive and
> recurrent neural networks. `juniper-recurrence` is its **recurrent / continuous-time** application —
> the structural sibling of [`juniper-cascor`](https://github.com/pcalnon/juniper-cascor) (a
> stateless, feed-forward, classification-first network). Where cascor grows a network for
> classification, recurrence remembers the past over a real time axis for regression. You don't need
> the rest of the platform to use this repo — everything it depends on installs automatically from
> PyPI.

---

## The model in one minute

A **Legendre Memory Unit** maintains a small linear state that holds a sliding-window summary of its
input, projected onto Legendre polynomials. That linear memory obeys a fixed differential equation
whose matrices never change and are never trained.

Two consequences make this model unusual:

1. **Δt-native by construction.** Because the memory is *linear*, its exact discrete update is a
   matrix exponential of the (fixed) state matrix evaluated at the real step gap `Δt`. There is no ODE
   solver and no differentiating through one — the dataset's `Δt` channel *is* the discretisation
   step. Uneven sampling is handled exactly, not approximated.
2. **Closed-form training, no BPTT.** The memory is frozen, so the only trained surface is the
   **readout** that maps the memory state to the target. With a linear readout that's an ordinary
   least-squares solve: deterministic, fast, and free of the instabilities of training recurrence by
   backpropagation-through-time.

The readout is a **configurable spectrum**, so you can trade simplicity for capacity without changing
the memory:

| Rung | Readout | How it's fit | Reach for it when |
|------|---------|--------------|-------------------|
| **0 / 1** | Linear, optionally ridge-regularised (GCV-selected λ) | closed-form least squares | the default — fast, deterministic, no tuning |
| **2a** | Random Fourier Features + ridge (numpy) | closed-form least squares on lifted features | the target is nonlinear but you still want a deterministic solve |
| **2b** | Torch MLP (optional `[torch]` extra) | gradient descent | you need maximum readout capacity |

The model implements the shared [`juniper-model-core`](https://github.com/pcalnon/juniper-ml)
`TrainableModel` interface and passes its conformance kit unchanged — it was the first non-cascor
model to validate that shared seam.

For the full derivation, design rationale, and evaluation, see the design of record:
[`JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md)
(in `juniper-ml`).

---

## What's in the repository

A four-part monorepo following the Juniper "model family" layout — each independently-publishable
package lives in a same-named subdirectory.

| Path | What it is | PyPI |
|------|-----------|------|
| [`juniper-recurrence-model/`](./juniper-recurrence-model/) | **Model library.** The Δt-native LMU memory unit and `LMURegressor` (the `juniper-model-core` model). numpy-only at the core. | [`juniper-recurrence-model`](https://pypi.org/project/juniper-recurrence-model/) |
| [`juniper-recurrence/`](./juniper-recurrence/) | **Application.** FastAPI + CLI service wrapping the model on the shared [`juniper-service-core`](https://pypi.org/project/juniper-service-core/) framework. | [`juniper-recurrence`](https://pypi.org/project/juniper-recurrence/) |
| [`juniper-recurrence-client/`](./juniper-recurrence-client/) | **HTTP client.** A lean `requests`-based client for the service, mirroring the other Juniper backend clients. | [`juniper-recurrence-client`](https://pypi.org/project/juniper-recurrence-client/) |
| [`bench/`](./bench/) | **Benchmark harness.** Datasets, baselines, and evaluation bands. Not published. | — |

Datasets are *never* generated or vendored here — sequence data is produced by
[`juniper-data`](https://github.com/pcalnon/juniper-data) and pulled in through
[`juniper-data-client`](https://pypi.org/project/juniper-data-client/).

---

## Quick start

### 1. The model library (fastest path)

Pure `pip` + numpy, no service required.

```bash
pip install juniper-recurrence-model
```

```python
import numpy as np
from juniper_recurrence_model import LMURegressor, LMUSerializer

# n sequences, T steps each, F features per step.
n, T, F = 48, 6, 3
rng = np.random.default_rng(0)
X = rng.normal(size=(n, T, F))
y = X.reshape(n, -1) @ rng.normal(size=(T * F, 1))

# Per-step time gaps (irregular). dt[:, 0] = 0; later columns are the real elapsed time.
dt = np.zeros((n, T))
dt[:, 1:] = rng.integers(1, 4, size=(n, T - 1))

model = LMURegressor(d=6)            # d = LMU memory order; theta inferred from dt at fit time
result = model.fit(X, y, dt=dt)      # closed-form readout solve — deterministic, no BPTT
preds = model.predict(X, dt=dt)      # (n, 1)
print(result.final_metrics["r2"])

LMUSerializer().save(model, "/tmp/lmu")   # lossless round-trip to /tmp/lmu.npz
```

`dt` (Δt) and the readout mask both default to uniform gaps and the final step, so the plain
`model.predict(X)` interface works too. See the
[model README](./juniper-recurrence-model/README.md) for the raw `VariableStepLMUMemory` unit and the
readout-spectrum API.

### 2. The service

```bash
pip install juniper-recurrence
juniper-recurrence serve --host 127.0.0.1 --port 8210
```

Train on a [`juniper-data`](https://github.com/pcalnon/juniper-data) dataset and inspect the model:

```bash
curl -sX POST localhost:8210/v1/train \
  -H 'Content-Type: application/json' \
  -d '{"dataset": {"dataset_id": "<id>"}, "d": 16}'
curl -s localhost:8210/v1/model
```

Training runs **inline** — a one-shot closed-form solve — so `POST /v1/train` returns the result in
the response; there are no background jobs or WebSocket streams in v1. There's also a headless CLI
(`juniper-recurrence train …`) that fits and persists a model without a server. Full route reference,
configuration, and Docker notes are in the [application README](./juniper-recurrence/README.md).

### 3. The client

```bash
pip install juniper-recurrence-client
```

```python
from juniper_recurrence_client import JuniperRecurrenceClient

client = JuniperRecurrenceClient("http://localhost:8211", api_key="…")
client.train(name="equities", d=16)
client.predict(dataset_id="ds-1")

# Walk-forward cross-validation over the dataset's full split
result = client.crossval(name="equities", n_folds=4, scheme="expanding", embargo=2)
print(result["eval_aggregate"])
```

See the [client README](./juniper-recurrence-client/README.md) for the full method surface,
authentication, and the error hierarchy.

---

## HTTP API

Every `/v1/*` route requires an `X-API-Key` header when API keys are configured; health and docs are
always exempt.

| Route | Method | Purpose |
|-------|--------|---------|
| `/v1/train` | POST | Train the LMU on a dataset (synchronous closed-form solve). |
| `/v1/training/status` | GET | `idle` / `trained`, last metrics, and training events. |
| `/v1/predict` | POST | Predictions for inline `X` (+ `dt`) or a dataset reference. |
| `/v1/crossval` | POST | Walk-forward cross-validation (expanding / sliding, with embargo). |
| `/v1/crossval/status` | GET | Cross-validation run state + aggregate results. |
| `/v1/model` | GET | Current model topology + regression metrics. |
| `/v1/dataset` | GET | Descriptor of the trained-on dataset. |
| `/v1/health`, `/v1/health/ready` | GET | Liveness / readiness (exempt). |
| `/docs` | GET | OpenAPI / Swagger UI (exempt). |

Configuration reads the `JUNIPER_RECURRENCE_` environment namespace (e.g.
`JUNIPER_RECURRENCE_PORT`, default `8210`) and honours Docker `_FILE` secret indirection. When no API
keys are configured, authentication is disabled (development default). The
[application README](./juniper-recurrence/README.md#configuration) has the full variable table.

---

## How it fits the Juniper platform

`juniper-recurrence` is the glue and the network surface; the heavy lifting is shared, reusable
packages it consumes from PyPI:

```text
juniper-data ──datasets──▶ juniper-data-client ──▶ ─────────────────────────┐
                                                                            │
juniper-service-core ──create_app + lifecycle──▶ ───────────────────────────┼───┐
                                                                            │   │
juniper-model-core ──TrainableModel seam──▶ juniper-recurrence-model ──▶ ───┘   │
                                                                                │
   ┌────────────────────────────────────────────────────────────────────────────┘
   │
   └───▶ juniper-recurrence (app) ──HTTP──▶ juniper-recurrence-client ──▶ juniper-canopy (dashboard + visualization)
```

- **[`juniper-service-core`](https://pypi.org/project/juniper-service-core/)** — the FastAPI app
  factory and training lifecycle. `juniper-recurrence` was its first real consumer.
- **[`juniper-model-core`](https://github.com/pcalnon/juniper-ml)** — the shared `TrainableModel`
  seam the regressor implements.
- **[`juniper-data`](https://github.com/pcalnon/juniper-data)** / `juniper-data-client` — the source
  of all sequence datasets (the 3-D windowed irregular-`Δt` contract).
- **[`juniper-observability`](https://pypi.org/project/juniper-observability/)** — optional Prometheus
  `/metrics` and request-ID propagation (guarded imports; the app and client run without it).

It does **not** depend on the rest of the platform at runtime beyond these packages — installing any
of the three published packages pulls everything it needs.

---

## Status

**Live.** All three packages are published to PyPI and the model passes the `juniper-model-core`
conformance kit. The application exposes the train / predict / model / dataset / cross-validation
surface on `juniper-service-core`. Current versions are shown by the badges above; per-package history
lives in each package's `CHANGELOG.md`.

---

## Development

Each package is tested from its own subdirectory (CI runs the Python 3.12 / 3.13 / 3.14 matrix at 90%
coverage):

```bash
cd juniper-recurrence-model  && pip install -e ".[test]"               && python -m pytest
cd juniper-recurrence        && pip install -e ".[test,observability]" && python -m pytest
cd juniper-recurrence-client && pip install -e ".[test]"               && python -m pytest

# Benchmark harness — run from the repo root so `import bench` resolves
pip install -e "juniper-recurrence/.[test,bench]" && python -m pytest bench/
```

See [`AGENTS.md`](./AGENTS.md) for the full contributor guide (conventions, per-package publishing,
and the design-of-record links).

---

## License

MIT — see [LICENSE](./LICENSE).
