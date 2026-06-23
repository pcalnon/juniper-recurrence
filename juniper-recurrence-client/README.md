# juniper-recurrence-client

HTTP client for the **juniper-recurrence** service — the FastAPI app wrapping the Δt-native LMU
recurrence model and its cross-validation API. A lean `requests`-based client mirroring
[`juniper-data-client`](https://github.com/pcalnon/juniper-data-client) and
[`juniper-cascor-client`](https://github.com/pcalnon/juniper-cascor-client), so consumers
(notably **juniper-canopy**'s recurrence backend adapter) drive every Juniper backend the same way.

## Install

```bash
pip install juniper-recurrence-client          # once published
pip install -e ".[test]"                        # local development
```

`requests`-only at the core; `pip install juniper-recurrence-client[observability]` adds the
optional `juniper-observability` integration (X-Request-ID propagation + the `on_request` hook).

> **Port:** `8211` is the juniper-recurrence service's default host port; under Docker it maps to the container's `8210` (see the app README). Point the client at whichever `host:port` the service is published on.

## Quick start

```python
from juniper_recurrence_client import JuniperRecurrenceClient

client = JuniperRecurrenceClient("http://localhost:8211", api_key="…")

# Train the LMU regressor on a dataset (by id / name / generator)
client.train(name="equities", d=16)

# Predict — inline X with Δt, or a dataset reference
client.predict(dataset_id="ds-1")

# Walk-forward cross-validation over the dataset's _full split
result = client.crossval(name="equities", n_folds=4, scheme="expanding", embargo=2)
print(result["eval_aggregate"])

# Inspect
client.get_model()        # topology + metrics
client.training_status()  # state + events
client.is_ready()         # readiness probe
```

## API surface

| Method | Endpoint |
|--------|----------|
| `train(*, dataset_id / name / generator, params, split, d, theta, ridge)` | `POST /v1/train` |
| `training_status()` | `GET /v1/training/status` |
| `predict(*, X / dt / target_dt / seq_lengths, or a dataset ref)` | `POST /v1/predict` |
| `crossval(*, n_folds, scheme, embargo, min_train, dataset ref, d, theta, ridge)` | `POST /v1/crossval` |
| `crossval_status()` | `GET /v1/crossval/status` |
| `get_model()` | `GET /v1/model` |
| `get_dataset()` | `GET /v1/dataset` |
| `health_check()` / `is_ready()` / `wait_for_ready()` | `GET /v1/health[/ready]` |

## Authentication

Pass `api_key=…`, or set `JUNIPER_RECURRENCE_API_KEY` (or the Docker-secret
`JUNIPER_RECURRENCE_API_KEY_FILE`, a path whose stripped contents are the key). The key is sent
as the `X-API-Key` header. Note the asymmetry: the **server** reads the *plural*
`JUNIPER_RECURRENCE_API_KEYS` (its accepted set); the **client** sends one key under the
*singular* env var.

## Errors

All errors derive from `JuniperRecurrenceClientError`: `JuniperRecurrenceConnectionError`,
`JuniperRecurrenceTimeoutError`, `JuniperRecurrenceNotFoundError` (404),
`JuniperRecurrenceConflictError` (409 — a run already in progress, or no trained model yet),
`JuniperRecurrenceValidationError` (400/422), `JuniperRecurrenceConfigurationError`.

## License

MIT — see [LICENSE](https://github.com/pcalnon/juniper-recurrence/blob/main/LICENSE).
