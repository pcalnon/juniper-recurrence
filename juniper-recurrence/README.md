# juniper-recurrence

**Project**: Juniper — Cascade Correlation Neural Network Research Platform
**Application**: juniper-recurrence (FastAPI + CLI service)
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.1.0

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

Once running:

* `GET /v1/health` → `{"status": "ok"}` (liveness)
* `GET /v1/health/ready` → `{"status": "ready"}` (readiness)
* `GET /docs` → OpenAPI / Swagger UI

The training / predict / model / dataset routes and the headless
`juniper-recurrence train` subcommand land in WS-4b PR-2.

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

## Development

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Ecosystem

Part of the [Juniper](https://github.com/pcalnon) ML research platform. See the
WS-4b build plan (`notes/JUNIPER_RECURRENCE_WS4B_APP_BUILD_PLAN_2026-06-15.md` in
`juniper-ml`) for the design of record.
