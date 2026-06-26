"""Domain Prometheus metrics for the juniper-recurrence service (OBS-04).

Train / predict / cross-validation counters plus last-value gauges (run duration and
the final / eval-aggregate metrics such as ``r2`` / ``mse``), registered through the
idempotent ``juniper_observability.register_or_reuse`` helper so they surface
automatically via the ``/metrics`` Prometheus endpoint wired in
:mod:`juniper_recurrence.app`.

``juniper-observability`` (and its ``prometheus-client`` dependency) is the optional
``[observability]`` extra -- exactly as ``app.py`` guards the ``/metrics`` mount -- so this
module degrades to no-ops when the extra is absent: the routers call the ``record_*``
helpers unconditionally and an observability-less deployment still trains and predicts,
just without domain metrics.

The service is single-worker and trains synchronously in-request (D-WS4b-2), so the
collectors register at import with no locking -- there is no concurrent-registration race
to guard (unlike juniper-cascor's threaded trainer, which lazy-inits behind a lock).

Design: ``notes/JUNIPER_RECURRENCE_METRICS_ENDPOINT_DESIGN_2026-06-18.md`` section 8 (juniper-ml).
"""

from __future__ import annotations

from collections.abc import Mapping

try:
    from juniper_observability import register_or_reuse
    from prometheus_client import Counter, Gauge

    _TRAIN_RUNS = register_or_reuse(Counter, "juniper_recurrence_train_runs_total", "Total completed training runs.")
    _PREDICT_REQUESTS = register_or_reuse(Counter, "juniper_recurrence_predict_requests_total", "Total completed prediction requests.")
    _CROSSVAL_RUNS = register_or_reuse(Counter, "juniper_recurrence_crossval_runs_total", "Total completed cross-validation runs.")
    _TRAIN_LAST_METRIC = register_or_reuse(Gauge, "juniper_recurrence_train_last_metric", "Most recent training final metric, keyed by metric name (e.g. r2, mse).", ["metric"])
    _TRAIN_LAST_DURATION = register_or_reuse(Gauge, "juniper_recurrence_train_last_duration_seconds", "Wall-clock duration of the most recent training run.")
    _CROSSVAL_LAST_METRIC = register_or_reuse(Gauge, "juniper_recurrence_crossval_last_metric", "Most recent cross-validation eval-aggregate metric, keyed by metric name.", ["metric"])
    _CROSSVAL_LAST_DURATION = register_or_reuse(Gauge, "juniper_recurrence_crossval_last_duration_seconds", "Wall-clock duration of the most recent cross-validation run.")

    ENABLED = True
except ImportError:  # pragma: no cover - optional [observability] extra absent; degrade to no-ops (mirrors app.py)
    ENABLED = False


def _set_last_metrics(gauge, values: Mapping[str, float]) -> None:
    """Set one labelled sample per numeric metric (non-numeric / bool values are skipped)."""
    for name, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            gauge.labels(metric=name).set(float(value))


def record_train(duration_seconds: float, final_metrics: Mapping[str, float]) -> None:
    """Record a completed training run: increment the counter; set duration + per-metric gauges."""
    if not ENABLED:
        return
    _TRAIN_RUNS.inc()
    _TRAIN_LAST_DURATION.set(duration_seconds)
    _set_last_metrics(_TRAIN_LAST_METRIC, final_metrics)


def record_predict() -> None:
    """Record a completed prediction request (increment the counter)."""
    if not ENABLED:
        return
    _PREDICT_REQUESTS.inc()


def record_crossval(duration_seconds: float, eval_aggregate: Mapping[str, float]) -> None:
    """Record a completed cross-validation run: increment the counter; set duration + per-metric gauges."""
    if not ENABLED:
        return
    _CROSSVAL_RUNS.inc()
    _CROSSVAL_LAST_DURATION.set(duration_seconds)
    _set_last_metrics(_CROSSVAL_LAST_METRIC, eval_aggregate)
