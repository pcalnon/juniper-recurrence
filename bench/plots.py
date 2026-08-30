"""Visual outputs for the recurrence benchmark — forecast-vs-truth, residuals, dataset shape.

Closes **G-5** of the CLI-experimentation plan (``notes/JUNIPER_2026-07-29_JUNIPER-ECOSYSTEM_
CASCOR-RECURRENCE-CLI-TEST-VALIDATION-EXPERIMENTATION-PLAN.md:217``), which recorded that
"recurrence has **zero plotting code**; there is no forecast-vs-truth, residual, or dataset
visual anywhere" — the analysis surface was the markdown bands report and nothing else.

Run it::

    python -m bench.plots                       # every dataset, into bench/plots/
    python -m bench.plots --out-dir /tmp/p      # elsewhere (cf. run_benchmark --results-dir)
    python -m bench.plots --datasets multi_sine irregular_sine
    python -m bench.plots --skip-forecast       # comparison + dataset figures only (no fitting)

Three figure families, and what each is honestly good for
---------------------------------------------------------
``dataset``      Δt distribution and the target series. Answers "what does this data actually
                 look like?" — in particular whether a row labelled *irregular* really has a
                 spread of gaps. Needs no model and no committed results.

``forecast``     Predicted vs true, plus residuals, on **one** walk-forward fold.
                 **This is illustrative, not the scored result.** ``cross_validate`` discards
                 predictions (``juniper_model_core/crossval/executor.py:136-137`` computes
                 ``y_pred``, scores it, and drops it), so a picture of predictions requires
                 re-fitting. This module therefore fits the LAST fold only, with the same
                 estimator and the same aux split the executor uses. The number to quote is
                 the mean ± std in ``REPORT.md``; this figure shows the *shape* of the error,
                 which an aggregate cannot.

``comparison``  Per-model r² across datasets, read straight from the committed
                 ``bench/results/*.json``. No recomputation, so it can never disagree with
                 the report it sits beside.

Why matplotlib is an optional extra
-----------------------------------
``[bench-plots]``, kept out of ``[bench]`` exactly like ``[bench-equities]`` and
``[bench-torch]``: the published app dist stays plot-free, and a bench run that only wants
numbers does not grow a plotting dependency. Absent, this module exits with an install hint
rather than an ImportError traceback — the same graceful-degradation contract
``run_benchmark.py`` gives the torch readout row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from juniper_model_core.crossval import walk_forward_folds
from juniper_recurrence_model import LMURegressor

from bench import datasets

_PLOTS = Path(__file__).resolve().parent / "plots"
_RESULTS = Path(__file__).resolve().parent / "results"
_HEADLINE_D = 16
_N_FOLDS = 5
_EMBARGO = 2
#: Lower bound of the interpretable r2 band. r2 = 0 is "as good as predicting the mean"; below
#: -1 the score carries no further comparative information, only magnitude of failure.
_R2_FLOOR = -1.0

_INSTALL_HINT = (
    "matplotlib is not installed. The plotting extra is deliberately separate from [bench]:\n"
    "    pip install 'juniper-recurrence[bench,bench-plots]'\n"
    "(or, from a checkout: pip install './juniper-recurrence[bench,bench-plots]')"
)


class PlotsUnavailable(RuntimeError):
    """Raised when the optional plotting extra is not installed."""


def _pyplot() -> Any:
    """Import pyplot on the Agg backend, or fail with an actionable message.

    The backend is forced BEFORE pyplot is imported: CI and the experiment host are headless,
    and matplotlib's default backend selection would otherwise try (and fail) to find a display.
    """
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via test monkeypatch
        raise PlotsUnavailable(_INSTALL_HINT) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _flat(y: np.ndarray) -> np.ndarray:
    """Collapse a (n, output_dim) target to (n,) for plotting.

    Every bench dataset is single-output today; averaging rather than indexing [:, 0] means a
    future multi-output row degrades to a summary line instead of silently plotting one column
    and labelling it as the target.
    """
    arr = np.asarray(y)
    return arr.reshape(arr.shape[0], -1).mean(axis=1) if arr.ndim > 1 else arr


def plot_dataset(ds: datasets.Dataset, out_dir: Path) -> Path:
    """Δt distribution + target series for one dataset."""
    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_dt, ax_y) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # dt[:, 0] is 0 by contract (the first step of a window has no predecessor); including it
    # would put a spike at zero in every dataset and flatten the distribution that matters.
    gaps = np.asarray(ds.dt)[:, 1:].ravel()
    ax_dt.hist(gaps, bins=40, color="#4C78A8")
    ax_dt.set_title(f"{ds.name} — per-step Δt ({ds.grid} grid)")
    ax_dt.set_xlabel("Δt")
    ax_dt.set_ylabel("count")
    ax_dt.text(
        0.99,
        0.95,
        f"mean {gaps.mean():.3f}  sd {gaps.std():.3f}",
        transform=ax_dt.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    y = _flat(ds.y)
    ax_y.plot(np.arange(y.shape[0]), y, lw=0.8, color="#333333")
    ax_y.set_title(f"{ds.name} — target by window index (n={y.shape[0]})")
    ax_y.set_xlabel("window index (chronological)")
    ax_y.set_ylabel("y")

    path = out_dir / f"{ds.name}_dataset.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_forecast_and_residuals(ds: datasets.Dataset, out_dir: Path, d: int = _HEADLINE_D) -> Path:
    """Predicted vs true and residuals on the LAST walk-forward fold.

    Deliberately one fold, and deliberately re-fitted: see the module docstring. The estimator,
    the fold construction and the aux split all mirror ``run_benchmark.run_dataset`` so the
    picture corresponds to the ``lmu_var_d{d}`` row of the report rather than to some other model.
    """
    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)

    theta = float(np.median(np.asarray(ds.dt).sum(axis=1)))
    if not np.isfinite(theta) or theta <= 0:
        theta = float(ds.X.shape[1])

    folds = walk_forward_folds(ds.X.shape[0], n_folds=_N_FOLDS, embargo=_EMBARGO)
    fold = folds[-1]
    train_idx = np.asarray(fold.train_idx)
    eval_idx = np.asarray(fold.eval_idx)

    model = LMURegressor(d=d, theta=theta)
    model.fit(
        ds.X[train_idx],
        ds.y[train_idx],
        dt=ds.dt[train_idx],
        target_dt=ds.target_dt[train_idx],
    )
    y_pred = np.asarray(model.predict(ds.X[eval_idx], dt=ds.dt[eval_idx], target_dt=ds.target_dt[eval_idx]))

    truth = _flat(ds.y[eval_idx])
    pred = _flat(y_pred)
    resid = pred - truth

    fig, (ax_f, ax_r) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    idx = np.arange(truth.shape[0])
    # Truth is drawn thick underneath and prediction thin on top. Equal weights make a good fit
    # look like ONE line and a reader cannot tell whether truth was plotted at all — on
    # multi_sine the residuals are ~1e-7, so the curves are genuinely indistinguishable.
    ax_f.plot(idx, truth, lw=3.0, color="#333333", alpha=0.35, label="truth", zorder=1)
    ax_f.plot(idx, pred, lw=1.1, color="#E45756", label=f"lmu_var_d{d}", zorder=2)
    ax_f.set_title(
        f"{ds.name} — forecast vs truth, fold {len(folds)} of {len(folds)} "
        f"(illustrative single fold; scored figures are in REPORT.md)"
    )
    ax_f.set_ylabel("y")
    ax_f.legend(loc="best", fontsize=9)

    ax_r.axhline(0.0, color="#888888", lw=0.8)
    ax_r.plot(idx, resid, lw=0.9, color="#4C78A8")
    # ``g`` not ``f``: a well-fit row has residuals around 1e-7, and %.4f renders those as
    # "0.0000" beside an axis labelled 1e-7 — a caption that contradicts its own plot.
    ax_r.set_title(f"residual (pred − truth) — mean {resid.mean():+.3g}, sd {resid.std():.3g}")
    ax_r.set_xlabel("eval-slice index (chronological)")
    ax_r.set_ylabel("residual")

    path = out_dir / f"{ds.name}_forecast.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_model_comparison(results_dir: Path, out_dir: Path, metric: str = "r2") -> Path | None:
    """Per-model ``metric`` across every dataset JSON in ``results_dir``.

    Returns ``None`` when there is nothing to plot, rather than writing an empty figure that
    would read as "all models scored zero".
    """
    files = sorted(p for p in results_dir.glob("*.json"))
    if not files:
        return None

    per_dataset: dict[str, dict[str, float]] = {}
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        models = doc.get("models")
        if not isinstance(models, dict):
            continue
        scores = {
            name: float(m["mean"][metric])
            for name, m in models.items()
            if isinstance(m, dict) and metric in (m.get("mean") or {})
        }
        if scores:
            per_dataset[doc.get("name", path.stem)] = scores
    if not per_dataset:
        return None

    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted({m for scores in per_dataset.values() for m in scores})
    ds_names = sorted(per_dataset)
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(ds_names)), 6), constrained_layout=True)
    base = np.arange(len(ds_names))
    for i, model in enumerate(model_names):
        # NaN (not 0.0) for a model a dataset never ran: 0.0 is a legitimate r2 value, so
        # substituting it would invent a scored result. NaN simply leaves the bar absent.
        vals = [per_dataset[d].get(model, np.nan) for d in ds_names]
        ax.bar(base + i * width, vals, width=width, label=model)

    ax.set_xticks(base + 0.4 - width / 2)
    ax.set_xticklabels(ds_names, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.legend(fontsize=8, ncols=2, loc="best")

    title = f"Recurrence bench — {metric} by model (walk-forward CV mean, from committed results)"
    if metric == "r2":
        # r2 is unbounded BELOW, and one catastrophic row destroys the shared axis: equities_seq
        # scores about -7000, which compresses every other dataset into an invisible line at zero
        # and makes the figure worse than no figure. Clip to the interpretable band and say so —
        # below -1 the only information is "worse than predicting the mean", which the floor
        # conveys just as well. The worst offender is NAMED so clipping never hides a result.
        scored = [(d, m, v) for d, s in per_dataset.items() for m, v in s.items()]
        off = [(d, m, v) for d, m, v in scored if v < _R2_FLOOR]
        ax.set_ylim(_R2_FLOOR, 1.05)
        if off:
            worst_d, worst_m, worst_v = min(off, key=lambda t: t[2])
            title += (
                f"\n{len(off)} of {len(scored)} scores clipped at r2={_R2_FLOOR:g} "
                f"(worst: {worst_d} / {worst_m}, r2={worst_v:.4g})"
            )
    ax.set_title(title, fontsize=10)

    path = out_dir / f"model_comparison_{metric}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    # argv threads explicitly rather than falling through to sys.argv — the cascor#486 class,
    # where an importing test runner's own arguments get parsed as the tool's.
    parser = argparse.ArgumentParser(description="Render benchmark figures (G-5).")
    parser.add_argument("--out-dir", type=Path, default=_PLOTS, help="Directory for the PNGs (default: bench/plots).")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_RESULTS,
        help="Committed per-dataset JSON to read for the comparison figure (default: bench/results).",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset names (default: all).")
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip the forecast/residual figures, which re-fit a model and are the slow part.",
    )
    args = parser.parse_args(argv)

    try:
        _pyplot()
    except PlotsUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    written: list[Path] = []
    comparison = plot_model_comparison(args.results_dir, args.out_dir)
    if comparison is not None:
        written.append(comparison)
    else:
        print(f"[plots] no usable results under {args.results_dir} — comparison figure skipped", flush=True)

    wanted = args.datasets if args.datasets else list(datasets.DATASETS)
    for name in wanted:
        factory = datasets.DATASETS.get(name)
        if factory is None:
            print(f"[plots] {name} SKIPPED — no such dataset", flush=True)
            continue
        try:
            ds = factory()
        except Exception as exc:  # noqa: BLE001 — a networked/optional dataset must not abort the run
            print(f"[plots] {name} SKIPPED — {type(exc).__name__}: {exc}", flush=True)
            continue
        written.append(plot_dataset(ds, args.out_dir))
        if not args.skip_forecast:
            try:
                written.append(plot_forecast_and_residuals(ds, args.out_dir))
            except Exception as exc:  # noqa: BLE001 — one unfittable row must not lose the others
                print(f"[plots] {name} forecast SKIPPED — {type(exc).__name__}: {exc}", flush=True)

    for path in written:
        print(f"[plots] wrote {path}", flush=True)
    print(f"[plots] {len(written)} figure(s) in {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
