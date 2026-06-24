"""The recurrence benchmark (C2): walk-forward CV of the Δt-native LMU vs a fixed-Δt control,
naive persistence, and a linear ridge baseline, on irregular- and regular-Δt datasets.

Run from the repo root:  ``python -m bench.run_benchmark``
Writes ``bench/results/<dataset>.json`` + ``bench/results/REPORT.md`` (reproducible from the seeds).

Design + acceptance bands: juniper-ml ``notes/JUNIPER_RECURRENCE_EVALUATION_DESIGN_2026-06-18.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from juniper_model_core.crossval import cross_validate, walk_forward_folds
from juniper_recurrence_model import LMURegressor, MLPReadoutSpec, RFFReadoutSpec

from bench import baselines, datasets

# DP-3 Rung 2b: the torch MLP readout row runs only when the optional [torch] extra is installed
# (install ``.[bench,bench-torch]``). Probe with ``find_spec`` rather than importing torch, so the
# offline bench and its torch-free CI lane keep working — the MLP row is simply omitted when torch is
# absent (mirrors the equities row's graceful skip rather than aborting the whole dataset).
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

_RESULTS = Path(__file__).resolve().parent / "results"
_HEADLINE_D = 16
_D_GRID = (8, 16, 32)
_N_FOLDS = 5
_EMBARGO = 2
# Regularized-readout variant. The LMURegressor default ridge=0 (plain lstsq) is an intentional
# juniper-model-core conformance setting ("overfit-tiny exactly") that overfits low-signal real
# data; this variant gives a fair real-data comparison (it reaches the efficient-market ceiling on
# equities). The ratified ridge=0 primary bands are deliberately left untouched. See
# juniper-recurrence#28 + the juniper-ml findings doc §3.2.
_RIDGE_VARIANT = 1.0


def _run_cv(
    factory: Any, ds: datasets.Dataset, aux: dict[str, np.ndarray], folds: list
) -> dict[str, dict[str, float]]:
    result = cross_validate(factory, ds.X, ds.y, folds, aux=aux)
    return {
        "mean": result.eval_aggregate,
        "std": result.eval_std,
        "n_folds": len(result.folds),
    }


def run_dataset(ds: datasets.Dataset) -> dict[str, Any]:
    """Run all models on one dataset through identical walk-forward folds."""
    theta = float(
        np.median(ds.dt.sum(axis=1))
    )  # data-driven memory window, held constant across LMU variants
    if not np.isfinite(theta) or theta <= 0:
        theta = float(ds.X.shape[1])
    folds = walk_forward_folds(ds.X.shape[0], n_folds=_N_FOLDS, embargo=_EMBARGO)
    aux_real = {"dt": ds.dt, "target_dt": ds.target_dt}
    aux_fixed = {"dt": baselines.uniform_dt(ds.dt), "target_dt": ds.target_dt}

    models: dict[str, dict[str, Any]] = {}
    models[f"lmu_var_d{_HEADLINE_D}"] = _run_cv(
        lambda i: LMURegressor(d=_HEADLINE_D, theta=theta), ds, aux_real, folds
    )
    models[f"lmu_fixed_d{_HEADLINE_D}"] = _run_cv(
        lambda i: LMURegressor(d=_HEADLINE_D, theta=theta), ds, aux_fixed, folds
    )
    models["naive_persistence"] = _run_cv(
        lambda i: baselines.NaivePersistence(), ds, aux_real, folds
    )
    models["linear_ridge"] = _run_cv(
        lambda i: baselines.LinearRidge(ridge=1e-3), ds, aux_real, folds
    )
    # Regularized-readout LMU (ridge>0): a fair comparison on low-signal real data, where the ridge=0
    # default overfits. Both var-Δt and fixed-Δt so the Δt contribution stays isolated (recurrence#28).
    models[f"lmu_var_d{_HEADLINE_D}_ridge{_RIDGE_VARIANT}"] = _run_cv(
        lambda i: LMURegressor(d=_HEADLINE_D, theta=theta, ridge=_RIDGE_VARIANT),
        ds,
        aux_real,
        folds,
    )
    models[f"lmu_fixed_d{_HEADLINE_D}_ridge{_RIDGE_VARIANT}"] = _run_cv(
        lambda i: LMURegressor(d=_HEADLINE_D, theta=theta, ridge=_RIDGE_VARIANT),
        ds,
        aux_fixed,
        folds,
    )
    # Nonlinear RFF readout (DP-3 Rung 2a): standardize(M) → random Fourier features → GCV ridge.
    # On the near-linear datasets it ties the linear readout at the ceiling; on the delay_product
    # capacity dataset (a bilinear target the linear readout provably can't fit) it shows a clear
    # nonlinear≫linear r² gap (design §8a). Variable-Δt only — readout capacity is orthogonal to the
    # Δt thesis, so one row isolates it without doubling the table.
    models[f"lmu_var_d{_HEADLINE_D}_rff"] = _run_cv(
        lambda i: LMURegressor(d=_HEADLINE_D, theta=theta, readout=RFFReadoutSpec()),
        ds,
        aux_real,
        folds,
    )
    # Nonlinear torch MLP readout (DP-3 Rung 2b) — only when the optional [torch] extra is installed.
    # Trained full-budget (no early stopping: walk-forward CV passes no held-out split, and reusing the
    # eval fold as validation would leak), so weight_decay is its only regularizer. Like RFF, a capacity
    # probe vs the linear readout — on delay_product it fits the bilinear target a linear readout
    # provably cannot. Default (untuned) MLPReadoutSpec() for a reproducible, non-cherry-picked row.
    if _TORCH_AVAILABLE:
        models[f"lmu_var_d{_HEADLINE_D}_mlp"] = _run_cv(
            lambda i: LMURegressor(
                d=_HEADLINE_D, theta=theta, readout=MLPReadoutSpec()
            ),
            ds,
            aux_real,
            folds,
        )
    # d-sensitivity sweep for the variable-Δt LMU (headline d already covered above)
    sweep = {
        f"lmu_var_d{d}": _run_cv(
            lambda i, d=d: LMURegressor(d=d, theta=theta), ds, aux_real, folds
        )
        for d in _D_GRID
        if d != _HEADLINE_D
    }

    return {
        "name": ds.name,
        "grid": ds.grid,
        "n_windows": int(ds.X.shape[0]),
        "theta": theta,
        "models": {**models, **sweep},
    }


def evaluate_bands(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate the ratified OQ-14 bands (``primary``) + the extension findings (``primary=False``).

    Primary bands are scored only on the pre-registered ``PRIMARY_DATASETS`` and drive the overall
    verdict; the noise-sweep / real-data extensions get the same measurements tagged informational
    (not pass/fail — they were never pre-registered against these thresholds, per the DP-5 guardrail).
    """
    bands: list[dict[str, Any]] = []

    def rmse(ds_name: str, model: str) -> float:
        return results[ds_name]["models"][model]["mean"]["rmse"]

    def r2(ds_name: str, model: str) -> float:
        return results[ds_name]["models"][model]["mean"]["r2"]

    def r2_std(ds_name: str, model: str) -> float:
        return results[ds_name]["models"][model]["std"]["r2"]

    var, fixed = f"lmu_var_d{_HEADLINE_D}", f"lmu_fixed_d{_HEADLINE_D}"

    # Band 1 — Δt thesis: var-Δt LMU >= 25% lower RMSE than fixed-Δt on every irregular-Δt dataset
    # (primary = the canonical irregular_sine; noise variants + equities_seq are informational).
    for name, res in results.items():
        if res["grid"] != "irregular":
            continue
        fr, vr = rmse(name, fixed), rmse(name, var)
        improvement = (fr - vr) / fr if fr > 0 else 0.0
        bands.append(
            {
                "band": f"1 — Δt thesis ({name}): var-Δt ≥25% lower RMSE than fixed-Δt",
                "value": f"{improvement:+.1%} RMSE reduction (fixed={fr:.4f} → var={vr:.4f})",
                "pass": improvement >= 0.25,
                "primary": name == "irregular_sine",
            }
        )

    # Band 2 — beats naive, matches/beats linear (every dataset).
    for name in results:
        beats_naive = r2(name, var) > r2(name, "naive_persistence")
        matches_linear = r2(name, var) >= r2(name, "linear_ridge") or abs(
            r2(name, var) - r2(name, "linear_ridge")
        ) <= (r2_std(name, var) + r2_std(name, "linear_ridge"))
        bands.append(
            {
                "band": f"2 — {name}: LMU beats naive & matches/beats linear (r2)",
                "value": f"LMU={r2(name, var):.4f}  naive={r2(name, 'naive_persistence'):.4f}  linear={r2(name, 'linear_ridge'):.4f}",
                "pass": bool(beats_naive and matches_linear),
                "primary": name in datasets.PRIMARY_DATASETS,
            }
        )

    # Band 2b (informational) — real-data fairness. The headline ridge=0 LMU (Band 2) is a
    # conformance default that overfits low-signal real data; with a regularized readout the
    # Δt-LMU matches/beats linear on the *stationary* target. See recurrence#28 + findings §3.2.
    rkey = f"lmu_var_d{_HEADLINE_D}_ridge{_RIDGE_VARIANT}"
    for name in results:
        if name == "equities_seq" and rkey in results[name]["models"]:
            vr, lr = r2(name, rkey), r2(name, "linear_ridge")
            bands.append(
                {
                    "band": f"2b — {name} (regularized readout, ridge={_RIDGE_VARIANT}): Δt-LMU matches/beats linear (r2)",
                    "value": f"LMU(ridge)={vr:.4f}  linear={lr:.4f}  [ridge0 LMU={r2(name, var):.4f}]",
                    "pass": bool(vr >= lr),
                    "primary": False,
                }
            )

    # Band 3 — no regular-grid penalty: var ≈ fixed on regular-Δt datasets (within 10% RMSE).
    for name, res in results.items():
        if res["grid"] == "regular":
            fr, vr = rmse(name, fixed), rmse(name, var)
            penalty = abs(vr - fr) / fr if fr > 0 else 0.0
            bands.append(
                {
                    "band": f"3 — {name} (regular): var-Δt ≈ fixed-Δt (≤10% RMSE gap)",
                    "value": f"{penalty:.1%} gap (fixed={fr:.4f}, var={vr:.4f})",
                    "pass": penalty <= 0.10,
                    "primary": name == "multi_sine",
                }
            )

    # Band 4 (informational) — DP-3 readout capacity: the nonlinear RFF readout vs the linear readout
    # on the same Δt-aware LMU. On the delay_product capacity dataset (a bilinear target the linear
    # readout provably can't fit) expect a clear nonlinear≫linear gap; on the near-linear datasets
    # expect a tie. Informational only — never pre-registered against the ratified bands. Design §8a.
    rff_key = f"lmu_var_d{_HEADLINE_D}_rff"
    for name in results:
        if rff_key not in results[name]["models"]:
            continue
        rff_r2, lin_r2 = r2(name, rff_key), r2(name, var)
        is_capacity = name == "delay_product"
        bands.append(
            {
                "band": f"4 — {name}: RFF readout vs linear readout ({'capacity gap' if is_capacity else 'tie'})",
                "value": f"rff={rff_r2:.4f}  linear={lin_r2:.4f}  gap={rff_r2 - lin_r2:+.4f}",
                "pass": (rff_r2 - lin_r2) >= 0.10
                if is_capacity
                else abs(rff_r2 - lin_r2) <= 0.10,
                "primary": False,
            }
        )

    # Band 4b (informational) — DP-3 Rung 2b: the torch MLP readout vs the linear readout, the same
    # capacity probe as Band 4 (RFF). Present only when the bench ran with the [torch] extra. On
    # delay_product expect a clear capacity gap; on near-linear datasets the untuned full-budget MLP
    # (no early stopping in CV) may trail the linear readout — informational, never pre-registered.
    mlp_key = f"lmu_var_d{_HEADLINE_D}_mlp"
    for name in results:
        if mlp_key not in results[name]["models"]:
            continue
        mlp_r2, lin_r2 = r2(name, mlp_key), r2(name, var)
        is_capacity = name == "delay_product"
        bands.append(
            {
                "band": f"4b — {name}: MLP readout vs linear readout ({'capacity gap' if is_capacity else 'comparison'})",
                "value": f"mlp={mlp_r2:.4f}  linear={lin_r2:.4f}  gap={mlp_r2 - lin_r2:+.4f}",
                "pass": (mlp_r2 - lin_r2) >= 0.10
                if is_capacity
                else abs(mlp_r2 - lin_r2) <= 0.10,
                "primary": False,
            }
        )

    return bands


def _render_report(
    results: dict[str, dict[str, Any]],
    bands: list[dict[str, Any]],
    skipped: dict[str, str] | None = None,
) -> str:
    lines = [
        "# Recurrence Benchmark — Results",
        "",
        "Generated by `python -m bench.run_benchmark` (reproducible from the dataset seeds).",
        "",
    ]
    if skipped:
        lines.append(
            "> **Skipped datasets:** "
            + "; ".join(f"`{n}` ({why})" for n, why in skipped.items())
            + "."
        )
        lines.append("")
    lines += ["## Per-dataset metrics (walk-forward CV; mean ± std across folds)", ""]
    for name, res in results.items():
        lines.append(
            f"### `{name}` ({res['grid']}-Δt, n_windows={res['n_windows']}, θ={res['theta']:.3f}, {_N_FOLDS} folds)"
        )
        lines.append("")
        lines.append("| model | rmse | mae | r2 |")
        lines.append("|---|---|---|---|")
        for model, m in res["models"].items():
            mean, std = m["mean"], m["std"]
            lines.append(
                f"| `{model}` | {mean['rmse']:.4f} ± {std['rmse']:.4f} | {mean['mae']:.4f} | {mean['r2']:.4f} ± {std['r2']:.4f} |"
            )
        lines.append("")
    primary = [b for b in bands if b.get("primary")]
    secondary = [b for b in bands if not b.get("primary")]
    lines.append(
        "## Acceptance bands (OQ-14 — ratified; scored on the pre-registered datasets)"
    )
    lines.append("")
    lines.append("| band | result | verdict |")
    lines.append("|---|---|---|")
    for b in primary:
        lines.append(
            f"| {b['band']} | {b['value']} | {'✅ PASS' if b['pass'] else '❌ MISS'} |"
        )
    lines.append("")
    overall = (
        "PASS"
        if primary and all(b["pass"] for b in primary)
        else "PARTIAL — see misses above"
    )
    lines.append(f"**Overall (ratified bands):** {overall}")
    lines.append("")
    if secondary:
        lines.append(
            "## Extension findings (informational — noise sweep + real data; not scored against the ratified bands)"
        )
        lines.append("")
        lines.append("| measurement | result | Δt-positive |")
        lines.append("|---|---|---|")
        for b in secondary:
            lines.append(
                f"| {b['band']} | {b['value']} | {'✅' if b['pass'] else '➖'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for name, factory in datasets.DATASETS.items():
        print(f"[bench] {name} …", flush=True)
        try:
            ds = factory()
            res = run_dataset(ds)
        except Exception as exc:  # noqa: BLE001 — a dataset failing (e.g. networked equities_seq) must not abort the run
            skipped[name] = type(exc).__name__
            print(f"[bench] {name} SKIPPED — {type(exc).__name__}: {exc}", flush=True)
            continue
        results[name] = res
        (_RESULTS / f"{name}.json").write_text(
            json.dumps(res, indent=2, sort_keys=True)
        )

    bands = evaluate_bands(results)
    report = _render_report(results, bands, skipped)
    (_RESULTS / "REPORT.md").write_text(report)
    print("\n" + report)
    print(
        f"[bench] wrote {_RESULTS}/REPORT.md + per-dataset JSON"
        + (f" ({len(skipped)} skipped)" if skipped else "")
    )


if __name__ == "__main__":
    main()
