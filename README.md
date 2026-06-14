# juniper-recurrence

Recurrent / continuous-time neural-network application for the
[Juniper ML research platform](https://github.com/pcalnon/juniper-ml) — the structural
sibling of [juniper-cascor](https://github.com/pcalnon/juniper-cascor).

Where cascor is a stateless, feed-forward, classification-first constructive network,
**juniper-recurrence** adds memory of the past over a real, possibly irregular, time axis,
in service of **time-series regression**.

## Selected model — P3-C (LMU + Approach-C)

The model is a **Legendre Memory Unit whose linear memory cell is discretized in closed form
at the actual per-step time gap Δt** (a matrix exponential of a *fixed* Legendre state matrix —
no ODE solver, no autodiff-through-solver). It is the only first-principles-clean ("C1") option
that is natively irregular-Δt, and the only one with a *measured* irregular-Δt win
(grid-invariance ≈1.15×).

Full design of record (in juniper-ml):
[`notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md).

## Repository layout

This repo follows the Juniper "model family" pattern (precedent:
[`juniper-cascor/juniper-cascor-protocol/`](https://github.com/pcalnon/juniper-cascor/tree/main/juniper-cascor-protocol)):
each independently-publishable, model-specific package lives in a same-named subdirectory.

| Path | What it is | Status |
|------|-----------|--------|
| [`juniper-recurrence-model/`](./juniper-recurrence-model/) | The model-specific core package (`juniper-recurrence-model` on PyPI): the Δt-native LMU memory unit and, later, the recurrent model implementing the shared `juniper-model-core` interfaces. | **scaffolded** — Δt-native LMU unit landed; `TrainableModel` wiring pending `juniper-model-core` |
| `juniper_recurrence/` (app) | The dual-mode FastAPI service (`create_app()`) + CLI (`main.py`), subclassing `juniper-service-core` and injecting the model. | **planned** — gated on `juniper-service-core` / `juniper-model-core` (WS-0/WS-2/WS-3) |

The shared abstractions (`juniper-service-core`, `juniper-model-core`) live as subdirectories of
[juniper-ml](https://github.com/pcalnon/juniper-ml), not here — this repo holds only
recurrence-*specific* code (dependency arrow points specific → common).

## Status

Pre-implementation scaffold. Workstream-0 (design ratification) is not yet ratified; no code is
deployed on this basis. See the design doc for the workstream plan, risks, and open questions.

## License

MIT — see [LICENSE](./LICENSE).
