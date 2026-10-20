"""Dataset generation for the recurrence benchmark.

Generates the §3 datasets directly via the ``juniper-data`` generators (no live service),
returning one ordered window set + the per-step Δt so the walk-forward CV does its own splitting.
The synthetic rows use the composed ``full`` view (every window); the equities rows use the
composed ``trainval`` view (train | val, entity-major -- plan W5.2), so the final ``test``
partition is never in a fold, and carry the per-window keys the folds are cut on.

Generator output (verified): ``X_{split} (n,T,F)``, ``y_{split} (n,output_dim)``, ``dt_{split} (n,T)``,
``target_dt_{split} (n,)`` (+ ``observed_mask_{split}``) for ``train`` / ``val`` / ``test``; the composed
views are derived by juniper-recurrence-model (``derive_composed_split``, 0.6.0). For ``equities_seq``
the regression target is ``y_reg_{split}`` (the stationary next-day log-return), not the one-hot
``y_{split}`` direction label, and the artifact also carries ``window_end_date_{split}`` (``YYYYMMDD``
integers) and ``ticker_code_{split}``.

The ``DATASETS`` set is the pre-registered benchmark scope (DP-5 guardrail). Beyond the three
synthetic datasets of the ratified eval it adds four **extensions**: a noise-robustness sweep
(``*_noise0.10`` / ``*_noise0.25`` — does the Δt advantage survive observation noise?), the
``delay_product`` capacity dataset (design §8a), the ``ar_p`` linear floor (plan W-5), and two
real-data irregular-Δt sanity checks (``equities_seq``, one ticker; ``equities_seq_panel``, two
tickers whose folds are cut on the global date per entity -- plan W5.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

__all__ = [
    "Dataset",
    "irregular_sine",
    "multi_sine",
    "mackey_glass",
    "ar_p",
    "delay_product",
    "equities_seq",
    "DATASETS",
    "PRIMARY_DATASETS",
]


@dataclass(frozen=True)
class Dataset:
    """A generated, chronologically-ordered window set for the benchmark."""

    name: str
    grid: str  # "irregular" | "regular"
    X: np.ndarray  # (n, T, F)
    y: np.ndarray  # (n, output_dim)
    dt: np.ndarray  # (n, T) real per-step gaps (dt[:, 0] == 0)
    target_dt: np.ndarray  # (n,) irregular forecast horizon
    # W5.2 (juniper-recurrence-model 0.6.0): the per-window keys a panel's folds are cut on, and the
    # row's own embargo. ``order`` is ``window_end_date`` (``YYYYMMDD`` ints) and ``groups`` is
    # ``ticker_code``; both ``None`` for a synthetic series, whose folds stay row-index folds.
    # ``embargo`` is in windows; ``None`` means the harness default (``run_benchmark._EMBARGO``). The
    # equities rows set it to the look-back -- the gap that keeps an eval window's history clear of
    # every train window's -- as ``POST /v1/crossval`` does by default since juniper-recurrence 0.8.0.
    order: np.ndarray | None = None
    groups: np.ndarray | None = None
    embargo: int | None = None


def _full(out: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Read one array from the whole-dataset view, deriving it on first use.

    Decision 11 (juniper-data#369) retired the ``*_full`` family from the NPZ contract,
    so the generators below emit only ``train`` / ``val`` / ``test`` and this used to be
    a bare ``KeyError`` on every dataset. The ``"full"`` case of :func:`_composed`.
    """
    return _composed(out, key, "full")


def _composed(out: dict[str, np.ndarray], key: str, view: str) -> np.ndarray:
    """Read one array from a composed view -- ``"full"`` (train | val | test) or ``"trainval"`` (train | val) -- deriving the view on first use.

    ``derive_composed_split`` (juniper-recurrence-model 0.6.0) rebuilds either view ENTITY-major,
    stable-sorting on ``ticker_code`` when the artifact carries one. That is load-bearing for
    the equities rows: juniper-data laid its retired ``_full`` down entity-major while the
    partitions are split-major, so a plain ``np.concatenate`` would hold the same rows in a
    DIFFERENT order for any multi-ticker request, and model-core's folds slice by row index
    unless given ``order`` / ``groups`` -- a change to what the benchmark measures, not a
    refactor. A legacy artifact that still ships ``*_full`` keeps the producer's own arrays,
    byte for byte. ``trainval`` (plan W5.2) is the cross-validation pool: the final ``test``
    partition is never in a fold.

    The derived keys are merged back into ``out`` so the reconstruction runs once per
    dataset rather than once per call.
    """
    view_key = f"{key}_{view}"
    if view_key not in out:
        from juniper_recurrence_model.data import derive_composed_split

        out.update(derive_composed_split(out, view))
    if view_key not in out:
        available = ", ".join(sorted(out)) or "<empty>"
        raise KeyError(
            f"{view_key!r} could not be derived; generator emitted: {available}"
        )
    return np.asarray(out[view_key])


def irregular_sine(
    *,
    n_steps: int = 2000,
    lookback: int = 32,
    jitter: float = 0.6,
    noise_std: float = 0.0,
    seed: int = 0,
    name: str = "irregular_sine",
) -> Dataset:
    """Irregular-Δt superimposed sinusoids — the thesis dataset (timing varies window-to-window).

    ``noise_std`` adds Gaussian observation noise (0 = the exact closed-form signal); the
    robustness sweep raises it to check the Δt advantage survives a noisy signal.
    """
    from juniper_data.generators.irregular_sine import (
        IrregularSineGenerator,
        IrregularSineParams,
    )

    out = IrregularSineGenerator.generate(
        IrregularSineParams(
            n_steps=n_steps,
            lookback=lookback,
            jitter=jitter,
            noise_std=noise_std,
            seed=seed,
        )
    )
    return Dataset(
        name,
        "irregular",
        _full(out, "X"),
        _full(out, "y"),
        _full(out, "dt"),
        _full(out, "target_dt"),
    )


def multi_sine(
    *,
    n_steps: int = 2000,
    lookback: int = 32,
    noise_std: float = 0.0,
    seed: int = 0,
    name: str = "multi_sine",
) -> Dataset:
    """Regular-Δt superimposed sinusoids — the control (Δt-awareness must not hurt here)."""
    from juniper_data.generators.multi_sine import MultiSineGenerator, MultiSineParams

    out = MultiSineGenerator.generate(
        MultiSineParams(
            n_steps=n_steps, lookback=lookback, noise_std=noise_std, seed=seed
        )
    )
    return Dataset(
        name,
        "regular",
        _full(out, "X"),
        _full(out, "y"),
        _full(out, "dt"),
        _full(out, "target_dt"),
    )


def mackey_glass(*, n_steps: int = 2000, lookback: int = 32, seed: int = 0) -> Dataset:
    """Regular-Δt Mackey-Glass chaotic series — a harder regular-Δt sanity check."""
    from juniper_data.generators.mackey_glass import (
        MackeyGlassGenerator,
        MackeyGlassParams,
    )

    out = MackeyGlassGenerator.generate(
        MackeyGlassParams(n_steps=n_steps, lookback=lookback, seed=seed)
    )
    return Dataset(
        "mackey_glass",
        "regular",
        _full(out, "X"),
        _full(out, "y"),
        _full(out, "dt"),
        _full(out, "target_dt"),
    )


def ar_p(
    *, n_steps: int = 2000, lookback: int = 32, seed: int = 0, name: str = "ar_p"
) -> Dataset:
    """Regular-Δt stable AR(2) linear-stochastic series — the linear-floor extension (plan W-5).

    A stable AR(p) is exactly representable by a linear readout over a long-enough window, so this
    row probes the floor: Δt-awareness and nonlinear readouts must not LOSE to the linear baseline
    here. Extension row — informational, not scored against the ratified bands.
    """
    from juniper_data.generators.ar_p import ArPGenerator, ArPParams

    out = ArPGenerator.generate(
        ArPParams(n_steps=n_steps, lookback=lookback, seed=seed)
    )
    return Dataset(
        name,
        "regular",
        _full(out, "X"),
        _full(out, "y"),
        _full(out, "dt"),
        _full(out, "target_dt"),
    )


def delay_product(
    *,
    n_steps: int = 2000,
    lookback: int = 32,
    lag1: int = 2,
    lag2: int = 8,
    jitter: float = 0.6,
    noise_std: float = 0.0,
    seed: int = 0,
    name: str = "delay_product",
) -> Dataset:
    """Irregular-Δt signal with a BILINEAR delay-product target ``y = x(t−τ₁)·x(t−τ₂)`` — the DP-3
    capacity dataset (design §8a).

    The two delayed values sit inside the lookback (``lag1`` / ``lag2`` steps back from the window
    end), so the target is a *quadratic form* in the (linear) LMU memory state: a linear readout
    provably cannot fit it (r² ≪ 1) while the nonlinear RFF readout can — a measurable
    nonlinear ≫ linear r² gap, in contrast to the *tie* the two readouts reach on the near-linear
    synthetics. This is the instrument that demonstrates nonlinear-readout *capacity* rather than
    merely confirming the linear ceiling. Requires ``juniper-data>=0.9.0`` (the ``[bench]`` extra
    pins it; ``delay_product`` does not exist before 0.9.0).
    """
    from juniper_data.generators.delay_product import (
        DelayProductGenerator,
        DelayProductParams,
    )

    out = DelayProductGenerator.generate(
        DelayProductParams(
            n_steps=n_steps,
            lookback=lookback,
            lag1=lag1,
            lag2=lag2,
            jitter=jitter,
            noise_std=noise_std,
            seed=seed,
        )
    )
    return Dataset(
        name,
        "irregular",
        _full(out, "X"),
        _full(out, "y"),
        _full(out, "dt"),
        _full(out, "target_dt"),
    )


def equities_seq(
    *,
    symbols: tuple[str, ...] = ("AAPL",),
    start_date: str = "2010-01-01",
    end_date: str = "2022-01-01",
    lookback: int = 32,
    regression_target: str = "log_return",
    name: str = "equities_seq",
) -> Dataset:
    """Real irregular-Δt sequences from equities' daily history (calendar gaps = genuine Δt).

    The default is a **single ticker** (one chronological series); ``equities_seq_panel`` passes
    two. Either way the window set is the composed ``trainval`` pool -- the final ``test``
    partition is never in a fold (plan W5.2) -- and the dataset carries the per-window
    ``window_end_date`` / ``ticker_code`` keys plus ``embargo = lookback``, so ``run_dataset``
    cuts a panel's folds on the global date with a per-entity embargo of one look-back (the
    route's own rule since juniper-recurrence 0.8.0) instead of by row index over an
    entity-major pool. The regression target ``y_reg`` is the
    **stationary next-day log-return** (``regression_target="log_return"``, juniper-data ≥ 0.8.0),
    not the raw next-day close: the raw price level is non-stationary and a closed-form LMU readout
    extrapolates it badly (a r²≈−50 artifact — see the juniper-ml findings doc §3.2 and
    recurrence#28). Requires the ``[equities]`` extra (yfinance / pandas) and network access at
    generation time (Yahoo Finance + SEC EDGAR); a network failure is reported by the runner, not
    fatal.

    The stationary target requires ``juniper-data>=0.8.0`` (which added ``regression_target``; the
    bundled ``sp500_constituents.csv`` ships from 0.7.1 onward). The ``[bench]`` extra pins it — and
    it must, because ``EquitiesParams`` ignores unknown kwargs (pydantic ``extra='ignore'``), so an
    older pin would silently drop this argument and re-measure the raw non-stationary close.
    """
    from juniper_data.generators.equities_seq import (
        EquitiesSeqGenerator,
        EquitiesSeqParams,
    )

    out = EquitiesSeqGenerator.generate(
        EquitiesSeqParams(
            symbols=list(symbols),
            start_date=start_date,
            end_date=end_date,
            lookback=lookback,
            normalize_features=True,
            use_cache=True,
            regression_target=regression_target,
        )
    )
    pool = "trainval"
    has_keys = all(
        f"{key}_{split}" in out
        for key in ("window_end_date", "ticker_code")
        for split in ("train", "val")
    )
    return Dataset(
        name,
        "irregular",
        _composed(out, "X", pool),
        _composed(out, "y_reg", pool),
        _composed(out, "dt", pool),
        _composed(out, "target_dt", pool),
        order=_composed(out, "window_end_date", pool) if has_keys else None,
        groups=_composed(out, "ticker_code", pool) if has_keys else None,
        embargo=int(lookback),
    )


#: The three pre-registered datasets the ratified OQ-14 bands are scored against (DP-5 guardrail).
PRIMARY_DATASETS = ("irregular_sine", "multi_sine", "mackey_glass")

#: name -> generator factory. The primary set plus the extensions (noise sweep + the ar_p linear
#: floor + the delay_product
#: capacity dataset + real data). Noise variants probe whether the Δt advantage survives observation
#: noise; ``delay_product`` is the DP-3 capacity dataset (a bilinear target only a nonlinear readout
#: can fit — design §8a); ``equities_seq`` is the real-data irregular-Δt sanity check. Extension
#: results are reported as informational — not scored against the ratified bands (pre-registered for
#: the primary set).
DATASETS = {
    "irregular_sine": irregular_sine,
    "irregular_sine_noise0.10": partial(
        irregular_sine, noise_std=0.10, name="irregular_sine_noise0.10"
    ),
    "irregular_sine_noise0.25": partial(
        irregular_sine, noise_std=0.25, name="irregular_sine_noise0.25"
    ),
    "multi_sine": multi_sine,
    "multi_sine_noise0.10": partial(
        multi_sine, noise_std=0.10, name="multi_sine_noise0.10"
    ),
    "multi_sine_noise0.25": partial(
        multi_sine, noise_std=0.25, name="multi_sine_noise0.25"
    ),
    "mackey_glass": mackey_glass,
    "ar_p": ar_p,
    "delay_product": delay_product,
    "equities_seq": equities_seq,
    # W5.2: the multi-ticker row. Two entities whose windows interleave in time; the folds are cut on
    # the global date per entity (see run_dataset), which a row-index cut over the entity-major pool
    # would get wrong (ticker A's 2021 in train, ticker B's 2015 in eval). Informational, like the
    # single-ticker row; needs the [bench-equities] extra and network at generation time.
    "equities_seq_panel": partial(
        equities_seq, symbols=("AAPL", "MSFT"), name="equities_seq_panel"
    ),
}
