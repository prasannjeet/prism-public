"""Inferential statistics over the per-conversation records: Wilson 95% CIs,
paired scenario bootstrap, and exact McNemar -- per model, never pooled, over both the strict
and refined scorings. Pure stdlib math; a re-analysis with zero model calls."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bench.metrics import CellRecord

# Two-sided 95% normal quantile (the 0.975 standard-normal inverse CDF). Canonical published
# value; statistics.NormalDist().inv_cdf(0.975) reproduces it to 1 ULP (1.9599639845400536).
Z95 = 1.959963984540054
BOOTSTRAP_SEED = 0
N_RESAMPLES = 10_000


def _success(record: CellRecord, *, refined: bool) -> bool:
    return record.task_success_refined if refined else record.task_success


def _scoring_label(refined: bool) -> str:
    return "refined" if refined else "strict"


@dataclass(frozen=True)
class WilsonRow:
    model: str
    config: str
    scoring: str
    n: int
    successes: int
    rate: float
    ci_low: float
    ci_high: float


def _wilson_bounds(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z = Z95
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # p=0/p=1 pin a bound exactly to the unit edge; float arithmetic lands a hair off, so pin it.
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return (low, high)


def wilson_ci(records: Sequence[CellRecord], *, refined: bool = False) -> list[WilsonRow]:
    """Wilson 95% CI on per-(model, config) success rate, for one scoring."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        g = groups[(r.model, r.config)]
        g[0] += 1
        g[1] += int(_success(r, refined=refined))
    rows: list[WilsonRow] = []
    for (model, config), (n, successes) in sorted(groups.items()):
        low, high = _wilson_bounds(successes, n)
        rows.append(
            WilsonRow(
                model=model,
                config=config,
                scoring=_scoring_label(refined),
                n=n,
                successes=successes,
                rate=successes / n if n else 0.0,
                ci_low=low,
                ci_high=high,
            )
        )
    return rows


@dataclass(frozen=True)
class BootstrapRow:
    model: str
    baseline: str
    scoring: str
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_scenarios: int
    n_resamples: int


def _scenario_rates(
    records: Sequence[CellRecord], *, refined: bool
) -> dict[tuple[str, str], dict[str, float]]:
    """(model, config) -> {scenario -> seed-mean success rate}."""
    sums: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        g = sums[(r.model, r.config, r.scenario)]
        g[0] += 1
        g[1] += int(_success(r, refined=refined))
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (model, config, scenario), (n, c) in sums.items():
        out[(model, config)][scenario] = c / n if n else 0.0
    return out


def bootstrap_deltas(
    records: Sequence[CellRecord],
    *,
    refined: bool = False,
    rng: random.Random | None = None,
) -> list[BootstrapRow]:
    """Paired scenario bootstrap of the C5-minus-baseline success-rate delta, per model.

    Resamples *scenarios* (not runs: runs within a scenario are correlated) with
    replacement; reports 2.5/97.5 percentile bounds (nearest-rank on the sorted resample
    distribution) as the 95% CI and a two-sided percentile p-value. One row per (model,
    baseline) with at least one shared scenario.
    """
    rng = rng if rng is not None else random.Random(BOOTSTRAP_SEED)
    rates = _scenario_rates(records, refined=refined)
    models = sorted({m for (m, _c) in rates})
    rows: list[BootstrapRow] = []
    for model in models:
        c5 = rates.get((model, "C5"), {})
        for baseline in ("C1", "C2", "C3", "C4"):
            base = rates.get((model, baseline), {})
            shared = sorted(set(c5) & set(base))
            if not shared:
                continue
            diffs = [c5[s] - base[s] for s in shared]
            n = len(diffs)
            observed = sum(diffs) / n
            samples: list[float] = []
            for _ in range(N_RESAMPLES):
                picked = rng.choices(diffs, k=n)
                samples.append(sum(picked) / n)
            samples.sort()
            ci_low = samples[int(0.025 * (N_RESAMPLES - 1))]
            ci_high = samples[int(0.975 * (N_RESAMPLES - 1))]
            frac_le = sum(1 for v in samples if v <= 0) / N_RESAMPLES
            frac_ge = sum(1 for v in samples if v >= 0) / N_RESAMPLES
            p_value = min(1.0, 2 * min(frac_le, frac_ge))
            rows.append(
                BootstrapRow(
                    model=model,
                    baseline=baseline,
                    scoring=_scoring_label(refined),
                    delta=observed,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=p_value,
                    n_scenarios=n,
                    n_resamples=N_RESAMPLES,
                )
            )
    return rows


@dataclass(frozen=True)
class McNemarRow:
    model: str
    baseline: str
    scoring: str
    b: int
    c: int
    n_discordant: int
    p_value: float


def _all_seeds_pass(
    records: Sequence[CellRecord], *, refined: bool
) -> dict[tuple[str, str], dict[str, bool]]:
    """(model, config) -> {scenario -> every seed succeeded} (the all-seeds-pass collapse)."""
    agg: dict[tuple[str, str, str], bool] = {}
    for r in records:
        key = (r.model, r.config, r.scenario)
        ok = _success(r, refined=refined)
        agg[key] = ok if key not in agg else (agg[key] and ok)
    out: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for (model, config, scenario), passed in agg.items():
        out[(model, config)][scenario] = passed
    return out


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact-binomial McNemar p: n=b+c trials at 0.5, tail from m=max(b,c)."""
    n = b + c
    if n == 0:
        return 1.0
    m = max(b, c)
    tail = sum(math.comb(n, i) for i in range(m, n + 1)) * 0.5**n
    return min(1.0, 2 * tail)


def mcnemar(records: Sequence[CellRecord], *, refined: bool = False) -> list[McNemarRow]:
    """Exact-binomial McNemar test, C5 vs each baseline, per model, on scenario-level
    pass/fail (all-seeds-pass collapse). Exact rather than chi-square because the discordant
    counts are small. One row per (model, baseline) with a shared scenario.
    """
    passed = _all_seeds_pass(records, refined=refined)
    models = sorted({m for (m, _c) in passed})
    rows: list[McNemarRow] = []
    for model in models:
        c5 = passed.get((model, "C5"), {})
        for baseline in ("C1", "C2", "C3", "C4"):
            base = passed.get((model, baseline), {})
            shared = sorted(set(c5) & set(base))
            if not shared:
                continue
            b = sum(1 for s in shared if c5[s] and not base[s])
            c = sum(1 for s in shared if base[s] and not c5[s])
            rows.append(
                McNemarRow(
                    model=model,
                    baseline=baseline,
                    scoring=_scoring_label(refined),
                    b=b,
                    c=c,
                    n_discordant=b + c,
                    p_value=_exact_mcnemar_p(b, c),
                )
            )
    return rows


def write_stats(records: Sequence[CellRecord], out_dir: Path) -> None:
    """Write wilson_ci.csv, bootstrap_deltas.csv, mcnemar.csv (strict + refined rows stacked).
    A fresh Random(BOOTSTRAP_SEED) per scoring pass keeps output byte-identical on re-run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wilson: list[WilsonRow] = []
    boot: list[BootstrapRow] = []
    mcn: list[McNemarRow] = []
    for refined in (False, True):
        wilson += wilson_ci(records, refined=refined)
        boot += bootstrap_deltas(records, refined=refined, rng=random.Random(BOOTSTRAP_SEED))
        mcn += mcnemar(records, refined=refined)
    pd.DataFrame([r.__dict__ for r in wilson]).to_csv(out_dir / "wilson_ci.csv", index=False)
    pd.DataFrame([r.__dict__ for r in boot]).to_csv(out_dir / "bootstrap_deltas.csv", index=False)
    pd.DataFrame([r.__dict__ for r in mcn]).to_csv(out_dir / "mcnemar.csv", index=False)
