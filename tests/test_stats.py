from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.metrics import CellRecord
from bench.stats import (
    BootstrapRow,
    McNemarRow,
    WilsonRow,
    bootstrap_deltas,
    mcnemar,
    wilson_ci,
    write_stats,
)


def _rec(
    model: str,
    config: str,
    scenario: str,
    seed: int,
    *,
    strict: bool,
    refined: bool | None = None,
) -> CellRecord:
    """Minimal CellRecord for stats tests; only the fields stats.py reads carry meaning."""
    return CellRecord(
        model=model,
        workflow="form_booking",
        scenario=scenario,
        config=config,
        seed=seed,
        task_success=strict,
        task_success_refined=strict if refined is None else refined,
        terminal="committed",
        branch=None,
        contamination=0,
        turns=1,
        tool_calls=1,
        tool_errors=0,
        recovered=False,
        attempted={},
        admitted={},
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
    )


def test_wilson_known_reference() -> None:
    # 8 successes / 10 -> Wilson 95% interval ~ [0.490, 0.943].
    records = [_rec("m", "C5", "s", i, strict=i < 8) for i in range(10)]
    rows = [r for r in wilson_ci(records) if r.config == "C5"]
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, WilsonRow)
    assert row.scoring == "strict"
    assert row.n == 10
    assert row.successes == 8
    assert row.rate == 0.8
    assert abs(row.ci_low - 0.490) < 1e-3
    assert abs(row.ci_high - 0.943) < 1e-3


def test_wilson_edges_stay_in_unit_interval() -> None:
    all_fail = [_rec("m", "C1", "s", i, strict=False) for i in range(10)]
    all_pass = [_rec("m", "C5", "s", i, strict=True) for i in range(10)]
    low = next(r for r in wilson_ci(all_fail) if r.config == "C1")
    high = next(r for r in wilson_ci(all_pass) if r.config == "C5")
    assert low.rate == 0.0
    assert low.ci_low == 0.0
    assert high.ci_high == 1.0
    assert low.ci_high > 0.0  # interval is non-degenerate even at p=0
    assert high.ci_low < 1.0


def test_wilson_emits_both_scorings() -> None:
    records = [
        _rec("m", "C5", "s", 0, strict=False, refined=True),
        _rec("m", "C5", "s", 1, strict=True, refined=True),
    ]
    by_scoring = {r.scoring: r for r in wilson_ci(records, refined=False)} | {
        r.scoring: r for r in wilson_ci(records, refined=True)
    }
    assert by_scoring["strict"].rate == 0.5
    assert by_scoring["refined"].rate == 1.0


def test_bootstrap_separable_case() -> None:
    # C5 every seed passes, C1 every seed fails, across 3 scenarios -> delta = 1.0.
    records = []
    for s in ("a", "b", "c"):
        for seed in range(2):
            records.append(_rec("m", "C5", s, seed, strict=True))
            records.append(_rec("m", "C1", s, seed, strict=False))
    rows = [r for r in bootstrap_deltas(records) if r.baseline == "C1"]
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, BootstrapRow)
    assert row.delta == 1.0
    assert row.ci_low == 1.0
    assert row.ci_high == 1.0
    assert row.n_scenarios == 3
    assert row.p_value < 0.05


def test_bootstrap_null_case() -> None:
    # C5 and C2 identical per scenario -> delta 0, CI brackets 0, p ~ 1.
    records = []
    for s in ("a", "b", "c"):
        for seed in range(2):
            records.append(_rec("m", "C5", s, seed, strict=(s != "a")))
            records.append(_rec("m", "C2", s, seed, strict=(s != "a")))
    row = next(r for r in bootstrap_deltas(records) if r.baseline == "C2")
    assert row.delta == 0.0
    assert row.ci_low <= 0.0 <= row.ci_high
    assert row.p_value == 1.0


def test_bootstrap_is_deterministic() -> None:
    records = []
    for s in ("a", "b", "c", "d"):
        for seed in range(2):
            records.append(_rec("m", "C5", s, seed, strict=(s in ("a", "b", "c"))))
            records.append(_rec("m", "C4", s, seed, strict=(s == "a")))
    first = bootstrap_deltas(records)
    second = bootstrap_deltas(records)
    assert [r.__dict__ for r in first] == [r.__dict__ for r in second]


def test_bootstrap_one_row_per_baseline() -> None:
    records = []
    for cfg in ("C5", "C1", "C2", "C3", "C4"):
        for s in ("a", "b"):
            for seed in range(2):
                records.append(_rec("m", cfg, s, seed, strict=(cfg == "C5")))
    baselines = {r.baseline for r in bootstrap_deltas(records)}
    assert baselines == {"C1", "C2", "C3", "C4"}


def test_mcnemar_all_seeds_pass_collapse() -> None:
    # C5 passes scenario a in BOTH seeds; C1 fails both -> b=1 discordant for a.
    # Scenario b: C5 passes one seed only -> NOT all-seeds-pass -> C5 fail.
    records = [
        _rec("m", "C5", "a", 0, strict=True), _rec("m", "C5", "a", 1, strict=True),
        _rec("m", "C1", "a", 0, strict=False), _rec("m", "C1", "a", 1, strict=False),
        _rec("m", "C5", "b", 0, strict=True), _rec("m", "C5", "b", 1, strict=False),
        _rec("m", "C1", "b", 0, strict=False), _rec("m", "C1", "b", 1, strict=False),
    ]
    row = next(r for r in mcnemar(records) if r.baseline == "C1")
    assert isinstance(row, McNemarRow)
    assert row.b == 1   # a: C5 pass, C1 fail
    assert row.c == 0
    assert row.n_discordant == 1
    assert abs(row.p_value - 2 * 0.5 ** 1) < 1e-9   # = 1.0 exactly (single discordant pair)


def test_mcnemar_exact_b5_c0() -> None:
    records = []
    for i in range(5):
        s = f"s{i}"
        records.append(_rec("m", "C5", s, 0, strict=True))
        records.append(_rec("m", "C2", s, 0, strict=False))
    row = next(r for r in mcnemar(records) if r.baseline == "C2")
    assert row.b == 5
    assert row.c == 0
    assert abs(row.p_value - 2 * 0.5 ** 5) < 1e-9   # 0.0625


def test_mcnemar_symmetric_and_empty() -> None:
    # b == c == 3 -> p = 1.0 ; and no discordant pairs -> p = 1.0.
    sym = []
    for i in range(3):
        sym += [_rec("m", "C5", f"p{i}", 0, strict=True), _rec("m", "C3", f"p{i}", 0, strict=False)]
        sym += [_rec("m", "C5", f"q{i}", 0, strict=False), _rec("m", "C3", f"q{i}", 0, strict=True)]
    row = next(r for r in mcnemar(sym) if r.baseline == "C3")
    assert row.b == 3
    assert row.c == 3
    assert row.p_value == 1.0

    concordant = [
        _rec("m", "C5", "x", 0, strict=True), _rec("m", "C4", "x", 0, strict=True),
    ]
    row2 = next(r for r in mcnemar(concordant) if r.baseline == "C4")
    assert row2.n_discordant == 0
    assert row2.p_value == 1.0


def test_write_stats_emits_three_csvs_both_scorings(tmp_path: Path) -> None:
    records = []
    for cfg in ("C5", "C1", "C2", "C3", "C4"):
        for s in ("a", "b"):
            for seed in range(2):
                strict = cfg == "C5"
                refined = strict or (cfg == "C2" and s == "a")
                records.append(_rec("m", cfg, s, seed, strict=strict, refined=refined))
    write_stats(records, tmp_path)
    for name in ("wilson_ci.csv", "bootstrap_deltas.csv", "mcnemar.csv"):
        frame = pd.read_csv(tmp_path / name)
        assert set(frame["scoring"]) == {"strict", "refined"}
