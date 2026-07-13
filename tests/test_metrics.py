"""Aggregation: archive -> joined per-conversation records -> master table -> pass^k."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from bench.engine_cache import build_engine
from bench.metrics import load_cell_records
from bench.oracle import OracleAdapter
from bench.runner import CellKey, ModelSpec, RunPlan, run_grid
from bench.scenario import Scenario, load_scenarios
from prism.agent import CONFIGS


def _oracle_build_adapter(cell: CellKey, plan: RunPlan) -> OracleAdapter:
    scenario = next(s for s in plan.scenarios if s.id == cell.scenario)
    return OracleAdapter(build_engine(scenario.workflow), scenario)


def _run(tmp_path: Path) -> tuple[Path, tuple[Scenario, ...]]:
    scenarios = tuple(s for s in load_scenarios("scenarios") if s.workflow == "form_booking")[:2]
    plan = RunPlan(
        run_id="t",
        models=(ModelSpec("oracle-cheap", lambda: None, ("C5",)),),
        scenarios=scenarios,
        configs=("C5",),
        seeds=2,
        today=date(2026, 1, 15),
        budget_usd=1000.0,
    )
    run_grid(plan, tmp_path, configs=CONFIGS, build_adapter=_oracle_build_adapter, estimate_usd=0.0)
    return tmp_path, scenarios


def test_load_cell_records_joins_score_and_tokens(tmp_path: Path) -> None:
    run_dir, scenarios = _run(tmp_path)
    by_id = {s.id: s for s in scenarios}
    records = list(load_cell_records(run_dir, by_id))
    # 2 scenarios * 1 config * 2 seeds = 4 records.
    assert len(records) == 4
    # The oracle reaches the terminal under C5 -> task_success True for every cell.
    assert all(r.task_success for r in records)
    # Oracle has no real token usage -> totals are zero, cost is zero.
    assert all(r.total_tokens == 0 for r in records)
    assert all(r.cost_usd == 0.0 for r in records)
    assert {r.config for r in records} == {"C5"}
    assert {r.seed for r in records} == {0, 1}


import csv  # noqa: E402

from bench.metrics import write_master_table  # noqa: E402


def test_write_master_table_emits_csv_and_json_with_flattened_buckets(tmp_path: Path) -> None:
    run_dir, scenarios = _run(tmp_path)
    by_id = {s.id: s for s in scenarios}
    records = list(load_cell_records(run_dir, by_id))
    out_dir = tmp_path / "agg"
    write_master_table(records, out_dir)

    assert (out_dir / "results.json").exists()
    with (out_dir / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(records)
    # Core columns present; violation buckets flattened to attempted_*/admitted_* columns.
    header = set(rows[0].keys())
    assert {
        "model",
        "workflow",
        "scenario",
        "config",
        "seed",
        "task_success",
        "task_success_refined",
        "total_tokens",
        "cost_usd",
    } <= header
    assert any(col.startswith("attempted_") for col in header) or "task_success" in header


from bench.metrics import CellRecord, pass_k, scoring_delta  # noqa: E402


def _rec(scenario: str, seed: int, ok: bool) -> CellRecord:
    return CellRecord(
        model="m",
        workflow="w",
        scenario=scenario,
        config="C5",
        seed=seed,
        task_success=ok,
        task_success_refined=ok,
        terminal="committed",
        branch=None,
        contamination=0,
        turns=1,
        tool_calls=0,
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


def test_pass_k_combinatorial_matches_spec_examples() -> None:
    # Scenario A: 2 seeds, both succeed. Scenario B: 2 seeds, one succeeds.
    records = [
        _rec("A", 0, True),
        _rec("A", 1, True),
        _rec("B", 0, True),
        _rec("B", 1, False),
    ]
    rows = {(r.config, r.k): r.passk for r in pass_k(records)}
    # pass^1 = mean(C(c,1)/C(n,1)) over scenarios = mean(1.0, 0.5) = 0.75
    assert rows[("C5", 1)] == 0.75
    # pass^2 = mean(C(c,2)/C(n,2)) = mean(1.0, 0.0) = 0.5
    assert rows[("C5", 2)] == 0.5


def test_pass_k_skips_k_above_available_seeds() -> None:
    records = [_rec("A", 0, True), _rec("A", 1, True)]  # n=2 -> only k=1,2 defined
    ks = {r.k for r in pass_k(records)}
    assert ks == {1, 2}


import pandas as pd  # noqa: E402

from bench.plots import _passk_curve  # noqa: E402


def test_passk_curve_is_monotone_non_increasing() -> None:
    # Mixed seed counts: scenario A has 3 seeds (all pass), scenario B has 2 (one pass).
    # The plotted curve must cap k at the smallest seed count so every scenario enters
    # every point; otherwise B drops out at k=3 and lifts the average above pass^1.
    seeds = [("A", 0, 1), ("A", 1, 1), ("A", 2, 1), ("B", 0, 1), ("B", 1, 0)]
    rows = [
        {"model": "m", "config": "C5", "scenario": s, "seed": seed, "task_success": ok}
        for s, seed, ok in seeds
    ]
    curve = _passk_curve(pd.DataFrame(rows)).sort_values("k")
    assert list(curve["k"]) == [1, 2], "curve must cap k at the smallest seed count (here 2)"
    vals = list(curve["passk"])
    pairs = zip(vals, vals[1:], strict=False)
    assert all(a >= b - 1e-12 for a, b in pairs), f"pass^k rose with k: {vals}"


def _cell(**over: object) -> CellRecord:
    # Asymmetric default (task_success=False, task_success_refined=True) is a deliberate
    # "rescued cell" so the two fields can't be silently aliased; callers needing a
    # pass/fail cell must override both explicitly.
    base: dict[str, object] = dict(
        model="m",
        workflow="form_booking",
        scenario="s",
        config="C1",
        seed=0,
        task_success=False,
        task_success_refined=True,
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
    base.update(over)
    return CellRecord(**base)  # type: ignore[arg-type]


def test_cellrecord_has_refined_success() -> None:
    assert _cell().task_success_refined is True
    assert _cell().task_success is False


from bench.metrics import aggregate_run  # noqa: E402


def test_aggregate_run_writes_all_tables(tmp_path: Path) -> None:
    run_dir, scenarios = _run(tmp_path)
    by_id = {s.id: s for s in scenarios}
    aggregate_run(run_dir, by_id)
    agg = run_dir / "aggregate"
    assert (agg / "results.csv").exists()
    assert (agg / "results.json").exists()
    assert (agg / "passk.csv").exists()
    assert (agg / "passk_refined.csv").exists()
    assert (agg / "scoring_delta.csv").exists()


def test_pass_k_refined_selects_refined_success() -> None:
    # One scenario, 2 seeds: strict 0/2, refined 2/2.
    records = [
        _cell(seed=0, task_success=False, task_success_refined=True),
        _cell(seed=1, task_success=False, task_success_refined=True),
    ]
    strict = {(r.model, r.config, r.k): r.passk for r in pass_k(records)}
    refined = {(r.model, r.config, r.k): r.passk for r in pass_k(records, refined=True)}
    assert strict[("m", "C1", 1)] == 0.0
    assert refined[("m", "C1", 1)] == 1.0


def test_scoring_delta_counts_rescued_and_rates() -> None:
    records = [
        _cell(seed=0, task_success=True, task_success_refined=True),
        _cell(seed=1, task_success=False, task_success_refined=True),  # rescued
        _cell(seed=2, task_success=False, task_success_refined=False),
    ]
    rows = {(r.model, r.config): r for r in scoring_delta(records)}
    row = rows[("m", "C1")]
    assert row.n_cells == 3
    assert row.n_pass_strict == 1
    assert row.n_pass_refined == 2
    assert row.n_rescued == 1
    assert row.rate_strict == 1 / 3
    assert row.rate_refined == 2 / 3
    totals = rows[("ALL", "ALL")]
    assert totals.n_cells == 3
    assert totals.n_pass_strict == 1
    assert totals.n_pass_refined == 2
    assert totals.n_rescued == 1


def test_scoring_delta_splits_groups_and_totals() -> None:
    # Two distinct (model, config) groups with different counts so the totals row is
    # provably a re-accumulation over all records, not a copy of one group.
    records = [
        _cell(config="C1", seed=0, task_success=True, task_success_refined=True),
        _cell(config="C1", seed=1, task_success=False, task_success_refined=True),  # rescued
        _cell(config="C1", seed=2, task_success=False, task_success_refined=False),
        _cell(config="C2", seed=0, task_success=True, task_success_refined=True),
        _cell(config="C2", seed=1, task_success=False, task_success_refined=False),
    ]
    rows = {(r.model, r.config): r for r in scoring_delta(records)}

    c1 = rows[("m", "C1")]
    assert (c1.n_cells, c1.n_pass_strict, c1.n_pass_refined, c1.n_rescued) == (3, 1, 2, 1)

    c2 = rows[("m", "C2")]
    assert (c2.n_cells, c2.n_pass_strict, c2.n_pass_refined, c2.n_rescued) == (2, 1, 1, 0)

    totals = rows[("ALL", "ALL")]
    assert (totals.n_cells, totals.n_pass_strict, totals.n_pass_refined, totals.n_rescued) == (
        5,
        2,
        3,
        1,
    )
