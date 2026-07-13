"""The full-run launcher: runs seed-blocks in order under one global budget, and a billing
pause in any block halts the remaining blocks (the whole run stops cleanly, resumable)."""

from __future__ import annotations

from datetime import date

from bench.engine_cache import build_engine
from bench.full_run import build_run_plans
from bench.oracle import OracleAdapter
from bench.run_full import launch_full_run
from bench.runner import (
    CellKey,
    ModelSpec,
    RunPlan,
    enumerate_cells,
    is_done,
    live_models,
    select_models,
)
from bench.scenario import load_scenarios
from prism.agent import CONFIGS


def _form_scenarios(n: int):
    return tuple(s for s in load_scenarios("scenarios") if s.workflow == "form_booking")[:n]


def _plan(scenarios, seeds: int) -> RunPlan:
    model = ModelSpec("oracle", adapter_factory=lambda: None, allowed_configs=("C1", "C5"))
    return RunPlan("t", (model,), scenarios, ("C1", "C5"), seeds, date(2026, 1, 15), 1000.0)


def _oracle_adapter(cell: CellKey, plan: RunPlan) -> object:
    scenario = next(s for s in plan.scenarios if s.id == cell.scenario)
    return OracleAdapter(build_engine(scenario.workflow), scenario)


def test_launch_runs_all_blocks_when_clear(tmp_path) -> None:
    scenarios = _form_scenarios(2)
    plans = [_plan(scenarios[:1], 1), _plan(scenarios[1:2], 1)]
    summaries = launch_full_run(
        tmp_path,
        plans,
        configs=CONFIGS,
        build_adapter=_oracle_adapter,
        budget_usd=1000.0,
        estimate_usd=0.0,
        require_prices=False,
    )
    assert len(summaries) == 2
    assert all(not s.stopped_on_billing and not s.stopped_on_budget for s in summaries)
    for plan in plans:
        for cell in enumerate_cells(plan):
            assert is_done(tmp_path, cell) is True


class _BillingExc(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__("insufficient_quota: please check your plan and billing details.")


def test_billing_in_first_block_halts_remaining_blocks(tmp_path) -> None:
    scenarios = _form_scenarios(2)
    plans = [_plan(scenarios[:1], 1), _plan(scenarios[1:2], 1)]

    def _broke(cell: CellKey, plan: RunPlan) -> object:
        class _Adapter:
            name = "oracle"
            usages: list = []  # noqa: RUF012

            def complete(self, *a, **k):  # type: ignore[no-untyped-def]
                raise _BillingExc()

        return _Adapter()

    summaries = launch_full_run(
        tmp_path,
        plans,
        configs=CONFIGS,
        build_adapter=_broke,
        budget_usd=1000.0,
        estimate_usd=0.0,
        require_prices=False,
    )
    assert len(summaries) == 1
    assert summaries[0].stopped_on_billing is True
    # The second block never ran: none of its cells exist on disk.
    for cell in enumerate_cells(plans[1]):
        assert not (cell.path(tmp_path) / "meta.json").exists()


def test_gemini_only_continuation_plan_assembly() -> None:
    scenarios = load_scenarios("scenarios")
    lineup = select_models(live_models(), ["gemini-3.1-flash-lite"])
    plans = build_run_plans(
        scenarios, run_id="turnbudget-01", today=date(2026, 1, 15), budget_usd=20.0,
        models=lineup, max_continuation_turns=5,
    )
    assert plans
    assert all(p.max_continuation_turns == 5 for p in plans)
    assert all(tuple(m.name for m in p.models) == ("gemini-3.1-flash-lite",) for p in plans)
