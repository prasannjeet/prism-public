"""Read-only progress snapshot: counts on-disk markers against the expected grid + sums spend."""

from __future__ import annotations

from datetime import date

from bench.prices import ModelPrice
from bench.runner import CellKey, ModelSpec, RunPlan, archive_cell
from bench.scenario import load_scenarios
from bench.status import summarize


def _one_scenario():
    return next(s for s in load_scenarios("scenarios") if s.workflow == "form_booking")


def _tiny_plan(scenario) -> RunPlan:
    model = ModelSpec("m", adapter_factory=lambda: None, allowed_configs=("C1", "C5"))
    return RunPlan("t", (model,), (scenario,), ("C1", "C5"), 2, date(2026, 1, 15), 300.0)


def test_summarize_counts_states_spend_and_breakdowns(tmp_path) -> None:
    scenario = _one_scenario()
    plan = _tiny_plan(scenario)
    prices = {"m": ModelPrice(1.0, 5.0)}
    # 4 expected cells: configs C1/C5 x seeds 0/1.
    cells = {
        (c, s): CellKey("m", scenario.workflow, scenario.id, c, s)
        for c in ("C1", "C5")
        for s in (0, 1)
    }
    totals = {"input_tokens": 1000, "output_tokens": 200}  # -> $0.002 per completed cell

    def _archive(cell: CellKey, status: str, with_tokens: bool) -> None:
        sidecar = {"cell": cell.as_dict(), "totals": totals if with_tokens else {}}
        archive_cell(tmp_path, cell, (), sidecar, {"cell": cell.as_dict(), "status": status})

    _archive(cells[("C1", 0)], "completed", True)
    _archive(cells[("C1", 1)], "completed", True)
    _archive(cells[("C5", 0)], "failed_infra", False)
    # cells[("C5", 1)] is left with no marker -> pending.

    report = summarize(tmp_path, [plan], prices=prices)
    assert report.total == 4
    assert report.completed == 2
    assert report.failed == 1
    assert report.pending == 1
    assert report.percent == 50.0
    assert abs(report.spent_usd - 0.004) < 1e-9
    # Breakdown is (done, total) per dimension value.
    assert report.by_config == {"C1": (2, 2), "C5": (0, 2)}
    assert report.by_model == {"m": (2, 4)}
    assert report.by_workflow == {"form_booking": (2, 4)}
