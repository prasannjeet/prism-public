"""For every authored scenario: the OracleAdapter drives it to its valid completion (proving
reachability), and the evaluator scores that golden log as task_success. Zero API cost."""

from __future__ import annotations

from datetime import date

import pytest

from bench.evaluator import evaluate
from bench.oracle import OracleAdapter
from bench.run import run_scenario
from bench.scenario import Scenario, load_scenarios
from bench.validity import correction_lands_before_completion
from prism.agent import CONFIGS
from prism.engine import StateEngine
from prism.schema import load_workflow

TODAY = date(2026, 1, 15)
SCENARIOS = load_scenarios("scenarios")


def _ids() -> list[str]:
    return [s.id for s in SCENARIOS]


def test_scenarios_exist() -> None:
    # Guard: a glob that silently finds nothing must fail, not vacuously pass.
    assert len(SCENARIOS) >= 40


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids())
def test_oracle_drives_scenario_to_success(scenario: Scenario) -> None:
    engine = StateEngine(load_workflow(f"workflows/{scenario.workflow}.yaml"))
    adapter = OracleAdapter(engine, scenario)
    result = run_scenario(scenario, CONFIGS["C5"], adapter, today=TODAY)
    metrics = evaluate(result.event_log, scenario)
    assert metrics.task_success is True, (
        f"{scenario.id}: terminal={metrics.terminal} "
        f"attempted={metrics.attempted} admitted={metrics.admitted}"
    )
    # The ideal trajectory must be clean: no rejected/admitted events under C5.
    assert metrics.attempted == {}, f"{scenario.id} ideal run had friction: {metrics.attempted}"


_CORRECTION_SCENARIOS = [s for s in SCENARIOS if s.category == "correction_branch_switch"]


@pytest.mark.parametrize(
    "scenario", _CORRECTION_SCENARIOS, ids=[s.id for s in _CORRECTION_SCENARIOS]
)
def test_correction_beats_land_before_completion(scenario: Scenario) -> None:
    # A branch-switch correction must arrive while the workflow is still incomplete, or the
    # agent could complete (and commit) the original branch before the correction lands.
    engine = StateEngine(load_workflow(f"workflows/{scenario.workflow}.yaml"))
    assert correction_lands_before_completion(scenario, engine), (
        f"{scenario.id}: workflow is already complete when the correction arrives"
    )
