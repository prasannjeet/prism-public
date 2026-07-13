"""Run one benchmark scenario end to end with the scripted oracle adapter.

No API keys, no network: the oracle plans the ideal tool-call sequence from the
workflow definition and the scenario's reference answers, the engine executes it,
and the evaluator scores the resulting event log. This demonstrates the full
runtime + harness + scoring path offline.

Usage: python examples/run_oracle.py [scenario_id]
"""

from __future__ import annotations

import argparse
from datetime import date

from bench.evaluator import evaluate
from bench.oracle import OracleAdapter
from bench.run import run_scenario
from bench.scenario import load_scenarios
from prism.agent import CONFIGS
from prism.engine import StateEngine
from prism.schema import load_workflow

TODAY = date(2026, 1, 15)  # the fixed date every benchmark conversation ran under


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario_id",
        nargs="?",
        default="w2-correction-branch-switch",
        help="scenario id (default: w2-correction-branch-switch); see scenarios/*/",
    )
    args = parser.parse_args()

    scenarios = {s.id: s for s in load_scenarios("scenarios")}
    if args.scenario_id not in scenarios:
        raise SystemExit(
            f"unknown scenario {args.scenario_id!r}; known ids: {', '.join(sorted(scenarios))}"
        )
    scenario = scenarios[args.scenario_id]
    engine = StateEngine(load_workflow(f"workflows/{scenario.workflow}.yaml"))
    adapter = OracleAdapter(engine, scenario)
    result = run_scenario(scenario, CONFIGS["C5"], adapter, today=TODAY)
    metrics = evaluate(result.event_log, scenario)

    print(f"scenario:      {scenario.id} (workflow {scenario.workflow})")
    print(f"events logged: {len(result.event_log)}")
    print(f"terminal:      {metrics.terminal}")
    print(f"task_success:  {metrics.task_success}")
    print(f"attempted violations: {metrics.attempted or '{}'}")
    assert metrics.task_success, "oracle run must succeed; the scenario suite guarantees it"
    print("OK: oracle drove the scenario to a successful terminal state.")


if __name__ == "__main__":
    main()
