"""The OracleAdapter plans the ideal tool sequence from a scenario's ground truth + the engine.
One tool call per complete(); empty when the plan is exhausted (so later turns are no-ops)."""

from __future__ import annotations

import pytest

from bench.oracle import OracleAdapter, plan_actions
from bench.scenario import Expected, Scenario
from prism.engine import StateEngine
from prism.models import Message
from prism.schema import Field, FieldType, Profile, Tool, Workflow, load_workflow


def _commit_scenario() -> Scenario:
    return Scenario(
        id="w2-happy",
        workflow="incident_investigation",
        category="happy_path",
        beats=(),
        expected=Expected(
            final_state={
                "affected_service": "checkout",
                "symptom": "server_down",
                "outage_time": "2026-01-15T16:00:00",
                "server_root_cause": "bad deploy at 15:58",
            },
            valid_completion=("committed",),
            required_evidence=("deploy_check", "log_check"),
        ),
    )


def _escalate_scenario() -> Scenario:
    return Scenario(
        id="w2-escalate",
        workflow="incident_investigation",
        category="insufficient_evidence_escalate",
        beats=(),
        expected=Expected(
            final_state={
                "affected_service": "checkout",
                "symptom": "server_down",
                "outage_time": "2026-01-15T16:00:00",
            },
            valid_completion=("escalated",),
            required_evidence=("deploy_check", "log_check"),
        ),
    )


def test_plan_reaches_commit_with_evidence_and_conclusion() -> None:
    engine = StateEngine(load_workflow("workflows/incident_investigation.yaml"))
    calls = plan_actions(engine, _commit_scenario())
    names = [c.name for c in calls]
    assert names[-1] == "commit"
    assert "check_deployments" in names and "check_logs" in names
    assert "confirm" not in names  # express profile
    assert names.index("check_logs") < names.index("commit")


def test_escalate_plan_skips_conclusion_and_calls_escalate() -> None:
    engine = StateEngine(load_workflow("workflows/incident_investigation.yaml"))
    calls = plan_actions(engine, _escalate_scenario())
    names = [c.name for c in calls]
    assert names[-1] == "escalate"
    assert "commit" not in names
    submitted_fields = [c.arguments.get("field") for c in calls if c.name == "submit_answer"]
    assert "server_root_cause" not in submitted_fields


def test_plan_raises_on_no_progress_conclusion_before_evidence() -> None:
    # Conclusion declared BEFORE its required evidence -> planner would otherwise loop forever.
    wf = Workflow(
        name="bad",
        profile=Profile.EXPRESS,
        fields=(
            Field(name="conc", type=FieldType.CONCLUSION, required=True, requires_evidence=("ev",)),
            Field(name="ev", type=FieldType.EVIDENCE, required=True, satisfied_by=("t",)),
        ),
        tools=(Tool(name="t"),),
    )
    engine = StateEngine(wf)
    scenario = Scenario(
        id="bad-1",
        workflow="bad",
        category="happy_path",
        beats=(),
        expected=Expected(final_state={"conc": "x"}, valid_completion=("committed",)),
    )
    with pytest.raises(RuntimeError, match="bad-1"):
        plan_actions(engine, scenario)


def test_adapter_emits_one_call_per_complete_then_empty() -> None:
    engine = StateEngine(load_workflow("workflows/incident_investigation.yaml"))
    adapter = OracleAdapter(engine, _commit_scenario())
    plan_len = len(plan_actions(engine, _commit_scenario()))
    msgs: list[Message] = []
    for _ in range(plan_len):
        resp = adapter.complete("", msgs, [])
        assert len(resp.tool_calls) == 1
    final = adapter.complete("", msgs, [])
    assert final.tool_calls == ()
    assert final.stop_reason == "end_turn"
