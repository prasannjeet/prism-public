"""W2 incident_investigation — loads through every guard and matches the design spec."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from prism.engine import StateEngine
from prism.gates import answer_gate, commit_gate, record_evidence
from prism.schema import FieldType, Leaf, Profile
from prism.validators import load_validated_workflow

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"


def _workflow():
    return load_validated_workflow(WORKFLOWS / "incident_investigation.yaml")


TODAY = date(2026, 1, 15)


def _engine() -> StateEngine:
    return StateEngine(_workflow())


def test_incident_investigation_loads_validates_and_is_express() -> None:
    workflow = _workflow()  # raises WorkflowConfigError if any schema/rule guard fails
    assert workflow.profile is Profile.EXPRESS


def test_field_names_in_declaration_order() -> None:
    assert [f.name for f in _workflow().fields] == [
        "affected_service",
        "symptom",
        "outage_time",
        "deploy_check",
        "log_check",
        "server_root_cause",
        "latency_window",
        "metrics_check",
        "latency_root_cause",
        "error_sample",
    ]


def test_node_kinds() -> None:
    kinds = {f.name: f.type for f in _workflow().fields}
    evidence = {n for n, t in kinds.items() if t is FieldType.EVIDENCE}
    conclusion = {n for n, t in kinds.items() if t is FieldType.CONCLUSION}
    assert evidence == {"deploy_check", "log_check", "metrics_check"}
    assert conclusion == {"server_root_cause", "latency_root_cause"}


def test_branch_wiring() -> None:
    wiring = {
        f.name: (f.show_when.field, f.show_when.value)
        for f in _workflow().fields
        if isinstance(f.show_when, Leaf)
    }
    assert wiring == {
        "outage_time": ("symptom", "server_down"),
        "deploy_check": ("symptom", "server_down"),
        "log_check": ("symptom", "server_down"),
        "server_root_cause": ("symptom", "server_down"),
        "latency_window": ("symptom", "slow_response"),
        "metrics_check": ("symptom", "slow_response"),
        "latency_root_cause": ("symptom", "slow_response"),
        "error_sample": ("symptom", "error_spike"),
    }


def test_evidence_satisfied_by() -> None:
    sat = {f.name: f.satisfied_by for f in _workflow().fields if f.satisfied_by}
    assert sat == {
        "deploy_check": ("check_deployments",),
        "log_check": ("check_logs",),
        "metrics_check": ("check_metrics",),
    }


def test_conclusion_requires_evidence() -> None:
    req = {f.name: f.requires_evidence for f in _workflow().fields if f.requires_evidence}
    assert req == {
        "server_root_cause": ("deploy_check", "log_check"),
        "latency_root_cause": ("metrics_check",),
    }


def test_validators_match_design() -> None:
    val = {f.name: dict(f.validators) for f in _workflow().fields if f.validators}
    assert val == {"outage_time": {"not_future": True}}


def test_all_tools_declared_with_descriptions_and_used() -> None:
    workflow = _workflow()
    assert {t.name for t in workflow.tools} == {"check_deployments", "check_logs", "check_metrics"}
    assert all(t.description for t in workflow.tools)  # every tool documents itself
    used = {tool for f in workflow.fields for tool in f.satisfied_by}
    assert used == {t.name for t in workflow.tools}  # no declared-but-unused tools


def test_error_spike_branch_has_no_conclusion() -> None:
    branch = [
        f
        for f in _workflow().fields
        if isinstance(f.show_when, Leaf) and f.show_when.value == "error_spike"
    ]
    assert [f.name for f in branch] == ["error_sample"]
    assert all(f.type is not FieldType.CONCLUSION for f in branch)  # escalate path, no self-resolve


def test_escalation_is_declarative_metadata() -> None:
    esc = _workflow().escalation
    assert esc is not None
    assert (esc.when, esc.action) == ("evidence_insufficient", "escalate_to_engineer")


def test_server_down_branch_activation() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "server_down")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert active == {
        "affected_service",
        "symptom",
        "outage_time",
        "deploy_check",
        "log_check",
        "server_root_cause",
    }


def test_switching_symptom_prunes_collected_evidence_in_one_transition() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "server_down")
    state = engine.with_evidence(state, "deploy_check")
    state = engine.with_evidence(state, "log_check")
    assert state.evidence == frozenset({"deploy_check", "log_check"})
    state = engine.with_answer(state, "symptom", "slow_response")
    assert state.evidence == frozenset()  # evidence on the old branch pruned in one transition


def test_premature_conclusion_blocked_under_enforce_admitted_without() -> None:
    # The C4-vs-C5 signature on a conclusion (an admissible violation).
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "server_down")
    blocked = answer_gate(
        engine, state, "server_root_cause", "bad deploy", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: missing deploy_check, log_check
    admitted = answer_gate(
        engine, state, "server_root_cause", "bad deploy", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: same attempt admitted and logged


def test_evidence_field_cannot_be_answered_directly() -> None:
    # Evidence-faking via submit_answer is an admissible violation (wrong_kind):
    # blocked under enforce, admitted + logged without.
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "server_down")
    blocked = answer_gate(
        engine, state, "deploy_check", "looks fine", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: blocked
    admitted = answer_gate(
        engine, state, "deploy_check", "looks fine", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted and logged as a wrong_kind violation


def test_outage_time_rejects_future_datetime() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "server_down")
    future = answer_gate(
        engine, state, "outage_time", "2026-02-01T09:00", enforce=True, turn=1, today=TODAY
    )
    assert future.accepted is False  # not_future: a future outage time is invalid


def test_full_server_down_investigation_completes_then_commits_express() -> None:
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("affected_service", "checkout-api"),
        ("symptom", "server_down"),
        ("outage_time", "2026-01-10T09:00"),  # before TODAY → passes not_future
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    for name, via in [("deploy_check", "check_deployments"), ("log_check", "check_logs")]:
        outcome = record_evidence(engine, state, name, via, enforce=True, turn=2)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    outcome = answer_gate(
        engine,
        state,
        "server_root_cause",
        "Bad 09:00 deploy crashed the service.",
        enforce=True,
        turn=3,
        today=TODAY,
    )
    assert outcome.accepted, outcome.message  # evidence now present → conclusion allowed
    state = outcome.state
    assert engine.is_complete(state)
    # express profile: commits with no confirm step
    assert commit_gate(engine, state, enforce=True, turn=4).accepted is True


def test_slow_response_single_evidence_conclusion_gates_then_completes() -> None:
    # The single-evidence shape: latency_root_cause is gated on one evidence (metrics_check).
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("affected_service", "checkout-api"),
        ("symptom", "slow_response"),
        ("latency_window", "14:00-14:30 UTC"),
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    blocked = answer_gate(
        engine, state, "latency_root_cause", "DB pool exhaustion", enforce=True, turn=2, today=TODAY
    )
    assert blocked.accepted is False  # blocked: missing metrics_check
    evidence = record_evidence(
        engine, state, "metrics_check", "check_metrics", enforce=True, turn=3
    )
    assert evidence.accepted, evidence.message
    state = evidence.state
    outcome = answer_gate(
        engine, state, "latency_root_cause", "DB pool exhaustion", enforce=True, turn=4, today=TODAY
    )
    assert outcome.accepted, outcome.message  # evidence now present → conclusion allowed
    state = outcome.state
    assert engine.is_complete(state)
    assert commit_gate(engine, state, enforce=True, turn=5).accepted is True  # express commit


def test_error_spike_branch_completes_without_a_conclusion() -> None:
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("affected_service", "checkout-api"),
        ("symptom", "error_spike"),
        ("error_sample", "NullPointerException in CheckoutController.submit"),
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    assert engine.is_complete(state)  # no conclusion node on this branch
    assert commit_gate(engine, state, enforce=True, turn=2).accepted is True
