"""W3 support_diagnosis — loads through every guard and matches the design spec."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from prism.engine import StateEngine
from prism.gates import answer_gate, commit_gate, confirm, record_evidence
from prism.schema import FieldType, Leaf, Profile, referenced_fields
from prism.validators import load_validated_workflow

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"

TODAY = date(2026, 1, 15)


def _workflow():
    return load_validated_workflow(WORKFLOWS / "support_diagnosis.yaml")


def _engine() -> StateEngine:
    return StateEngine(_workflow())


def test_support_diagnosis_loads_validates_and_is_verified() -> None:
    workflow = _workflow()  # raises WorkflowConfigError if any schema/rule guard fails
    assert workflow.profile is Profile.VERIFIED


def test_field_names_in_declaration_order() -> None:
    assert [f.name for f in _workflow().fields] == [
        "account_id",
        "customer_email",
        "plan_tier",
        "issue_summary",
        "category",
        "invoice_number",
        "billing_lookup",
        "payment_check",
        "billing_resolution",
        "outage_started",
        "service_status",
        "line_test",
        "connectivity_resolution",
        "callback_number",
    ]


def test_optional_field_is_exactly_callback_number() -> None:
    optional = {f.name for f in _workflow().fields if not f.required}
    assert optional == {"callback_number"}


def test_node_kinds() -> None:
    kinds = {f.name: f.type for f in _workflow().fields}
    evidence = {n for n, t in kinds.items() if t is FieldType.EVIDENCE}
    conclusion = {n for n, t in kinds.items() if t is FieldType.CONCLUSION}
    assert evidence == {"billing_lookup", "payment_check", "service_status", "line_test"}
    assert conclusion == {"billing_resolution", "connectivity_resolution"}


def test_branch_wiring_all_gates_on_category() -> None:
    wiring = {
        f.name: (f.show_when.field, f.show_when.value)
        for f in _workflow().fields
        if isinstance(f.show_when, Leaf)
    }
    assert wiring == {
        "invoice_number": ("category", "billing"),
        "billing_lookup": ("category", "billing"),
        "payment_check": ("category", "billing"),
        "billing_resolution": ("category", "billing"),
        "outage_started": ("category", "connectivity"),
        "service_status": ("category", "connectivity"),
        "line_test": ("category", "connectivity"),
        "connectivity_resolution": ("category", "connectivity"),
    }


def test_evidence_satisfied_by() -> None:
    sat = {f.name: f.satisfied_by for f in _workflow().fields if f.satisfied_by}
    assert sat == {
        "billing_lookup": ("check_billing_history",),
        "payment_check": ("check_payment_status",),
        "service_status": ("check_service_status",),
        "line_test": ("run_line_test",),
    }


def test_conclusion_requires_evidence_both_multi() -> None:
    req = {f.name: f.requires_evidence for f in _workflow().fields if f.requires_evidence}
    assert req == {
        "billing_resolution": ("billing_lookup", "payment_check"),
        "connectivity_resolution": ("service_status", "line_test"),
    }


def test_validators_match_design() -> None:
    val = {f.name: dict(f.validators) for f in _workflow().fields if f.validators}
    assert val == {
        "account_id": {"pattern": "ACC-[0-9]{6}"},
        "customer_email": {"pattern": "email"},
        "issue_summary": {"length": {"min": 10, "max": 200}},
        "outage_started": {"not_future": True},
    }


def test_all_tools_declared_with_descriptions_and_used() -> None:
    workflow = _workflow()
    assert {t.name for t in workflow.tools} == {
        "check_billing_history",
        "check_payment_status",
        "check_service_status",
        "run_line_test",
    }
    assert all(t.description for t in workflow.tools)  # every tool documents itself
    used = {tool for f in workflow.fields for tool in f.satisfied_by}
    assert used == {t.name for t in workflow.tools}  # no declared-but-unused tools


def test_plan_tier_is_a_non_driver_select() -> None:
    # plan_tier is a SELECT included for realism — nothing branches on it (a select need not drive).
    workflow = _workflow()
    plan_tier = workflow.field("plan_tier")
    assert plan_tier.type is FieldType.SELECT
    assert plan_tier.options == ("basic", "pro", "enterprise")
    drivers = {
        ref
        for f in workflow.fields
        if f.show_when is not None
        for ref in referenced_fields(f.show_when)
    }
    assert "plan_tier" not in drivers  # the only driver is `category`
    assert drivers == {"category"}


def test_billing_branch_activation() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "category", "billing")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert active == {
        "account_id",
        "customer_email",
        "plan_tier",
        "issue_summary",
        "category",
        "invoice_number",
        "billing_lookup",
        "payment_check",
        "billing_resolution",
        "callback_number",
    }
    # connectivity branch stays closed
    assert "outage_started" not in active
    assert "service_status" not in active
    assert "connectivity_resolution" not in active


def test_switching_category_prunes_heterogeneous_subtree_in_one_transition() -> None:
    # W3's signature: a single category switch prunes BOTH answers and evidence together.
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "category", "billing")
    state = engine.with_answer(state, "invoice_number", "INV-5001")
    state = engine.with_evidence(state, "billing_lookup")
    state = engine.with_evidence(state, "payment_check")
    assert "invoice_number" in state.answers
    assert state.evidence == frozenset({"billing_lookup", "payment_check"})
    state = engine.with_answer(state, "category", "connectivity")
    # one transition: the billing answer AND both billing evidence are gone
    assert "invoice_number" not in state.answers
    assert state.evidence == frozenset()
    assert state.answers["category"] == "connectivity"


def test_account_id_pattern_rejects_malformed() -> None:
    engine = _engine()
    state = engine.initial_state()
    bad = answer_gate(engine, state, "account_id", "ACC-12", enforce=True, turn=1, today=TODAY)
    assert bad.accepted is False  # regex ACC-[0-9]{6} requires 6 digits
    good = answer_gate(engine, state, "account_id", "ACC-123456", enforce=True, turn=1, today=TODAY)
    assert good.accepted is True


def test_issue_summary_length_rejects_too_short() -> None:
    engine = _engine()
    state = engine.initial_state()
    short = answer_gate(
        engine, state, "issue_summary", "too short", enforce=True, turn=1, today=TODAY
    )
    assert short.accepted is False  # 9 chars < min 10
    ok = answer_gate(
        engine,
        state,
        "issue_summary",
        "Double charged on my last invoice.",
        enforce=True,
        turn=1,
        today=TODAY,
    )
    assert ok.accepted is True


def test_outage_started_rejects_future_date() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "category", "connectivity")
    future = answer_gate(
        engine, state, "outage_started", "2026-02-01", enforce=True, turn=1, today=TODAY
    )
    assert future.accepted is False  # not_future: a future outage date is invalid
    past = answer_gate(
        engine, state, "outage_started", "2026-01-10", enforce=True, turn=1, today=TODAY
    )
    assert past.accepted is True


def test_inactive_field_answer_blocked_under_enforce_admitted_without() -> None:
    # Wrong-branch temptation: answer a billing field while on the connectivity branch.
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "category", "connectivity")
    blocked = answer_gate(
        engine, state, "invoice_number", "INV-1", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: not on the active path
    admitted = answer_gate(
        engine, state, "invoice_number", "INV-1", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted and logged (engine prunes the inactive answer)


def test_evidence_field_cannot_be_answered_directly() -> None:
    # Evidence-faking via submit_answer is an admissible violation (wrong_kind):
    # blocked under enforce, admitted + logged without.
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "category", "billing")
    blocked = answer_gate(
        engine, state, "billing_lookup", "looks fine", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: blocked
    admitted = answer_gate(
        engine, state, "billing_lookup", "looks fine", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted and logged as a wrong_kind violation


def test_premature_conclusion_blocked_under_enforce_admitted_without() -> None:
    # The conclusion-requires-evidence boundary on W3's billing branch (needs both evidence).
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "category", "billing")
    blocked = answer_gate(
        engine, state, "billing_resolution", "refund issued", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: missing billing_lookup, payment_check
    admitted = answer_gate(
        engine, state, "billing_resolution", "refund issued", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: same attempt admitted and logged


def test_full_billing_diagnosis_completes_then_commits_after_confirm() -> None:
    # The verified end-to-end: form -> branch -> answer -> evidence -> conclusion
    # -> confirm -> commit.
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("account_id", "ACC-123456"),
        ("customer_email", "sam@example.com"),
        ("plan_tier", "pro"),
        ("issue_summary", "I was double charged on my latest invoice."),
        ("category", "billing"),
        ("invoice_number", "INV-5001"),
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    for name, via in [
        ("billing_lookup", "check_billing_history"),
        ("payment_check", "check_payment_status"),
    ]:
        outcome = record_evidence(engine, state, name, via, enforce=True, turn=2)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    outcome = answer_gate(
        engine,
        state,
        "billing_resolution",
        "Duplicate charge confirmed; refund issued.",
        enforce=True,
        turn=3,
        today=TODAY,
    )
    assert outcome.accepted, outcome.message  # evidence now present -> conclusion allowed
    state = outcome.state
    assert engine.is_complete(state)  # callback_number is optional -> not required to complete
    # verified profile: commit blocked until an explicit confirm
    assert commit_gate(engine, state, enforce=True, turn=4).accepted is False
    state = confirm(engine, state, turn=4).state
    assert commit_gate(engine, state, enforce=True, turn=5).accepted is True


def test_confirmation_resets_when_data_changes_after_confirm() -> None:
    # Verified integrity: a mutation after confirm invalidates the confirmation (re-confirm needed).
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("account_id", "ACC-123456"),
        ("customer_email", "sam@example.com"),
        ("plan_tier", "pro"),
        ("issue_summary", "Connectivity has been down since this morning."),
        ("category", "connectivity"),
        ("outage_started", "2026-01-10"),
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    for name, via in [
        ("service_status", "check_service_status"),
        ("line_test", "run_line_test"),
    ]:
        outcome = record_evidence(engine, state, name, via, enforce=True, turn=2)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    outcome = answer_gate(
        engine,
        state,
        "connectivity_resolution",
        "Regional outage confirmed; line test failed upstream.",
        enforce=True,
        turn=3,
        today=TODAY,
    )
    assert outcome.accepted, outcome.message
    state = outcome.state
    assert engine.is_complete(state)
    state = confirm(engine, state, turn=4).state
    assert state.confirmed is True
    # change the optional callback number after confirming -> confirmation resets
    changed = answer_gate(
        engine, state, "callback_number", "+1-555-0100", enforce=True, turn=5, today=TODAY
    )
    assert changed.accepted is True
    state = changed.state
    assert state.confirmed is False  # confirmation invalidated by the later mutation
    assert commit_gate(engine, state, enforce=True, turn=6).accepted is False  # must re-confirm
