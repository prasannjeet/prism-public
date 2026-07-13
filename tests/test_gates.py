"""Gate tests — both enforce branches for every check; exact event tuples."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from io import StringIO

from prism.engine import State, StateEngine
from prism.events import Event, EventType, JsonlEventSink
from prism.gates import GateOutcome, answer_gate, commit_gate, confirm, record_evidence
from prism.schema import parse_workflow

_TODAY = date(2026, 1, 15)


def _engine(profile: str = "express") -> StateEngine:
    data = {
        "workflow": "incident",
        "profile": profile,
        "fields": [
            {
                "name": "symptom",
                "type": "select",
                "required": True,
                "options": ["down", "slow"],
                "prompt": "What did you observe?",
            },
            {
                "name": "outage_time",
                "type": "datetime",
                "required": True,
                "show_when": "symptom = down",
                "validate": {"not_future": True},
            },
            {
                "name": "deploy_check",
                "type": "evidence",
                "required": True,
                "show_when": "symptom = down",
                "satisfied_by": ["check_deployments"],
            },
            {
                "name": "verdict",
                "type": "conclusion",
                "required": True,
                "show_when": "symptom = down",
                "requires_evidence": ["deploy_check"],
            },
            {"name": "latency", "type": "text", "required": True, "show_when": "symptom = slow"},
        ],
        "tools": [{"name": "check_deployments"}],
    }
    return StateEngine(parse_workflow(data))


def _down_state(engine: StateEngine) -> State:
    return engine.with_answer(engine.initial_state(), "symptom", "down")


# --- answer gate: accept path ---------------------------------------------------


def test_accepted_answer_emits_submission_and_activations() -> None:
    engine = _engine()
    outcome = answer_gate(
        engine, engine.initial_state(), "symptom", "down", enforce=True, turn=1, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.message is None
    assert dict(outcome.state.answers) == {"symptom": "down"}
    assert outcome.events == (
        Event(EventType.ANSWER_SUBMITTED, 1, {"field": "symptom", "value": "down"}),
        Event(EventType.BRANCH_ACTIVATED, 1, {"field": "outage_time"}),
        Event(EventType.BRANCH_ACTIVATED, 1, {"field": "deploy_check"}),
        Event(EventType.BRANCH_ACTIVATED, 1, {"field": "verdict"}),
    )


def test_branch_switch_emits_activation_and_cleanup_events() -> None:
    engine = _engine()
    state = _down_state(engine)
    state = engine.with_answer(state, "outage_time", "2026-01-10 16:00")
    state = engine.with_evidence(state, "deploy_check")
    outcome = answer_gate(engine, state, "symptom", "slow", enforce=True, turn=5, today=_TODAY)
    assert outcome.accepted is True
    assert dict(outcome.state.answers) == {"symptom": "slow"}
    assert outcome.state.evidence == frozenset()
    assert outcome.events == (
        Event(EventType.ANSWER_SUBMITTED, 5, {"field": "symptom", "value": "slow"}),
        Event(EventType.BRANCH_ACTIVATED, 5, {"field": "latency"}),
        Event(EventType.BRANCH_CLEANED, 5, {"field": "outage_time"}),
        Event(EventType.BRANCH_CLEANED, 5, {"field": "deploy_check"}),
    )


# --- answer gate: structural rejection (both modes) ------------------------------


def test_unknown_field_rejected_in_both_modes() -> None:
    engine = _engine()
    for enforce in (True, False):
        outcome = answer_gate(
            engine, engine.initial_state(), "ghost", "x", enforce=enforce, turn=1, today=_TODAY
        )
        assert outcome.accepted is False
        assert outcome.state == engine.initial_state()
        assert outcome.message == "Unknown field 'ghost'."
        assert outcome.events == (
            Event(
                EventType.ANSWER_REJECTED,
                1,
                {"field": "ghost", "value": "x", "reason": "unknown_field", "code": None},
            ),
        )


# --- answer/evidence gate: post-commit rejection (both modes) --------------------


def test_answer_after_commit_rejected_both_modes() -> None:
    engine = _engine()
    base = replace(engine.initial_state(), submitted=True)
    for enforce in (True, False):
        out = answer_gate(engine, base, "symptom", "down", enforce=enforce, turn=1, today=_TODAY)
        assert out.accepted is False
        assert out.state == base
        assert out.events[0].type == EventType.ANSWER_REJECTED
        assert out.events[0].payload["reason"] == "already_submitted"


def test_evidence_after_commit_rejected_both_modes() -> None:
    engine = _engine()
    base = replace(engine.initial_state(), submitted=True)
    for enforce in (True, False):
        out = record_evidence(
            engine, base, "deploy_check", "check_deployments", enforce=enforce, turn=1
        )
        assert out.accepted is False
        assert out.state == base
        assert out.events[0].type == EventType.EVIDENCE_REJECTED
        assert out.events[0].payload["reason"] == "already_submitted"


# --- answer gate: admissible violations (enforce-sensitive) ----------------------


def test_answer_to_evidence_field_rejected_when_enforced() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(engine, state, "deploy_check", "done", enforce=True, turn=2, today=_TODAY)
    assert outcome.accepted is False
    assert outcome.state == state
    assert outcome.message == (
        "Field 'deploy_check' is evidence; it is collected via tools, not answers."
    )
    assert outcome.events == (
        Event(
            EventType.ANSWER_REJECTED,
            2,
            {"field": "deploy_check", "value": "done", "reason": "wrong_kind", "code": None},
        ),
    )


def test_answer_to_evidence_field_admitted_when_unenforced() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(
        engine, state, "deploy_check", "done", enforce=False, turn=2, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.message is None
    assert outcome.state.answers["deploy_check"] == "done"
    # The faked answer never satisfies the evidence requirement.
    assert engine.is_satisfied(engine.workflow.field("deploy_check"), outcome.state) is False
    assert outcome.events == (
        Event(
            EventType.VIOLATION_ADMITTED,
            2,
            {
                "gate": "answer",
                "field": "deploy_check",
                "value": "done",
                "reason": "wrong_kind",
                "code": None,
            },
        ),
    )


def test_inactive_field_answer_rejected_when_enforced() -> None:
    engine = _engine()
    outcome = answer_gate(
        engine, engine.initial_state(), "latency", "200ms", enforce=True, turn=1, today=_TODAY
    )
    assert outcome.accepted is False
    assert outcome.state == engine.initial_state()
    assert outcome.message == "Field 'latency' is not on the active path."
    assert outcome.events == (
        Event(
            EventType.ANSWER_REJECTED,
            1,
            {"field": "latency", "value": "200ms", "reason": "inactive_field", "code": None},
        ),
    )


def test_inactive_field_answer_admitted_then_pruned_when_unenforced() -> None:
    engine = _engine()
    outcome = answer_gate(
        engine, engine.initial_state(), "latency", "200ms", enforce=False, turn=1, today=_TODAY
    )
    assert outcome.accepted is True
    # The engine prunes the inactive answer immediately; the event is the only record.
    assert "latency" not in outcome.state.answers
    assert outcome.events == (
        Event(
            EventType.VIOLATION_ADMITTED,
            1,
            {
                "gate": "answer",
                "field": "latency",
                "value": "200ms",
                "reason": "inactive_field",
                "code": None,
            },
        ),
    )


def test_conclusion_before_evidence_rejected_when_enforced() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(
        engine, state, "verdict", "bad deploy", enforce=True, turn=2, today=_TODAY
    )
    assert outcome.accepted is False
    assert outcome.state == state
    assert outcome.message == "Cannot answer 'verdict' yet: missing evidence deploy_check."
    assert outcome.events == (
        Event(
            EventType.ANSWER_REJECTED,
            2,
            {
                "field": "verdict",
                "value": "bad deploy",
                "reason": "evidence_missing",
                "code": None,
            },
        ),
    )


def test_conclusion_before_evidence_admitted_when_unenforced() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(
        engine, state, "verdict", "bad deploy", enforce=False, turn=2, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.state.answers["verdict"] == "bad deploy"
    assert outcome.events == (
        Event(
            EventType.VIOLATION_ADMITTED,
            2,
            {
                "gate": "answer",
                "field": "verdict",
                "value": "bad deploy",
                "reason": "evidence_missing",
                "code": None,
            },
        ),
    )


def test_conclusion_after_evidence_accepted() -> None:
    engine = _engine()
    state = engine.with_evidence(_down_state(engine), "deploy_check")
    outcome = answer_gate(
        engine, state, "verdict", "bad deploy", enforce=True, turn=3, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.events == (
        Event(EventType.ANSWER_SUBMITTED, 3, {"field": "verdict", "value": "bad deploy"}),
    )


def test_invalid_value_rejected_when_enforced() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(
        engine, state, "outage_time", "not-a-date", enforce=True, turn=2, today=_TODAY
    )
    assert outcome.accepted is False
    assert outcome.state == state
    assert outcome.message == "Must be a date and time in ISO format."
    assert outcome.events == (
        Event(
            EventType.ANSWER_REJECTED,
            2,
            {
                "field": "outage_time",
                "value": "not-a-date",
                "reason": "invalid_value",
                "code": "validation.type",
            },
        ),
    )


def test_invalid_value_admitted_when_unenforced() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(
        engine, state, "outage_time", "not-a-date", enforce=False, turn=2, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.state.answers["outage_time"] == "not-a-date"
    assert outcome.events == (
        Event(
            EventType.VIOLATION_ADMITTED,
            2,
            {
                "gate": "answer",
                "field": "outage_time",
                "value": "not-a-date",
                "reason": "invalid_value",
                "code": "validation.type",
            },
        ),
    )


def test_future_datetime_rejected_by_not_future_rule() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(
        engine, state, "outage_time", "2026-02-01 16:00", enforce=True, turn=2, today=_TODAY
    )
    assert outcome.accepted is False
    assert outcome.message == "Date cannot be in the future."
    assert outcome.events[0].payload["code"] == "validation.not_future"


# --- answer gate: canonical storage of DATE/DATETIME -----------------------------


def test_datetime_stored_canonically_with_t_separator() -> None:
    # Pilot finding: a model submitted a space-separated datetime; the committed value
    # must be the canonical ISO form so the outcome scorer's structured exact-match holds.
    engine = _engine()
    state = _down_state(engine)
    outcome = answer_gate(
        engine, state, "outage_time", "2026-01-15 16:00:00", enforce=True, turn=2, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.state.answers["outage_time"] == "2026-01-15T16:00:00"


def test_date_stored_canonically_stays_isoformat() -> None:
    data = {
        "workflow": "form",
        "profile": "express",
        "fields": [{"name": "due", "type": "date", "required": True, "prompt": "Due date?"}],
    }
    engine = StateEngine(parse_workflow(data))
    outcome = answer_gate(
        engine, engine.initial_state(), "due", "2026-01-28", enforce=True, turn=1, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.state.answers["due"] == "2026-01-28"  # already canonical


def test_text_value_stored_unchanged() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "slow")
    outcome = answer_gate(engine, state, "latency", "200 ms", enforce=True, turn=2, today=_TODAY)
    assert outcome.accepted is True
    assert outcome.state.answers["latency"] == "200 ms"  # TEXT: no canonicalization


def test_select_value_stored_unchanged() -> None:
    engine = _engine()
    outcome = answer_gate(
        engine, engine.initial_state(), "symptom", "down", enforce=True, turn=1, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.state.answers["symptom"] == "down"  # SELECT: no canonicalization


# --- answer gate: confirmation reset ---------------------------------------------


def test_accepted_answer_resets_stale_confirmation() -> None:
    engine = _engine("verified")
    state = engine.with_confirmation(_down_state(engine))
    outcome = answer_gate(
        engine, state, "outage_time", "2026-01-10 16:00", enforce=True, turn=3, today=_TODAY
    )
    assert outcome.accepted is True
    assert outcome.state.confirmed is False
    assert outcome.events == (
        Event(EventType.ANSWER_SUBMITTED, 3, {"field": "outage_time", "value": "2026-01-10 16:00"}),
        Event(EventType.CONFIRMATION_RESET, 3, {}),
    )


def test_rejected_answer_keeps_confirmation() -> None:
    engine = _engine("verified")
    state = engine.with_confirmation(_down_state(engine))
    outcome = answer_gate(
        engine, state, "outage_time", "not-a-date", enforce=True, turn=3, today=_TODAY
    )
    assert outcome.accepted is False
    assert outcome.state.confirmed is True


# --- evidence gate ----------------------------------------------------------------


def test_evidence_collected_on_active_field() -> None:
    engine = _engine()
    state = _down_state(engine)
    outcome = record_evidence(
        engine, state, "deploy_check", "check_deployments", enforce=True, turn=3
    )
    assert outcome.accepted is True
    assert outcome.message is None
    assert "deploy_check" in outcome.state.evidence
    assert outcome.events == (
        Event(
            EventType.EVIDENCE_COLLECTED,
            3,
            {"field": "deploy_check", "via": "check_deployments"},
        ),
    )


def test_evidence_unknown_field_rejected_in_both_modes() -> None:
    engine = _engine()
    for enforce in (True, False):
        outcome = record_evidence(
            engine, _down_state(engine), "ghost", "check_deployments", enforce=enforce, turn=3
        )
        assert outcome.accepted is False
        assert outcome.message == "Unknown field 'ghost'."
        assert outcome.events == (
            Event(
                EventType.EVIDENCE_REJECTED,
                3,
                {"field": "ghost", "via": "check_deployments", "reason": "unknown_field"},
            ),
        )


def test_evidence_on_non_evidence_field_rejected_in_both_modes() -> None:
    # Structural: the engine raises FieldKindError — there is nothing to admit even in C4.
    engine = _engine()
    for enforce in (True, False):
        outcome = record_evidence(
            engine, _down_state(engine), "symptom", "check_deployments", enforce=enforce, turn=3
        )
        assert outcome.accepted is False
        assert outcome.message == "Field 'symptom' is not an evidence field."
        assert outcome.events == (
            Event(
                EventType.EVIDENCE_REJECTED,
                3,
                {"field": "symptom", "via": "check_deployments", "reason": "wrong_kind"},
            ),
        )


def test_evidence_via_undeclared_tool_rejected_in_both_modes() -> None:
    engine = _engine()
    for enforce in (True, False):
        outcome = record_evidence(
            engine, _down_state(engine), "deploy_check", "check_metrics", enforce=enforce, turn=3
        )
        assert outcome.accepted is False
        assert outcome.message == (
            "Tool 'check_metrics' does not satisfy evidence field 'deploy_check'."
        )
        assert outcome.events == (
            Event(
                EventType.EVIDENCE_REJECTED,
                3,
                {"field": "deploy_check", "via": "check_metrics", "reason": "wrong_tool"},
            ),
        )


def test_evidence_on_inactive_field_rejected_when_enforced() -> None:
    engine = _engine()
    outcome = record_evidence(
        engine, engine.initial_state(), "deploy_check", "check_deployments", enforce=True, turn=1
    )
    assert outcome.accepted is False
    assert outcome.message == "Field 'deploy_check' is not on the active path."
    assert outcome.events == (
        Event(
            EventType.EVIDENCE_REJECTED,
            1,
            {"field": "deploy_check", "via": "check_deployments", "reason": "inactive_field"},
        ),
    )


def test_evidence_on_inactive_field_admitted_then_pruned_when_unenforced() -> None:
    engine = _engine()
    outcome = record_evidence(
        engine, engine.initial_state(), "deploy_check", "check_deployments", enforce=False, turn=1
    )
    assert outcome.accepted is True
    assert outcome.state.evidence == frozenset()  # pruned; the event is the record
    assert outcome.events == (
        Event(
            EventType.VIOLATION_ADMITTED,
            1,
            {
                "gate": "evidence",
                "field": "deploy_check",
                "value": "check_deployments",
                "reason": "inactive_field",
                "code": None,
            },
        ),
    )


def test_accepted_evidence_resets_stale_confirmation() -> None:
    engine = _engine("verified")
    state = engine.with_confirmation(_down_state(engine))
    outcome = record_evidence(
        engine, state, "deploy_check", "check_deployments", enforce=True, turn=4
    )
    assert outcome.accepted is True
    assert outcome.state.confirmed is False
    assert outcome.events == (
        Event(
            EventType.EVIDENCE_COLLECTED,
            4,
            {"field": "deploy_check", "via": "check_deployments"},
        ),
        Event(EventType.CONFIRMATION_RESET, 4, {}),
    )


# --- confirm gate -----------------------------------------------------------------


def test_confirm_sets_code_visible_state() -> None:
    engine = _engine("verified")
    outcome = confirm(engine, _down_state(engine), turn=2)
    assert outcome.accepted is True
    assert outcome.message is None
    assert outcome.state.confirmed is True
    assert outcome.events == (Event(EventType.CONFIRMATION_SET, 2, {}),)


# --- commit gate ------------------------------------------------------------------


def _complete_down_state(engine: StateEngine) -> State:
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    state = engine.with_answer(state, "outage_time", "2026-01-10 16:00")
    state = engine.with_evidence(state, "deploy_check")
    return engine.with_answer(state, "verdict", "bad deploy")


def test_premature_commit_blocked_when_enforced() -> None:
    engine = _engine()
    outcome = commit_gate(engine, engine.initial_state(), enforce=True, turn=1)
    assert outcome.accepted is False
    assert outcome.state.submitted is False
    assert outcome.message == "Cannot submit: missing symptom."
    assert outcome.events == (
        Event(EventType.COMMIT_BLOCKED, 1, {"reason": "not_complete", "remaining": ["symptom"]}),
    )


def test_premature_commit_admitted_when_unenforced() -> None:
    engine = _engine()
    outcome = commit_gate(engine, engine.initial_state(), enforce=False, turn=1)
    assert outcome.accepted is True
    assert outcome.state.submitted is True
    assert outcome.message is None
    assert outcome.events == (
        Event(
            EventType.COMMIT_VIOLATION_ADMITTED,
            1,
            {"reason": "not_complete", "remaining": ["symptom"]},
        ),
    )


def test_complete_express_commit_emits_declaration_ordered_snapshot() -> None:
    engine = _engine()
    outcome = commit_gate(engine, _complete_down_state(engine), enforce=True, turn=6)
    assert outcome.accepted is True
    assert outcome.state.submitted is True
    assert outcome.events == (
        Event(
            EventType.COMMITTED,
            6,
            {
                "final_answers": {
                    "symptom": "down",
                    "outage_time": "2026-01-10 16:00",
                    "verdict": "bad deploy",
                },
                "evidence": ["deploy_check"],
            },
        ),
    )
    final_answers = outcome.events[0].payload["final_answers"]
    assert list(final_answers) == ["symptom", "outage_time", "verdict"]  # declaration order


def test_duplicate_commit_is_noop_in_both_modes() -> None:
    engine = _engine()
    submitted = commit_gate(engine, _complete_down_state(engine), enforce=True, turn=6).state
    for enforce in (True, False):
        outcome = commit_gate(engine, submitted, enforce=enforce, turn=7)
        assert outcome.accepted is False
        assert outcome.state == submitted
        assert outcome.message == "Already submitted."
        assert outcome.events == (Event(EventType.COMMIT_NOOP, 7, {}),)


def test_verified_commit_blocked_until_confirmed() -> None:
    engine = _engine("verified")
    state = _complete_down_state(engine)
    blocked = commit_gate(engine, state, enforce=True, turn=6)
    assert blocked.accepted is False
    assert blocked.message == "Cannot submit: confirmation required."
    assert blocked.events == (
        Event(EventType.COMMIT_BLOCKED, 6, {"reason": "not_confirmed", "remaining": []}),
    )

    confirmed = confirm(engine, state, turn=7).state
    committed = commit_gate(engine, confirmed, enforce=True, turn=8)
    assert committed.accepted is True
    assert committed.state.submitted is True


def test_verified_unconfirmed_commit_admitted_when_unenforced() -> None:
    engine = _engine("verified")
    outcome = commit_gate(engine, _complete_down_state(engine), enforce=False, turn=6)
    assert outcome.accepted is True
    assert outcome.state.submitted is True
    assert outcome.events == (
        Event(
            EventType.COMMIT_VIOLATION_ADMITTED,
            6,
            {"reason": "not_confirmed", "remaining": []},
        ),
    )


def test_commit_blocked_again_after_post_confirmation_change() -> None:
    # The loophole the confirmation reset closes: confirm -> change -> commit must re-block.
    engine = _engine("verified")
    state = confirm(engine, _complete_down_state(engine), turn=5).state
    changed = answer_gate(
        engine, state, "outage_time", "2026-01-11 09:00", enforce=True, turn=6, today=_TODAY
    )
    assert changed.state.confirmed is False
    blocked = commit_gate(engine, changed.state, enforce=True, turn=7)
    assert blocked.accepted is False
    assert blocked.events == (
        Event(EventType.COMMIT_BLOCKED, 7, {"reason": "not_confirmed", "remaining": []}),
    )


# --- end-to-end determinism --------------------------------------------------------


def _scripted_run(buffer: StringIO) -> None:
    engine = _engine()
    sink = JsonlEventSink(buffer)
    state = engine.initial_state()

    def apply(outcome: GateOutcome) -> None:
        nonlocal state
        state = outcome.state
        for event in outcome.events:
            sink.emit(event)

    def answer(name: str, value: str, turn: int) -> None:
        apply(answer_gate(engine, state, name, value, enforce=True, turn=turn, today=_TODAY))

    answer("symptom", "down", 1)
    answer("verdict", "bad deploy", 2)  # rejected (evidence_missing) — rejections log too
    apply(record_evidence(engine, state, "deploy_check", "check_deployments", enforce=True, turn=3))
    answer("outage_time", "2026-01-10 16:00", 4)
    answer("verdict", "bad deploy", 5)
    apply(commit_gate(engine, state, enforce=True, turn=6))
    apply(commit_gate(engine, state, enforce=True, turn=7))  # duplicate -> commit_noop


def test_scripted_run_event_log_is_byte_identical() -> None:
    first, second = StringIO(), StringIO()
    _scripted_run(first)
    _scripted_run(second)
    assert first.getvalue() == second.getvalue()
    lines = first.getvalue().splitlines()
    assert len(lines) == 10
    assert lines[0] == (
        '{"seq": 0, "turn": 1, "type": "answer_submitted", "field": "symptom", "value": "down"}'
    )
    assert lines[-1] == '{"seq": 9, "turn": 7, "type": "commit_noop"}'
