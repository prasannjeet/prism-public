"""Consequence boundary: active-path, validation, confirmation, idempotent commit.

Wraps the pure engine; never modifies it. The `enforce` flag is the whole C4-vs-C5
difference: True rejects admissible violations, False admits them and logs `*_admitted`
events. Structural impossibilities (the engine itself would raise) reject in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from .engine import State, StateEngine
from .events import Event, EventType
from .schema import FieldType, Profile
from .validators import canonical_value, validate_field


@dataclass(frozen=True)
class GateOutcome:
    accepted: bool  # did the action take effect on state?
    state: State  # new state if accepted, original if rejected
    message: str | None  # reason surfaced to the LLM (None on accept/admit)
    events: tuple[Event, ...]  # seq-less; the sink assigns seq


def answer_gate(
    engine: StateEngine,
    state: State,
    name: str,
    raw: object,
    *,
    enforce: bool,
    turn: int,
    today: date,
) -> GateOutcome:
    workflow = engine.workflow
    if state.submitted:
        event = Event(
            EventType.ANSWER_REJECTED,
            turn,
            {"field": name, "value": raw, "reason": "already_submitted", "code": None},
        )
        message = "The workflow is already submitted; answers cannot change."
        return GateOutcome(accepted=False, state=state, message=message, events=(event,))
    if not workflow.has_field(name):
        # Structural: the engine would raise UnknownFieldError — nothing to admit in any mode.
        event = Event(
            EventType.ANSWER_REJECTED,
            turn,
            {"field": name, "value": raw, "reason": "unknown_field", "code": None},
        )
        message = f"Unknown field '{name}'."
        return GateOutcome(accepted=False, state=state, message=message, events=(event,))
    fld = workflow.field(name)
    if fld.type is FieldType.EVIDENCE:
        return _answer_violation(
            engine,
            state,
            name,
            raw,
            enforce=enforce,
            turn=turn,
            reason="wrong_kind",
            code=None,
            message=f"Field '{name}' is evidence; it is collected via tools, not answers.",
        )
    if not engine.is_active(fld, state.answers):
        return _answer_violation(
            engine,
            state,
            name,
            raw,
            enforce=enforce,
            turn=turn,
            reason="inactive_field",
            code=None,
            message=f"Field '{name}' is not on the active path.",
        )
    if fld.type is FieldType.CONCLUSION:
        missing = tuple(ev for ev in fld.requires_evidence if ev not in state.evidence)
        if missing:
            return _answer_violation(
                engine,
                state,
                name,
                raw,
                enforce=enforce,
                turn=turn,
                reason="evidence_missing",
                code=None,
                message=f"Cannot answer '{name}' yet: missing evidence {', '.join(missing)}.",
            )
    result = validate_field(fld, raw, today=today)
    if not result.ok:
        return _answer_violation(
            engine,
            state,
            name,
            raw,
            enforce=enforce,
            turn=turn,
            reason="invalid_value",
            code=result.code,
            message=result.message,
        )
    head = Event(EventType.ANSWER_SUBMITTED, turn, {"field": name, "value": raw})
    # Store the canonical ISO form for DATE/DATETIME so committed values are
    # representation-independent (a space- vs T-separated datetime scores the same).
    return _apply_answer(engine, state, name, canonical_value(fld, raw), turn, head)


def _answer_violation(
    engine: StateEngine,
    state: State,
    name: str,
    raw: object,
    *,
    enforce: bool,
    turn: int,
    reason: str,
    code: str | None,
    message: str | None,
) -> GateOutcome:
    if enforce:
        event = Event(
            EventType.ANSWER_REJECTED,
            turn,
            {"field": name, "value": raw, "reason": reason, "code": code},
        )
        return GateOutcome(accepted=False, state=state, message=message, events=(event,))
    head = Event(
        EventType.VIOLATION_ADMITTED,
        turn,
        {"gate": "answer", "field": name, "value": raw, "reason": reason, "code": code},
    )
    return _apply_answer(engine, state, name, raw, turn, head)


def _apply_answer(
    engine: StateEngine, state: State, name: str, raw: object, turn: int, head: Event
) -> GateOutcome:
    new_state = engine.with_answer(state, name, raw)
    events = [head, *_branch_events(engine, state, new_state, turn)]
    if state.confirmed:
        # A confirmation attests the data it saw; any later mutation invalidates it.
        new_state = replace(new_state, confirmed=False)
        events.append(Event(EventType.CONFIRMATION_RESET, turn, {}))
    return GateOutcome(accepted=True, state=new_state, message=None, events=tuple(events))


def _branch_events(
    engine: StateEngine, before: State, after: State, turn: int
) -> tuple[Event, ...]:
    before_active = {f.name for f in engine.active_fields(before.answers)}
    after_active = {f.name for f in engine.active_fields(after.answers)}
    events: list[Event] = []
    for fld in engine.workflow.fields:
        if fld.name in after_active and fld.name not in before_active:
            events.append(Event(EventType.BRANCH_ACTIVATED, turn, {"field": fld.name}))
    for fld in engine.workflow.fields:
        was_present = fld.name in before.answers or fld.name in before.evidence
        now_present = fld.name in after.answers or fld.name in after.evidence
        if was_present and not now_present:
            events.append(Event(EventType.BRANCH_CLEANED, turn, {"field": fld.name}))
    return tuple(events)


def record_evidence(
    engine: StateEngine, state: State, name: str, via: str, *, enforce: bool, turn: int
) -> GateOutcome:
    workflow = engine.workflow
    if state.submitted:
        return _evidence_rejected(
            state,
            name,
            via,
            turn,
            reason="already_submitted",
            message="The workflow is already submitted; evidence cannot change.",
        )
    if not workflow.has_field(name):
        return _evidence_rejected(
            state, name, via, turn, reason="unknown_field", message=f"Unknown field '{name}'."
        )
    fld = workflow.field(name)
    if fld.type is not FieldType.EVIDENCE:
        # Structural: engine.with_evidence raises FieldKindError — nothing to admit.
        return _evidence_rejected(
            state,
            name,
            via,
            turn,
            reason="wrong_kind",
            message=f"Field '{name}' is not an evidence field.",
        )
    if via not in fld.satisfied_by:
        # Caller-bug guard: the Phase-1.2 tool layer selects the field BY satisfied_by.
        return _evidence_rejected(
            state,
            name,
            via,
            turn,
            reason="wrong_tool",
            message=f"Tool '{via}' does not satisfy evidence field '{name}'.",
        )
    if not engine.is_active(fld, state.answers):
        if enforce:
            return _evidence_rejected(
                state,
                name,
                via,
                turn,
                reason="inactive_field",
                message=f"Field '{name}' is not on the active path.",
            )
        head = Event(
            EventType.VIOLATION_ADMITTED,
            turn,
            {
                "gate": "evidence",
                "field": name,
                "value": via,
                "reason": "inactive_field",
                "code": None,
            },
        )
        return _apply_evidence(engine, state, name, turn, head)
    head = Event(EventType.EVIDENCE_COLLECTED, turn, {"field": name, "via": via})
    return _apply_evidence(engine, state, name, turn, head)


def _evidence_rejected(
    state: State, name: str, via: str, turn: int, *, reason: str, message: str
) -> GateOutcome:
    event = Event(EventType.EVIDENCE_REJECTED, turn, {"field": name, "via": via, "reason": reason})
    return GateOutcome(accepted=False, state=state, message=message, events=(event,))


def _apply_evidence(
    engine: StateEngine, state: State, name: str, turn: int, head: Event
) -> GateOutcome:
    new_state = engine.with_evidence(state, name)
    events = [head, *_branch_events(engine, state, new_state, turn)]
    if state.confirmed:
        new_state = replace(new_state, confirmed=False)
        events.append(Event(EventType.CONFIRMATION_RESET, turn, {}))
    return GateOutcome(accepted=True, state=new_state, message=None, events=tuple(events))


def confirm(engine: StateEngine, state: State, *, turn: int) -> GateOutcome:
    new_state = engine.with_confirmation(state)
    event = Event(EventType.CONFIRMATION_SET, turn, {})
    return GateOutcome(accepted=True, state=new_state, message=None, events=(event,))


def escalate(engine: StateEngine, state: State, *, turn: int) -> GateOutcome:
    """Escape hatch: a valid terminal when a conclusion cannot be reached. Always admitted
    (not a violation), so it carries no enforce flag. Records the workflow's declared action."""
    action = engine.workflow.escalation.action if engine.workflow.escalation else "escalate"
    new_state = engine.with_escalation(state)
    event = Event(EventType.ESCALATED, turn, {"action": action})
    return GateOutcome(accepted=True, state=new_state, message=None, events=(event,))


def commit_gate(engine: StateEngine, state: State, *, enforce: bool, turn: int) -> GateOutcome:
    if state.submitted:
        event = Event(EventType.COMMIT_NOOP, turn, {})
        return GateOutcome(
            accepted=False, state=state, message="Already submitted.", events=(event,)
        )
    remaining = tuple(f.name for f in engine.remaining_mandatory(state))
    if remaining:
        return _commit_violation(
            engine,
            state,
            enforce=enforce,
            turn=turn,
            reason="not_complete",
            remaining=remaining,
            message=f"Cannot submit: missing {', '.join(remaining)}.",
        )
    if engine.workflow.profile is Profile.VERIFIED and not state.confirmed:
        return _commit_violation(
            engine,
            state,
            enforce=enforce,
            turn=turn,
            reason="not_confirmed",
            remaining=(),
            message="Cannot submit: confirmation required.",
        )
    new_state = engine.with_submission(state)
    payload: dict[str, object] = {
        "final_answers": _ordered_answers(engine, state),
        "evidence": _ordered_evidence(engine, state),
    }
    event = Event(EventType.COMMITTED, turn, payload)
    return GateOutcome(accepted=True, state=new_state, message=None, events=(event,))


def _commit_violation(
    engine: StateEngine,
    state: State,
    *,
    enforce: bool,
    turn: int,
    reason: str,
    remaining: tuple[str, ...],
    message: str,
) -> GateOutcome:
    payload: dict[str, object] = {"reason": reason, "remaining": list(remaining)}
    if enforce:
        event = Event(EventType.COMMIT_BLOCKED, turn, payload)
        return GateOutcome(accepted=False, state=state, message=message, events=(event,))
    new_state = engine.with_submission(state)
    event = Event(EventType.COMMIT_VIOLATION_ADMITTED, turn, payload)
    return GateOutcome(accepted=True, state=new_state, message=None, events=(event,))


def _ordered_answers(engine: StateEngine, state: State) -> dict[str, object]:
    return {
        f.name: state.answers[f.name] for f in engine.workflow.fields if f.name in state.answers
    }


def _ordered_evidence(engine: StateEngine, state: State) -> list[str]:
    return [f.name for f in engine.workflow.fields if f.name in state.evidence]
