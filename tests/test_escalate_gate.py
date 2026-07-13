"""The escalate gate: always accepts, emits ESCALATED, sets the terminal flag.

It is enforce-independent.
"""

from __future__ import annotations

from prism.engine import StateEngine
from prism.events import EventType
from prism.gates import escalate
from prism.schema import Escalation, Field, FieldType, Profile, Workflow


def _engine() -> StateEngine:
    wf = Workflow(
        name="t",
        profile=Profile.EXPRESS,
        fields=(Field(name="a", type=FieldType.TEXT, required=True),),
        escalation=Escalation(when="evidence_insufficient", action="escalate_to_engineer"),
    )
    return StateEngine(wf)


def test_escalate_accepts_and_sets_flag() -> None:
    engine = _engine()
    outcome = escalate(engine, engine.initial_state(), turn=2)
    assert outcome.accepted is True
    assert outcome.state.escalated is True
    assert outcome.message is None


def test_escalate_emits_escalated_event_with_action() -> None:
    engine = _engine()
    outcome = escalate(engine, engine.initial_state(), turn=2)
    assert [e.type for e in outcome.events] == [EventType.ESCALATED]
    assert outcome.events[0].payload == {"action": "escalate_to_engineer"}
    assert outcome.events[0].turn == 2


def test_escalate_default_action_when_no_escalation_configured() -> None:
    wf = Workflow(
        name="t",
        profile=Profile.EXPRESS,
        fields=(Field(name="a", type=FieldType.TEXT, required=True),),
    )
    engine = StateEngine(wf)
    outcome = escalate(engine, engine.initial_state(), turn=1)
    assert outcome.accepted is True
    assert outcome.events[0].payload == {"action": "escalate"}
