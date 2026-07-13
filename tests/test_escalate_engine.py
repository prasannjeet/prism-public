"""Engine support for the escalate terminal: a pure immutable flag, like submitted."""

from __future__ import annotations

from prism.engine import StateEngine
from prism.schema import Field, FieldType, Profile, Workflow


def _engine() -> StateEngine:
    wf = Workflow(
        name="t",
        profile=Profile.EXPRESS,
        fields=(Field(name="a", type=FieldType.TEXT, required=True),),
    )
    return StateEngine(wf)


def test_initial_state_not_escalated() -> None:
    assert _engine().initial_state().escalated is False


def test_with_escalation_sets_flag_immutably() -> None:
    engine = _engine()
    state = engine.initial_state()
    escalated = engine.with_escalation(state)
    assert escalated.escalated is True
    assert state.escalated is False  # original untouched (frozen dataclass)


def test_with_escalation_preserves_answers_and_evidence() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "a", "x")
    escalated = engine.with_escalation(state)
    assert escalated.answers == {"a": "x"}
    assert escalated.escalated is True
