"""Scenario validity guards beyond schema load. Pure: replays beats through the engine."""

from __future__ import annotations

from prism.engine import State, StateEngine
from prism.schema import FieldType

from .scenario import Beat, Scenario


def correction_lands_before_completion(scenario: Scenario, engine: StateEngine) -> bool:
    """True iff the workflow is still incomplete at the point the first correction beat
    arrives, so a correction genuinely tests mid-flow handling (it cannot be pre-empted by
    the agent completing and committing the original branch first)."""
    state = engine.initial_state()
    for beat in scenario.beats:
        if beat.intent == "correct":
            return not engine.is_complete(state)
        state = _apply(engine, state, beat)
    return True  # no correction beat: nothing to guard


def _apply(engine: StateEngine, state: State, beat: Beat) -> State:
    # Replays the user-supplied parts directly through the engine (with_answer/with_evidence),
    # NOT through the gates: an intentional, conservative approximation sufficient for the
    # completeness check.
    for part in beat.parts:
        fld = engine.workflow.field(part.field) if engine.workflow.has_field(part.field) else None
        if fld is not None and fld.type is FieldType.EVIDENCE:
            state = engine.with_evidence(state, part.field)
        elif part.value != "":
            state = engine.with_answer(state, part.field, part.value)
    return state
