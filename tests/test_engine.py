"""StateEngine tests — active path, cascading cleanup, completion. Both branches asserted."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from prism.engine import FieldKindError, StateEngine, UnknownFieldError
from prism.schema import parse_workflow


def _engine() -> StateEngine:
    # symptom -> two exclusive branches; one branch carries evidence + a conclusion that needs it.
    data = {
        "workflow": "incident",
        "profile": "express",
        "fields": [
            {"name": "symptom", "type": "select", "required": True, "options": ["down", "slow"]},
            {
                "name": "outage_time",
                "type": "datetime",
                "required": True,
                "show_when": "symptom = down",
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


def _names(fields) -> set[str]:
    return {f.name for f in fields}


def test_initial_state_only_root_active() -> None:
    engine = _engine()
    state = engine.initial_state()
    assert _names(engine.active_fields(state.answers)) == {"symptom"}
    assert _names(engine.remaining_mandatory(state)) == {"symptom"}
    assert engine.is_complete(state) is False


def test_branch_activates_on_answer_down() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    assert _names(engine.active_fields(state.answers)) == {
        "symptom",
        "outage_time",
        "deploy_check",
        "verdict",
    }


def test_other_branch_activates_on_answer_slow() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "slow")
    assert _names(engine.active_fields(state.answers)) == {"symptom", "latency"}


def test_inactive_field_answer_is_rejected_by_pruning() -> None:
    engine = _engine()
    # 'latency' belongs to the slow branch; answering it under the down branch must not stick.
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    state = engine.with_answer(state, "latency", "200ms")
    assert "latency" not in state.answers


def test_cascading_cleanup_on_branch_switch() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    state = engine.with_answer(state, "outage_time", "2026-06-11T16:00")
    state = engine.with_evidence(state, "deploy_check")
    assert state.answers["outage_time"] == "2026-06-11T16:00"
    assert "deploy_check" in state.evidence

    # Switch branches: every down-branch answer AND evidence must be cleaned up.
    state = engine.with_answer(state, "symptom", "slow")
    assert dict(state.answers) == {"symptom": "slow"}
    assert state.evidence == frozenset()
    assert _names(engine.active_fields(state.answers)) == {"symptom", "latency"}


def test_evidence_on_non_evidence_field_raises() -> None:
    engine = _engine()
    with pytest.raises(FieldKindError):
        engine.with_evidence(engine.initial_state(), "symptom")


def test_unknown_field_answer_raises() -> None:
    engine = _engine()
    with pytest.raises(UnknownFieldError):
        engine.with_answer(engine.initial_state(), "ghost", 1)


def test_conclusion_requires_evidence_before_satisfied() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    state = engine.with_answer(state, "outage_time", "2026-06-11T16:00")
    state = engine.with_answer(state, "verdict", "bad deploy")
    # verdict is answered but its required evidence is absent -> not satisfied, not complete.
    verdict = engine.workflow.field("verdict")
    assert engine.is_satisfied(verdict, state) is False
    assert "deploy_check" in _names(engine.remaining_mandatory(state))
    assert engine.is_complete(state) is False

    state = engine.with_evidence(state, "deploy_check")
    assert engine.is_satisfied(verdict, state) is True


def test_full_down_branch_completion() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    state = engine.with_answer(state, "outage_time", "2026-06-11T16:00")
    state = engine.with_evidence(state, "deploy_check")
    assert engine.is_complete(state) is False  # verdict still unanswered
    state = engine.with_answer(state, "verdict", "bad deploy")
    assert engine.remaining_mandatory(state) == ()
    assert engine.is_complete(state) is True


def test_slow_branch_completion() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "slow")
    assert engine.is_complete(state) is False
    state = engine.with_answer(state, "latency", "p99 spike at 16:00")
    assert engine.remaining_mandatory(state) == ()
    assert engine.is_complete(state) is True


def test_transitions_do_not_mutate_prior_state() -> None:
    engine = _engine()
    first = engine.with_answer(engine.initial_state(), "symptom", "down")
    second = engine.with_answer(first, "symptom", "slow")
    # 'first' must be untouched by the second transition (immutability).
    assert dict(first.answers) == {"symptom": "down"}
    assert _names(engine.active_fields(first.answers)) == {
        "symptom",
        "outage_time",
        "deploy_check",
        "verdict",
    }
    assert dict(second.answers) == {"symptom": "slow"}


# ---------------------------------------------------------------------------
# Task 2 — nested cascade + with_evidence unknown-field tests
# ---------------------------------------------------------------------------

NESTED: Mapping[str, Any] = {
    "workflow": "nested",
    "profile": "express",
    "fields": [
        {"name": "a", "type": "select", "required": True, "options": ["x", "y"]},
        {
            "name": "b",
            "type": "select",
            "required": False,
            "show_when": "a = x",
            "options": ["p", "q"],
        },
        {"name": "c", "type": "text", "required": False, "show_when": "b = p"},
    ],
}


def test_two_level_show_when_cascades_in_one_transition() -> None:
    engine = StateEngine(parse_workflow(NESTED))
    s = engine.with_answer(engine.initial_state(), "a", "x")
    assert "c" not in {f.name for f in engine.active_fields(s.answers)}
    s = engine.with_answer(s, "b", "p")
    s = engine.with_answer(s, "c", "hi")
    assert s.answers == {"a": "x", "b": "p", "c": "hi"}
    s = engine.with_answer(s, "a", "y")
    assert s.answers == {"a": "y"}


def test_with_evidence_unknown_field_raises() -> None:
    engine = StateEngine(parse_workflow(NESTED))
    with pytest.raises(UnknownFieldError):
        engine.with_evidence(engine.initial_state(), "ghost")


COMPOUND = {
    "workflow": "w",
    "profile": "express",
    "fields": [
        {"name": "a", "type": "select", "required": True, "options": ["x", "y"]},
        {"name": "b", "type": "select", "required": True, "options": ["p", "q"]},
        {"name": "amount", "type": "number", "required": True},
        {
            "name": "both",
            "type": "text",
            "required": False,
            "show_when": {"all_of": ["a = x", "b = p"]},
        },
        {
            "name": "either",
            "type": "text",
            "required": False,
            "show_when": {"any_of": ["a = y", "b = q"]},
        },
        {"name": "big", "type": "text", "required": False, "show_when": "amount >= 50"},
    ],
}


def _compound_engine() -> StateEngine:
    return StateEngine(parse_workflow(COMPOUND))


def _active(engine: StateEngine, answers: dict[str, object]) -> set[str]:
    return {f.name for f in engine.active_fields(answers)}


def test_all_of_active_only_when_both_true() -> None:
    engine = _compound_engine()
    assert "both" not in _active(engine, {"a": "x"})  # b unanswered -> false
    assert "both" not in _active(engine, {"a": "x", "b": "q"})  # b wrong
    assert "both" in _active(engine, {"a": "x", "b": "p"})  # both true


def test_any_of_active_when_either_true() -> None:
    engine = _compound_engine()
    assert "either" not in _active(engine, {"a": "x", "b": "p"})  # neither
    assert "either" in _active(engine, {"a": "y", "b": "p"})  # first
    assert "either" in _active(engine, {"a": "x", "b": "q"})  # second


def test_numeric_threshold_boundaries() -> None:
    engine = _compound_engine()
    assert "big" not in _active(engine, {"amount": "49"})
    assert "big" in _active(engine, {"amount": "50"})  # >= boundary
    assert "big" in _active(engine, {"amount": "75"})
    assert "big" not in _active(engine, {})  # unanswered -> false


def test_compound_guard_cascade_prune_on_one_conjunct_change() -> None:
    engine = _compound_engine()
    state = engine.with_answer(engine.initial_state(), "a", "x")
    state = engine.with_answer(state, "b", "p")
    state = engine.with_answer(state, "both", "hello")
    assert "both" in state.answers
    # flip b: the all_of conjunct fails, "both" deactivates and its answer is pruned to fixpoint
    state = engine.with_answer(state, "b", "q")
    assert "both" not in state.answers
    assert set(state.answers) == {"a", "b"}
