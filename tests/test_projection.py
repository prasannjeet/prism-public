"""Projection rendering tests — exact strings, both branches, declaration order."""

from __future__ import annotations

import pytest
import yaml

from prism.engine import StateEngine
from prism.projection import (
    render_current_field,
    render_detail,
    render_raw,
    render_schema,
    render_state,
)
from prism.schema import parse_workflow


def _engine() -> StateEngine:
    data = {
        "workflow": "incident",
        "profile": "express",
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


# --- Tier 1 -------------------------------------------------------------------


def test_render_schema_exact() -> None:
    engine = _engine()
    assert render_schema(engine.workflow) == (
        "=== WORKFLOW STRUCTURE ===\n"
        "Mandatory: symptom\n"
        '- symptom [SELECT, required, options: down|slow] - "What did you observe?"\n'
        "- outage_time [DATETIME, required, show_when: symptom = down]\n"
        "- deploy_check [EVIDENCE, required, show_when: symptom = down, via: check_deployments]\n"
        "- verdict [CONCLUSION, required, show_when: symptom = down, needs: deploy_check]\n"
        "- latency [TEXT, required, show_when: symptom = slow]"
    )


def test_render_schema_marks_optional_fields_and_empty_mandatory() -> None:
    workflow = parse_workflow({"workflow": "tiny", "fields": [{"name": "note", "type": "text"}]})
    assert render_schema(workflow) == (
        "=== WORKFLOW STRUCTURE ===\nMandatory: —\n- note [TEXT, optional]"
    )


# --- Tier 2 -------------------------------------------------------------------


def test_render_state_empty() -> None:
    engine = _engine()
    assert render_state(engine, engine.initial_state()) == (
        "Answered (0): —\n"
        "Active Branch: none\n"
        "Remaining Mandatory (1): symptom\n"
        "Evidence collected (0): —\n"
        "Complete: No"
    )


def test_render_state_down_branch() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    assert render_state(engine, state) == (
        'Answered (1): symptom = "down"\n'
        "Active Branch: symptom → down\n"
        "Remaining Mandatory (3): outage_time, deploy_check, verdict\n"
        "Evidence collected (0): —\n"
        "Complete: No"
    )


def test_render_state_other_branch() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "slow")
    assert render_state(engine, state) == (
        'Answered (1): symptom = "slow"\n'
        "Active Branch: symptom → slow\n"
        "Remaining Mandatory (1): latency\n"
        "Evidence collected (0): —\n"
        "Complete: No"
    )


def test_render_state_complete_down_branch() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    state = engine.with_answer(state, "outage_time", "2026-01-10 16:00")
    state = engine.with_evidence(state, "deploy_check")
    state = engine.with_answer(state, "verdict", "bad deploy")
    expected_answers = 'symptom = "down", outage_time = "2026-01-10 16:00", verdict = "bad deploy"'
    assert render_state(engine, state) == (
        f"Answered (3): {expected_answers}\n"
        "Active Branch: symptom → down\n"
        "Remaining Mandatory (0): —\n"
        "Evidence collected (1): deploy_check\n"
        "Complete: Yes"
    )


# --- Tier 3 -------------------------------------------------------------------


def test_render_detail_field_with_validators() -> None:
    engine = _engine()
    assert render_detail(engine.workflow, "outage_time") == (
        "Field: outage_time\n"
        "Type: DATETIME\n"
        "Required: yes\n"
        "Show when: symptom = down\n"
        "Validators: not_future = true"
    )


def test_render_detail_select_with_prompt_and_options() -> None:
    engine = _engine()
    assert render_detail(engine.workflow, "symptom") == (
        "Field: symptom\n"
        "Type: SELECT\n"
        "Required: yes\n"
        'Prompt: "What did you observe?"\n'
        "Options: down | slow"
    )


def test_render_detail_evidence_and_conclusion_attributes() -> None:
    engine = _engine()
    assert render_detail(engine.workflow, "deploy_check") == (
        "Field: deploy_check\n"
        "Type: EVIDENCE\n"
        "Required: yes\n"
        "Show when: symptom = down\n"
        "Satisfied by: check_deployments"
    )
    assert render_detail(engine.workflow, "verdict") == (
        "Field: verdict\n"
        "Type: CONCLUSION\n"
        "Required: yes\n"
        "Show when: symptom = down\n"
        "Requires evidence: deploy_check"
    )


def test_render_detail_unknown_field_raises() -> None:
    engine = _engine()
    with pytest.raises(KeyError):
        render_detail(engine.workflow, "ghost")


# --- C2 / C3 renderers ----------------------------------------------------------


def test_render_current_field_picks_first_outstanding_mandatory() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "down")
    assert render_current_field(engine, state) == (
        "Next required field:\n"
        "Field: outage_time\n"
        "Type: DATETIME\n"
        "Required: yes\n"
        "Show when: symptom = down\n"
        "Validators: not_future = true"
    )


def test_render_current_field_when_complete() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "symptom", "slow")
    state = engine.with_answer(state, "latency", "p99 spike")
    assert render_current_field(engine, state) == "All mandatory fields are complete."


def test_render_raw_round_trips_and_is_stable() -> None:
    engine = _engine()
    raw = render_raw(engine.workflow)
    assert raw == render_raw(engine.workflow)
    assert parse_workflow(yaml.safe_load(raw)) == engine.workflow


_VERIFIED = {
    "workflow": "v",
    "profile": "verified",
    "fields": [{"name": "a", "type": "text", "required": True}],
}
_EXPRESS = {
    "workflow": "e",
    "profile": "express",
    "fields": [{"name": "a", "type": "text", "required": True}],
}


def test_render_state_shows_confirmed_no_then_yes_for_verified() -> None:
    engine = StateEngine(parse_workflow(_VERIFIED))
    s0 = engine.initial_state()
    assert "Confirmed: No" in render_state(engine, s0)
    s1 = engine.with_confirmation(s0)
    assert "Confirmed: Yes" in render_state(engine, s1)


def test_render_state_omits_confirmed_for_express() -> None:
    engine = StateEngine(parse_workflow(_EXPRESS))
    assert "Confirmed:" not in render_state(engine, engine.initial_state())


COMPOUND = {
    "workflow": "w",
    "profile": "express",
    "fields": [
        {"name": "a", "type": "select", "required": True, "options": ["x", "y"]},
        {"name": "amount", "type": "number", "required": True},
        {
            "name": "g",
            "type": "text",
            "required": False,
            "show_when": {"all_of": ["a = x", "amount >= 50"]},
        },
    ],
}


def test_render_detail_shows_compound_guard() -> None:
    wf = parse_workflow(COMPOUND)
    detail = render_detail(wf, "g")
    assert "Show when: all(a = x, amount >= 50)" in detail


def test_render_schema_shows_compound_guard() -> None:
    wf = parse_workflow(COMPOUND)
    assert "show_when: all(a = x, amount >= 50)" in render_schema(wf)


def test_active_branch_recognizes_compound_parent() -> None:
    wf = parse_workflow(COMPOUND)
    engine = StateEngine(wf)
    state = engine.with_answer(
        engine.with_answer(engine.initial_state(), "a", "x"), "amount", "50"
    )
    # both a and amount gate the active child "g", so both appear in the active-branch line
    line = render_state(engine, state)
    assert "a → x" in line
    assert "amount → 50" in line
