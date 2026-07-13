"""Agent-facing tool surface over the gates."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from prism.engine import StateEngine
from prism.events import EventType
from prism.models import ToolCall
from prism.schema import parse_workflow
from prism.tools import ToolDispatcher, agent_tool_specs, domain_tool_specs


class _StubExecutor:
    """Minimal tool executor for tests that don't exercise evidence content."""

    def run(self, tool_name: str, arguments: object) -> str:
        return f"[stub {tool_name}]"

WORKFLOW: Mapping[str, Any] = {
    "workflow": "demo",
    "profile": "verified",
    "fields": [
        {"name": "symptom", "type": "select", "required": True, "options": ["server_down", "slow"]},
        {
            "name": "deploy_check",
            "type": "evidence",
            "required": True,
            "show_when": "symptom = server_down",
            "satisfied_by": ["check_deploys"],
        },
        {
            "name": "conclusion",
            "type": "conclusion",
            "required": True,
            "requires_evidence": ["deploy_check"],
        },
    ],
    "tools": [{"name": "check_deploys", "description": "List recent deploys."}],
}


def test_agent_tool_specs_include_core_tools() -> None:
    engine = StateEngine(parse_workflow(WORKFLOW))
    names = {s.name for s in agent_tool_specs(engine.workflow, get_detail=True)}
    assert names == {"submit_answer", "skip_optional", "get_detail", "confirm", "commit"}


def test_get_detail_omitted_when_flag_false() -> None:
    engine = StateEngine(parse_workflow(WORKFLOW))
    names = {s.name for s in agent_tool_specs(engine.workflow, get_detail=False)}
    assert names == {"submit_answer", "skip_optional", "confirm", "commit"}


def test_domain_tool_specs_come_from_workflow_tools() -> None:
    engine = StateEngine(parse_workflow(WORKFLOW))
    specs = domain_tool_specs(engine.workflow)
    assert [s.name for s in specs] == ["check_deploys"]
    assert specs[0].description == "List recent deploys."


# ---------------------------------------------------------------------------
# Task 5 — ToolDispatcher
# ---------------------------------------------------------------------------

TODAY = date(2026, 1, 15)


def _dispatcher() -> ToolDispatcher:
    engine = StateEngine(parse_workflow(WORKFLOW))
    return ToolDispatcher(engine, _StubExecutor(), enforce=True, today=TODAY)


def test_submit_answer_routes_through_answer_gate() -> None:
    d = _dispatcher()
    state = d.engine.initial_state()
    new_state, result, events = d.handle(
        state,
        ToolCall(
            id="t1", name="submit_answer", arguments={"field": "symptom", "value": "server_down"}
        ),
        turn=1,
    )
    assert new_state.answers["symptom"] == "server_down"
    assert result.is_error is False
    assert events[0].type is EventType.ANSWER_SUBMITTED


def test_rejected_answer_returns_error_result_and_unchanged_state() -> None:
    d = _dispatcher()
    state = d.engine.initial_state()
    new_state, result, events = d.handle(
        state,
        ToolCall(
            id="t1", name="submit_answer", arguments={"field": "symptom", "value": "not_an_option"}
        ),
        turn=1,
    )
    assert new_state == state
    assert result.is_error is True
    assert events[0].type is EventType.ANSWER_REJECTED


def test_get_detail_returns_field_detail_without_mutating_state() -> None:
    d = _dispatcher()
    state = d.engine.initial_state()
    new_state, result, events = d.handle(
        state, ToolCall(id="t1", name="get_detail", arguments={"field": "symptom"}), turn=1
    )
    assert new_state == state
    assert "Field: symptom" in result.content
    assert events == ()


def test_domain_tool_records_evidence_via_satisfied_by() -> None:
    d = _dispatcher()
    state = d.engine.with_answer(d.engine.initial_state(), "symptom", "server_down")
    new_state, result, events = d.handle(
        state, ToolCall(id="t1", name="check_deploys", arguments={}), turn=2
    )
    assert "deploy_check" in new_state.evidence
    assert result.is_error is False
    assert any(e.type is EventType.EVIDENCE_COLLECTED for e in events)


_SHARED_TOOL_WORKFLOW: Mapping[str, Any] = {
    "workflow": "shared",
    "profile": "express",
    "fields": [
        {"name": "symptom", "type": "select", "required": True, "options": ["a", "b"]},
        {
            "name": "first_pull",
            "type": "evidence",
            "required": True,
            "show_when": "symptom = a",
            "satisfied_by": ["audit"],
        },
        {
            "name": "second_pull",
            "type": "evidence",
            "required": True,
            "show_when": "symptom = a",
            "satisfied_by": ["audit"],
        },
    ],
    "tools": [{"name": "audit", "description": "Shared audit tool."}],
}


def test_shared_tool_collects_each_coactive_evidence_field_in_turn() -> None:
    # One tool satisfies two co-active evidence fields: successive calls must advance distinct
    # fields, not re-record the first one. Both must end collected.
    engine = StateEngine(parse_workflow(_SHARED_TOOL_WORKFLOW))
    d = ToolDispatcher(engine, _StubExecutor(), enforce=True, today=TODAY)
    state = engine.with_answer(engine.initial_state(), "symptom", "a")
    state, _result, _events = d.handle(
        state, ToolCall(id="t1", name="audit", arguments={}), turn=1
    )
    assert state.evidence == frozenset({"first_pull"})
    state, _result, _events = d.handle(
        state, ToolCall(id="t2", name="audit", arguments={}), turn=2
    )
    assert state.evidence == frozenset({"first_pull", "second_pull"})


def test_unknown_tool_returns_tool_error_event() -> None:
    d = _dispatcher()
    state = d.engine.initial_state()
    new_state, result, events = d.handle(
        state, ToolCall(id="t1", name="bogus", arguments={}), turn=1
    )
    assert new_state == state
    assert result.is_error is True
    assert events[0].type is EventType.TOOL_ERROR


# ---------------------------------------------------------------------------
# Task 2 — additional dispatcher-route tests
# ---------------------------------------------------------------------------

SKIP_WORKFLOW: Mapping[str, Any] = {
    "workflow": "demo_skip",
    "profile": "express",
    "fields": [
        {"name": "symptom", "type": "select", "required": True, "options": ["server_down", "slow"]},
        {"name": "note", "type": "text", "required": False},
        {"name": "extra", "type": "text", "required": False, "show_when": "symptom = slow"},
    ],
}


def _skip_dispatcher() -> ToolDispatcher:
    engine = StateEngine(parse_workflow(SKIP_WORKFLOW))
    return ToolDispatcher(engine, _StubExecutor(), enforce=True, today=TODAY)


def test_confirm_routes_through_confirm_gate() -> None:
    d = _dispatcher()
    state = d.engine.initial_state()
    new_state, result, events = d.handle(
        state, ToolCall(id="t1", name="confirm", arguments={}), turn=1
    )
    assert result.is_error is False
    assert events[0].type is EventType.CONFIRMATION_SET
    assert new_state.confirmed is True


def test_conclude_aliases_commit_gate() -> None:
    d = _dispatcher()
    state = d.engine.initial_state()
    _new, result, events = d.handle(state, ToolCall(id="t1", name="conclude", arguments={}), turn=1)
    assert result.is_error is True
    assert events[0].type is EventType.COMMIT_BLOCKED


def test_skip_optional_active_field_succeeds_without_event() -> None:
    d = _skip_dispatcher()
    state = d.engine.initial_state()
    new_state, result, events = d.handle(
        state, ToolCall(id="t1", name="skip_optional", arguments={"field": "note"}), turn=1
    )
    assert new_state == state
    assert result.is_error is False
    assert result.content == "Skipped optional field 'note'."
    assert events == ()


def test_skip_optional_required_field_errors() -> None:
    d = _skip_dispatcher()
    state = d.engine.initial_state()
    _new, result, events = d.handle(
        state, ToolCall(id="t1", name="skip_optional", arguments={"field": "symptom"}), turn=1
    )
    assert result.is_error is True
    assert events[0].type is EventType.TOOL_ERROR
    assert events[0].payload["reason"] == "not_optional_or_inactive"


def test_skip_optional_inactive_field_errors() -> None:
    d = _skip_dispatcher()
    state = d.engine.initial_state()
    _new, result, events = d.handle(
        state, ToolCall(id="t1", name="skip_optional", arguments={"field": "extra"}), turn=1
    )
    assert result.is_error is True
    assert events[0].payload["reason"] == "not_optional_or_inactive"


def test_skip_optional_unknown_field_errors() -> None:
    d = _skip_dispatcher()
    state = d.engine.initial_state()
    _new, result, events = d.handle(
        state, ToolCall(id="t1", name="skip_optional", arguments={"field": "ghost"}), turn=1
    )
    assert result.is_error is True
    assert events[0].payload["reason"] == "unknown_field"


def test_get_detail_unknown_field_stamps_real_turn() -> None:
    d = _dispatcher()
    state = d.engine.initial_state()
    _new, result, events = d.handle(
        state, ToolCall(id="t1", name="get_detail", arguments={"field": "ghost"}), turn=7
    )
    assert result.is_error is True
    assert events[0].type is EventType.TOOL_ERROR
    assert events[0].turn == 7
