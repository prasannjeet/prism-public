"""Agent-facing tools over the gates. Tools never mutate state directly — they route through a
gate, so the event log stays the single source of truth. Provider-neutral ToolSpecs only."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Protocol

from .engine import State, StateEngine
from .events import Event, EventType
from .gates import GateOutcome, answer_gate, commit_gate, confirm, escalate, record_evidence
from .models import ToolCall, ToolResult, ToolSpec
from .projection import render_detail
from .schema import Workflow

# Reserved names — anything else in a tool call is a domain (evidence) tool.
RESERVED = frozenset(
    {"submit_answer", "skip_optional", "get_detail", "confirm", "commit", "escalate"}
)

_SUBMIT = ToolSpec(
    name="submit_answer",
    description="Record the user's answer to a field.",
    input_schema={
        "type": "object",
        "properties": {"field": {"type": "string"}, "value": {"type": "string"}},
        "required": ["field", "value"],
    },
)
_SKIP = ToolSpec(
    name="skip_optional",
    description="Skip an optional field the user declined to provide.",
    input_schema={
        "type": "object",
        "properties": {"field": {"type": "string"}},
        "required": ["field"],
    },
)
_GET_DETAIL = ToolSpec(
    name="get_detail",
    description="Fetch the full definition (options, validators, description) of one field.",
    input_schema={
        "type": "object",
        "properties": {"field": {"type": "string"}},
        "required": ["field"],
    },
)
_CONFIRM = ToolSpec(
    name="confirm",
    description="Confirm the collected data is correct (required before commit on verified flows).",
    input_schema={"type": "object", "properties": {}},
)
_COMMIT = ToolSpec(
    name="commit",
    description="Submit the completed workflow.",
    input_schema={"type": "object", "properties": {}},
)
_ESCALATE = ToolSpec(
    name="escalate",
    description="Escalate to a human when the workflow cannot be completed (e.g. evidence is "
    "insufficient to reach a conclusion).",
    input_schema={"type": "object", "properties": {}},
)


def agent_tool_specs(workflow: Workflow, *, get_detail: bool) -> list[ToolSpec]:
    specs = [_SUBMIT, _SKIP]
    if get_detail:
        specs.append(_GET_DETAIL)
    specs.extend([_CONFIRM, _COMMIT])
    if workflow.escalation is not None:
        specs.append(_ESCALATE)
    return specs


def domain_tool_specs(workflow: Workflow) -> list[ToolSpec]:
    return [
        ToolSpec(
            name=t.name,
            description=t.description or f"Run the {t.name} tool.",
            input_schema={"type": "object", "properties": {}},
        )
        for t in workflow.tools
    ]


class ToolExecutor(Protocol):
    def run(self, tool_name: str, arguments: Mapping[str, object]) -> str: ...


class ToolDispatcher:
    """Maps a ToolCall to the matching gate/projection and returns (state, result, events).

    `commit` and `conclude` both target the commit gate; domain tools resolve to the active
    evidence field they satisfy via `satisfied_by`."""

    def __init__(
        self, engine: StateEngine, executor: ToolExecutor, *, enforce: bool, today: date
    ) -> None:
        self.engine = engine
        self._executor = executor
        self._enforce = enforce
        self._today = today

    def handle(
        self, state: State, call: ToolCall, *, turn: int
    ) -> tuple[State, ToolResult, tuple[Event, ...]]:
        match call.name:
            case "submit_answer":
                return self._from_gate(
                    answer_gate(
                        self.engine,
                        state,
                        str(call.arguments.get("field", "")),
                        call.arguments.get("value"),
                        enforce=self._enforce,
                        turn=turn,
                        today=self._today,
                    ),
                    call,
                )
            case "confirm":
                return self._from_gate(confirm(self.engine, state, turn=turn), call)
            case "commit" | "conclude":
                return self._from_gate(
                    commit_gate(self.engine, state, enforce=self._enforce, turn=turn), call
                )
            case "escalate":
                return self._from_gate(escalate(self.engine, state, turn=turn), call)
            case "skip_optional":
                return self._skip(state, str(call.arguments.get("field", "")), call, turn)
            case "get_detail":
                return self._detail(state, str(call.arguments.get("field", "")), call, turn)
            case _:
                return self._domain_or_error(state, call, turn)

    def _from_gate(
        self, outcome: GateOutcome, call: ToolCall
    ) -> tuple[State, ToolResult, tuple[Event, ...]]:
        content = outcome.message if outcome.message is not None else "ok"
        result = ToolResult(call.id, content, is_error=not outcome.accepted)
        return outcome.state, result, outcome.events

    def _skip(
        self, state: State, name: str, call: ToolCall, turn: int
    ) -> tuple[State, ToolResult, tuple[Event, ...]]:
        if not self.engine.workflow.has_field(name):
            return self._error(state, call, "unknown_field", turn)
        field = self.engine.workflow.field(name)
        if field.required or not self.engine.is_active(field, state.answers):
            return self._error(state, call, "not_optional_or_inactive", turn)
        return state, ToolResult(call.id, f"Skipped optional field '{name}'."), ()

    def _detail(
        self, state: State, name: str, call: ToolCall, turn: int
    ) -> tuple[State, ToolResult, tuple[Event, ...]]:
        if not self.engine.workflow.has_field(name):
            return self._error(state, call, "unknown_field", turn)
        return state, ToolResult(call.id, render_detail(self.engine.workflow, name)), ()

    def _domain_or_error(
        self, state: State, call: ToolCall, turn: int
    ) -> tuple[State, ToolResult, tuple[Event, ...]]:
        field = self._evidence_field_for(call.name, state)
        if field is None:
            return self._error(state, call, "unknown_tool", turn)
        content = self._executor.run(call.name, call.arguments)
        outcome = record_evidence(
            self.engine, state, field, via=call.name, enforce=self._enforce, turn=turn
        )
        result_content = content if outcome.accepted else (outcome.message or content)
        result = ToolResult(call.id, result_content, is_error=not outcome.accepted)
        return outcome.state, result, outcome.events

    def _evidence_field_for(self, tool_name: str, state: State) -> str | None:
        active_for_tool = [
            field
            for field in self.engine.active_fields(state.answers)
            if field.type.value == "evidence" and tool_name in field.satisfied_by
        ]
        # When one tool satisfies several co-active evidence fields, prefer one still uncollected
        # so each call advances a distinct field; otherwise the first call would re-record the same
        # field every time and the later fields could never be collected.
        for field in active_for_tool:
            if field.name not in state.evidence:
                return field.name
        if active_for_tool:
            return active_for_tool[0].name
        # Fall back to any evidence field (lets the gate reject an inactive one in enforce mode).
        for field in self.engine.workflow.fields:
            if field.type.value == "evidence" and tool_name in field.satisfied_by:
                return field.name
        return None

    def _error(
        self, state: State, call: ToolCall, reason: str, turn: int
    ) -> tuple[State, ToolResult, tuple[Event, ...]]:
        event = Event(EventType.TOOL_ERROR, turn, {"tool": call.name, "reason": reason})
        return state, ToolResult(call.id, f"Tool error: {reason}.", is_error=True), (event,)
