"""escalate tool: exposed only when the workflow declares escalation; routes to the gate."""

from __future__ import annotations

from datetime import date

from prism.engine import StateEngine
from prism.events import EventType
from prism.models import ToolCall
from prism.schema import Escalation, Field, FieldType, Profile, Workflow
from prism.tools import ToolDispatcher, agent_tool_specs


def _workflow(*, with_escalation: bool) -> Workflow:
    return Workflow(
        name="t",
        profile=Profile.EXPRESS,
        fields=(Field(name="a", type=FieldType.TEXT, required=True),),
        escalation=(
            Escalation(when="x", action="escalate_to_engineer") if with_escalation else None
        ),
    )


class _Stub:
    def run(self, tool_name: str, arguments: object) -> str:
        return "stub"


def test_escalate_spec_present_only_with_escalation() -> None:
    names_with = {
        s.name for s in agent_tool_specs(_workflow(with_escalation=True), get_detail=False)
    }
    names_without = {
        s.name for s in agent_tool_specs(_workflow(with_escalation=False), get_detail=False)
    }
    assert "escalate" in names_with
    assert "escalate" not in names_without


def test_dispatcher_routes_escalate_to_gate() -> None:
    engine = StateEngine(_workflow(with_escalation=True))
    dispatcher = ToolDispatcher(engine, _Stub(), enforce=True, today=date(2026, 1, 15))
    state, result, events = dispatcher.handle(
        engine.initial_state(), ToolCall("c1", "escalate", {}), turn=1
    )
    assert state.escalated is True
    assert result.is_error is False
    assert [e.type for e in events] == [EventType.ESCALATED]
