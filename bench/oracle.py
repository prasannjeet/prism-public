"""A deterministic ideal agent. Given a scenario's machine-readable ground truth and the engine,
it plans the ideal tool-call sequence (driver-before-branch ordering falls out of re-querying
remaining_mandatory after each simulated step). Used for the zero-API end-to-end test of every
scenario and as the reachability check. Replays like ScriptedAdapter, but the script is COMPUTED.
"""

from __future__ import annotations

from prism.engine import StateEngine
from prism.models import Message, ModelResponse, ModelSettings, ToolCall, ToolSpec, Usage
from prism.schema import FieldType, Profile

from .scenario import Scenario


def plan_actions(engine: StateEngine, scenario: Scenario) -> tuple[ToolCall, ...]:
    target_escalate = tuple(scenario.expected.valid_completion) == ("escalated",)
    final_state = scenario.expected.final_state
    state = engine.initial_state()
    calls: list[ToolCall] = []
    index = 0
    while True:
        remaining = engine.remaining_mandatory(state)
        actionable = [
            f for f in remaining if not (target_escalate and f.type is FieldType.CONCLUSION)
        ]
        if not actionable:
            break
        fld = actionable[0]
        index += 1
        if fld.type is FieldType.EVIDENCE:
            tool = fld.satisfied_by[0]  # schema guarantees satisfied_by is non-empty for EVIDENCE
            call = ToolCall(f"o{index}", tool, {})
            new_state = engine.with_evidence(state, fld.name)
        else:
            value = final_state[fld.name]
            call = ToolCall(f"o{index}", "submit_answer", {"field": fld.name, "value": value})
            new_state = engine.with_answer(state, fld.name, value)
        if new_state == state:
            raise RuntimeError(
                f"oracle made no progress on scenario '{scenario.id}': stuck on field '{fld.name}' "
                f"(a conclusion may be declared before its required evidence)"
            )
        calls.append(call)
        state = new_state

    index += 1
    if target_escalate:
        calls.append(ToolCall(f"o{index}", "escalate", {}))
    elif engine.workflow.profile is Profile.VERIFIED:
        calls.append(ToolCall(f"o{index}", "confirm", {}))
        index += 1
        calls.append(ToolCall(f"o{index}", "commit", {}))
    else:
        calls.append(ToolCall(f"o{index}", "commit", {}))
    return tuple(calls)


class OracleAdapter:
    """ModelAdapter that replays a computed ideal plan. Turn-agnostic: it acts as fast as the
    loop allows (it already knows the target), draining its plan, then returns empty."""

    name = "oracle"
    settings = ModelSettings()

    def __init__(self, engine: StateEngine, scenario: Scenario) -> None:
        self._plan = plan_actions(engine, scenario)
        self._i = 0

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        if self._i < len(self._plan):
            call = self._plan[self._i]
            self._i += 1
            return ModelResponse("", (call,), Usage(0, 0, 0, 0, {}), "tool_use")
        return ModelResponse("", (), Usage(0, 0, 0, 0, {}), "end_turn")
