"""Turn loop + the five configurations C1–C5 as data over one unchanged engine.

A Config is a small bundle of (which projection tiers render, which tools exist, one enforce
boolean). The loop body is identical for every config — the engine and loop never branch on the
configuration. That identity IS the paper's fairness argument."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Literal

from .engine import State, StateEngine
from .events import Event, EventSink, EventType
from .models import Message, ModelAdapter, ToolResult
from .projection import render_current_field, render_raw, render_schema, render_state
from .tools import ToolDispatcher, ToolExecutor, agent_tool_specs, domain_tool_specs

PerTurnPayload = Literal["none", "current_field", "raw_yaml"]


@dataclass(frozen=True)
class Config:
    name: str
    tier1_schema: bool
    tier2_state: bool
    get_detail_tool: bool
    per_turn_payload: PerTurnPayload
    enforce: bool
    system_prompt: str = ""


CONFIGS: dict[str, Config] = {
    "C1": Config(
        "C1",
        tier1_schema=False,
        tier2_state=False,
        get_detail_tool=False,
        per_turn_payload="none",
        enforce=False,
    ),
    "C2": Config(
        "C2",
        tier1_schema=False,
        tier2_state=False,
        get_detail_tool=False,
        per_turn_payload="current_field",
        enforce=True,
    ),
    "C3": Config(
        "C3",
        tier1_schema=False,
        tier2_state=True,
        get_detail_tool=False,
        per_turn_payload="raw_yaml",
        enforce=True,
    ),
    "C4": Config(
        "C4",
        tier1_schema=True,
        tier2_state=True,
        get_detail_tool=True,
        per_turn_payload="none",
        enforce=False,
    ),
    "C5": Config(
        "C5",
        tier1_schema=True,
        tier2_state=True,
        get_detail_tool=True,
        per_turn_payload="none",
        enforce=True,
    ),
}


def load_configs(prompts_dir: Path) -> dict[str, Config]:
    """Bind each base config to its versioned system prompt file (prompts/<name>.txt)."""
    result: dict[str, Config] = {}
    for name, config in CONFIGS.items():
        prompt_text = (prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
        result[name] = replace(config, system_prompt=prompt_text)
    return result


class Agent:
    """One turn loop, identical for every configuration. Collaborators are injected (no globals)."""

    def __init__(
        self,
        engine: StateEngine,
        adapter: ModelAdapter,
        config: Config,
        sink: EventSink,
        executor: ToolExecutor,
        *,
        today: date,
    ) -> None:
        self.engine = engine
        self.adapter = adapter
        self.config = config
        self._sink = sink
        self._dispatcher = ToolDispatcher(engine, executor, enforce=config.enforce, today=today)
        self._today = today
        self._tools = [
            *agent_tool_specs(engine.workflow, get_detail=config.get_detail_tool),
            *domain_tool_specs(engine.workflow),
        ]
        self._system = self._build_system()

    def _build_system(self) -> str:
        parts = [self.config.system_prompt, f"Today's date is {self._today.isoformat()}."]
        if self.config.tier1_schema:
            parts.append(render_schema(self.engine.workflow))
        return "\n\n".join(p for p in parts if p)

    def _turn_context(self, state: State) -> str:
        blocks: list[str] = []
        if self.config.tier2_state:
            blocks.append(render_state(self.engine, state))
        match self.config.per_turn_payload:
            case "current_field":
                blocks.append(render_current_field(self.engine, state))
            case "raw_yaml":
                blocks.append(render_raw(self.engine.workflow))
            case "none":
                pass
        return "\n\n".join(blocks)

    def _emit(self, event: Event) -> None:
        self._sink.emit(event)

    def run_turn(
        self, state: State, user_text: str, *, turn: int, max_tool_iterations: int = 25
    ) -> State:
        self._emit(Event(EventType.USER_MESSAGE, turn, {"text": user_text}))
        context = self._turn_context(state)
        first_text = f"{user_text}\n\n{context}" if context else user_text
        messages: list[Message] = [Message(role="user", text=first_text)]

        for _ in range(max_tool_iterations):
            response = self.adapter.complete(self._system, messages, self._tools)
            messages.append(
                Message(role="assistant", text=response.text, tool_calls=response.tool_calls)
            )
            if not response.tool_calls:
                self._emit(Event(EventType.ASSISTANT_MESSAGE, turn, {"text": response.text}))
                return state
            results: list[ToolResult] = []
            for call in response.tool_calls:
                self._emit(
                    Event(
                        EventType.TOOL_CALLED,
                        turn,
                        {"tool": call.name, "arguments": dict(call.arguments)},
                    )
                )
                state, result, events = self._dispatcher.handle(state, call, turn=turn)
                for event in events:
                    self._emit(event)
                self._emit(
                    Event(
                        EventType.TOOL_RESULT, turn, {"tool": call.name, "content": result.content}
                    )
                )
                results.append(result)
            messages.append(Message(role="tool", tool_results=tuple(results)))

        self._emit(
            Event(EventType.TOOL_ERROR, turn, {"tool": "<loop>", "reason": "max_iterations"})
        )
        return state
