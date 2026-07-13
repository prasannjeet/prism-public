"""Configurations are five literals of one frozen type over one unchanged engine."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from prism.agent import CONFIGS, Agent, Config  # noqa: F401
from prism.engine import StateEngine
from prism.events import MemoryEventSink
from prism.models import ModelResponse, ScriptedAdapter, ToolCall, Usage
from prism.schema import parse_workflow


class _StubExecutor:
    """Minimal tool executor for tests that don't exercise evidence content."""

    def run(self, tool_name: str, arguments: object) -> str:
        return f"[stub {tool_name}]"


def test_five_configs_named_c1_to_c5() -> None:
    assert set(CONFIGS) == {"C1", "C2", "C3", "C4", "C5"}


def test_config_is_frozen() -> None:
    c = CONFIGS["C5"]
    try:
        c.enforce = False  # type: ignore[misc]
        raise AssertionError("Config should be frozen")
    except AttributeError:
        pass


def test_config_matrix_matches_spec() -> None:
    # (tier1_schema, tier2_state, get_detail_tool, per_turn_payload, enforce)
    expected = {
        "C1": (False, False, False, "none", False),
        "C2": (False, False, False, "current_field", True),
        "C3": (False, True, False, "raw_yaml", True),
        "C4": (True, True, True, "none", False),
        "C5": (True, True, True, "none", True),
    }
    for name, row in expected.items():
        c = CONFIGS[name]
        assert (
            c.tier1_schema,
            c.tier2_state,
            c.get_detail_tool,
            c.per_turn_payload,
            c.enforce,
        ) == row, name


TODAY = date(2026, 1, 15)
WORKFLOW: Mapping[str, Any] = {
    "workflow": "demo",
    "profile": "express",
    "fields": [
        {"name": "symptom", "type": "select", "required": True, "options": ["down", "slow"]},
        {"name": "note", "type": "text", "required": False},
    ],
}


def _resp(calls: tuple[ToolCall, ...], text: str, stop: str) -> ModelResponse:
    return ModelResponse(text, calls, Usage(0, 0, 0, 0, {}), stop)


def _agent(config_name: str, script: list[ModelResponse], sink: MemoryEventSink) -> Agent:
    engine = StateEngine(parse_workflow(WORKFLOW))
    return Agent(
        engine, ScriptedAdapter(script), CONFIGS[config_name], sink, _StubExecutor(), today=TODAY
    )


def test_c5_system_prompt_includes_tier1_schema() -> None:
    sink = MemoryEventSink()
    agent = _agent("C5", [_resp((), "hi", "end_turn")], sink)
    state = agent.engine.initial_state()
    agent.run_turn(state, "hello", turn=1)
    system = agent.adapter.calls[0].system  # type: ignore[attr-defined]
    assert "=== WORKFLOW STRUCTURE ===" in system
    assert "2026-01-15" in system


def test_c1_system_prompt_omits_tier1_schema() -> None:
    sink = MemoryEventSink()
    agent = _agent("C1", [_resp((), "hi", "end_turn")], sink)
    agent.run_turn(agent.engine.initial_state(), "hello", turn=1)
    assert "=== WORKFLOW STRUCTURE ===" not in agent.adapter.calls[0].system  # type: ignore[attr-defined]


def test_turn_loop_executes_tool_then_replies_and_emits_events() -> None:
    sink = MemoryEventSink()
    call = ToolCall(id="t1", name="submit_answer", arguments={"field": "symptom", "value": "down"})
    agent = _agent("C5", [_resp((call,), "", "tool_use"), _resp((), "got it", "end_turn")], sink)
    state = agent.run_turn(agent.engine.initial_state(), "it is down", turn=1)
    assert state.answers["symptom"] == "down"
    assert [r["type"] for r in sink.records] == [
        "user_message",
        "tool_called",
        "answer_submitted",
        "tool_result",
        "assistant_message",
    ]


def test_max_tool_iterations_guard_ends_turn() -> None:
    sink = MemoryEventSink()
    loop = ToolCall(id="t1", name="get_detail", arguments={"field": "symptom"})
    # Always returns a tool call → would loop forever without the guard.
    agent = _agent("C5", [_resp((loop,), "", "tool_use")] * 50, sink)
    agent.run_turn(agent.engine.initial_state(), "x", turn=1, max_tool_iterations=3)
    assert any(
        r["type"] == "tool_error" and r.get("reason") == "max_iterations" for r in sink.records
    )


def test_c3_per_turn_payload_injects_raw_yaml() -> None:
    sink = MemoryEventSink()
    agent = _agent("C3", [_resp((), "ok", "end_turn")], sink)
    agent.run_turn(agent.engine.initial_state(), "hello", turn=1)
    last_user = agent.adapter.calls[0].messages[-1]  # type: ignore[attr-defined]
    assert "workflow: demo" in last_user.text


def test_load_configs_reads_prompt_per_config(tmp_path: Path) -> None:
    from prism.agent import load_configs

    for name in ("C1", "C2", "C3", "C4", "C5"):
        (tmp_path / f"{name}.txt").write_text(f"prompt for {name}", encoding="utf-8")
    configs = load_configs(tmp_path)
    assert configs["C3"].system_prompt == "prompt for C3"
    # Structural flags are unchanged from the CONFIGS matrix:
    assert configs["C5"].enforce is True and configs["C1"].enforce is False
