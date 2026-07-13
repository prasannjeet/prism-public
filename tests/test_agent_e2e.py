"""End-to-end scripted conversations — the C4-vs-C5 divergence is one enforce flag."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from prism.agent import CONFIGS, Agent
from prism.engine import StateEngine
from prism.events import MemoryEventSink
from prism.models import ModelResponse, ScriptedAdapter, ToolCall, Usage
from prism.schema import parse_workflow


class _StubExecutor:
    """Minimal tool executor for tests that don't exercise evidence content."""

    def run(self, tool_name: str, arguments: object) -> str:
        return f"[stub {tool_name}]"

TODAY = date(2026, 1, 15)
WORKFLOW: Mapping[str, Any] = {
    "workflow": "demo",
    "profile": "verified",
    "fields": [
        {"name": "symptom", "type": "select", "required": True, "options": ["down", "slow"]}
    ],
}


def _resp(calls: tuple[ToolCall, ...], stop: str) -> ModelResponse:
    return ModelResponse("", calls, Usage(0, 0, 0, 0, {}), stop)


def _run(config_name: str) -> list[dict[str, object]]:
    # Model tries to commit BEFORE answering anything — a premature, unconfirmed commit.
    script = [
        _resp((ToolCall("c1", "commit", {}),), "tool_use"),
        _resp((), "end_turn"),
    ]
    sink = MemoryEventSink()
    engine = StateEngine(parse_workflow(WORKFLOW))
    agent = Agent(
        engine, ScriptedAdapter(script), CONFIGS[config_name], sink, _StubExecutor(), today=TODAY
    )
    agent.run_turn(engine.initial_state(), "just submit it", turn=1)
    return sink.records


def test_c5_blocks_premature_commit() -> None:
    records = _run("C5")
    assert [r["type"] for r in records] == [
        "user_message",
        "tool_called",
        "commit_blocked",
        "tool_result",
        "assistant_message",
    ]
    blocked = records[2]
    assert blocked["turn"] == 1
    assert blocked["reason"] == "not_complete"
    assert blocked["remaining"] == ["symptom"]


def test_c4_admits_premature_commit_and_logs_it() -> None:
    records = _run("C4")
    assert [r["type"] for r in records] == [
        "user_message",
        "tool_called",
        "commit_violation_admitted",
        "tool_result",
        "assistant_message",
    ]
    admitted = records[2]
    assert admitted["turn"] == 1
    assert admitted["reason"] == "not_complete"
    assert admitted["remaining"] == ["symptom"]
