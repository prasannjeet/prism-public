"""ScenarioToolExecutor returns scripted evidence content; a declared-but-unscripted tool
returns a neutral default (an off-script live-model call is data, not a crash)."""

from __future__ import annotations

from bench.scenario_tools import NO_DATA, ScenarioToolExecutor


def test_returns_scripted_content() -> None:
    ex = ScenarioToolExecutor({"check_logs": "OutOfMemoryError at 16:00"})
    assert ex.run("check_logs", {}) == "OutOfMemoryError at 16:00"


def test_unscripted_tool_returns_neutral_default() -> None:
    ex = ScenarioToolExecutor({})
    assert ex.run("check_logs", {}) == NO_DATA


def test_arbitrary_unscripted_tool_returns_neutral_default() -> None:
    ex = ScenarioToolExecutor({"check_logs": "OutOfMemoryError at 16:00"})
    assert ex.run("check_metrics", {}) == NO_DATA
