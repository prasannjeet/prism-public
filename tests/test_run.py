"""run_scenario drives one scenario through one config with one adapter, returning the event log
+ final state. Driven by the OracleAdapter here -> a real, clean committed conversation, zero
API."""

from __future__ import annotations

from datetime import date

from bench.oracle import OracleAdapter
from bench.run import CONTINUATION_MESSAGE, run_scenario
from bench.scenario import Expected, Scenario
from prism.agent import CONFIGS
from prism.engine import StateEngine
from prism.models import ModelResponse, ModelSettings, ToolCall, Usage
from prism.schema import load_workflow

TODAY = date(2026, 1, 15)


def _w2_commit() -> Scenario:
    return Scenario(
        id="w2-happy",
        workflow="incident_investigation",
        category="happy_path",
        beats=(),
        expected=Expected(
            final_state={
                "affected_service": "checkout",
                "symptom": "server_down",
                "outage_time": "2026-01-15T16:00:00",
                "server_root_cause": "bad deploy at 15:58 crashed boot",
            },
            valid_completion=("committed",),
            required_evidence=("deploy_check", "log_check"),
        ),
        tool_script={"check_deployments": "deploy at 15:58", "check_logs": "config parse error"},
    )


def test_run_scenario_reaches_committed() -> None:
    scenario = _w2_commit()
    engine = StateEngine(load_workflow("workflows/incident_investigation.yaml"))
    adapter = OracleAdapter(engine, scenario)
    result = run_scenario(scenario, CONFIGS["C5"], adapter, today=TODAY)
    assert result.final_state.submitted is True
    types = [r["type"] for r in result.event_log]
    assert "committed" in types
    committed = next(r for r in result.event_log if r["type"] == "committed")
    assert committed["final_answers"]["symptom"] == "server_down"


_NO_USAGE = Usage(0, 0, 0, 0, {})


class _NeverTerminates:
    """Replies with plain text every call: no tool calls, so it never commits or escalates."""

    name = "stub"
    settings = ModelSettings()

    def complete(self, system, messages, tools):  # type: ignore[no-untyped-def]
        return ModelResponse("ok", (), _NO_USAGE, "end_turn")


class _ActsOnFirstNudge:
    """Plain text until the first continuation turn, then fires one tool call (commit/escalate).

    A continuation turn is identified by the first model call of a turn (len(messages) == 1)
    whose user message starts with CONTINUATION_MESSAGE; the config may append a context block
    after it, so we match the prefix, not equality."""

    name = "stub"
    settings = ModelSettings()

    def __init__(self, tool: str) -> None:
        self._tool = tool
        self._fired = False

    def complete(self, system, messages, tools):  # type: ignore[no-untyped-def]
        if (
            not self._fired
            and len(messages) == 1
            and messages[0].text.startswith(CONTINUATION_MESSAGE)
        ):
            self._fired = True
            return ModelResponse("", (ToolCall("c1", self._tool, {}),), _NO_USAGE, "tool_use")
        return ModelResponse("ok", (), _NO_USAGE, "end_turn")


def _nudge_turns(event_log) -> list:  # type: ignore[no-untyped-def]
    return [
        r
        for r in event_log
        if r.get("type") == "user_message" and r.get("text") == CONTINUATION_MESSAGE
    ]


def test_no_continuation_when_cap_is_zero() -> None:
    # Default cap 0 -> the frozen behavior: no nudge turns appended.
    result = run_scenario(_w2_commit(), CONFIGS["C5"], _NeverTerminates(), today=TODAY)
    assert _nudge_turns(result.event_log) == []
    assert result.final_state.submitted is False


def test_continuation_fires_until_cap_when_not_terminal() -> None:
    result = run_scenario(
        _w2_commit(), CONFIGS["C5"], _NeverTerminates(), today=TODAY, max_continuation_turns=5
    )
    nudges = _nudge_turns(result.event_log)
    assert len(nudges) == 5  # never terminal -> burns the whole cap
    assert all(r["text"] == CONTINUATION_MESSAGE for r in nudges)
    assert result.final_state.submitted is False


def test_no_continuation_when_already_terminal() -> None:
    # The OracleAdapter commits during the scripted turns; continuation sees a terminal state
    # and adds nothing even with a generous cap.
    scenario = _w2_commit()
    engine = StateEngine(load_workflow("workflows/incident_investigation.yaml"))
    adapter = OracleAdapter(engine, scenario)
    result = run_scenario(scenario, CONFIGS["C5"], adapter, today=TODAY, max_continuation_turns=5)
    assert result.final_state.submitted is True
    assert _nudge_turns(result.event_log) == []


def test_continuation_stops_as_soon_as_escalate_reached() -> None:
    result = run_scenario(
        _w2_commit(),
        CONFIGS["C5"],
        _ActsOnFirstNudge("escalate"),
        today=TODAY,
        max_continuation_turns=5,
    )
    assert result.final_state.escalated is True
    assert len(_nudge_turns(result.event_log)) == 1  # stopped after the first nudge


def test_admitted_premature_commit_stops_continuation_under_enforce_off() -> None:
    # C1 (enforce off): a premature commit is ADMITTED -> submitted is set (but the event is
    # commit_violation_admitted, not committed). The stop condition is the state flag, so the
    # loop stops and the cell is not re-prodded.
    result = run_scenario(
        _w2_commit(),
        CONFIGS["C1"],
        _ActsOnFirstNudge("commit"),
        today=TODAY,
        max_continuation_turns=5,
    )
    assert result.final_state.submitted is True
    assert len(_nudge_turns(result.event_log)) == 1
    types = [r["type"] for r in result.event_log]
    assert "commit_violation_admitted" in types
    assert "committed" not in types


def test_nudge_text_is_config_independent() -> None:
    for cfg in ("C1", "C5"):
        result = run_scenario(
            _w2_commit(), CONFIGS[cfg], _NeverTerminates(), today=TODAY, max_continuation_turns=2
        )
        nudges = _nudge_turns(result.event_log)
        assert len(nudges) == 2
        assert {r["text"] for r in nudges} == {CONTINUATION_MESSAGE}
