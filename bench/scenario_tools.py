"""Deterministic, scenario-keyed tool executor. Replaces prism.tools.StubToolExecutor: the
returned content is part of the scenario (so 'misleading evidence' is a controllable probe)."""

from __future__ import annotations

from collections.abc import Mapping

NO_DATA = "No data available for this request."


class ScenarioToolError(Exception):
    """Kept for API compatibility; run() no longer raises it for an unscripted tool."""


class ScenarioToolExecutor:
    def __init__(self, tool_script: Mapping[str, str]) -> None:
        self._tool_script = dict(tool_script)

    def run(self, tool_name: str, arguments: Mapping[str, object]) -> str:
        # Any tool reaching here is already a declared workflow tool (the dispatcher rejects
        # undeclared tools earlier). A declared-but-unscripted call - e.g. a live model taking
        # an off-branch path - is a navigation choice we want to measure, not a crash, so return
        # a neutral deterministic default and let the conversation continue.
        return self._tool_script.get(tool_name, NO_DATA)
