"""Single-cell driver: one scenario x one config x one adapter -> (event_log, final_state).
The reusable core both the oracle test-path and the live path call; the Phase 2.2 batch grid
wraps this with seeds, resume, rate-limiting, and archiving."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from prism.agent import Agent, Config
from prism.engine import State, StateEngine
from prism.events import MemoryEventSink
from prism.models import ModelAdapter
from prism.schema import load_workflow

from .cadence import pack
from .scenario import Scenario
from .scenario_tools import ScenarioToolExecutor

WORKFLOWS_DIR = Path("workflows")

# Turn-budget sensitivity re-run
# A generic, config-independent continuation message: identical for every config, contains no
# field values or workflow content. Sent after the scripted turns run out when no terminal action
# was taken, so a model that was one or two turns short can finish.
CONTINUATION_MESSAGE = (
    "Please go ahead. If you have everything you need, you can finish; "
    "if not, let me know what else you need from me."
)


@dataclass(frozen=True)
class RunResult:
    event_log: tuple[dict[str, object], ...]
    final_state: State


def run_scenario(
    scenario: Scenario,
    config: Config,
    adapter: ModelAdapter,
    *,
    today: date,
    workflows_dir: Path = WORKFLOWS_DIR,
    max_continuation_turns: int = 0,
) -> RunResult:
    engine = StateEngine(load_workflow(workflows_dir / f"{scenario.workflow}.yaml"))
    sink = MemoryEventSink()
    executor = ScenarioToolExecutor(scenario.tool_script)
    agent = Agent(engine, adapter, config, sink, executor, today=today)
    state = engine.initial_state()
    messages = pack(scenario, config.name)
    if not messages:
        # No scripted user turns (e.g. oracle-only fixtures): drive one empty turn so the
        # adapter can act.
        messages = [""]
    for turn, message in enumerate(messages, start=1):
        state = agent.run_turn(state, message, turn=turn)
    # Turn-budget sensitivity: after the scripted turns run out, nudge the agent to continue if it
    # has taken no terminal action. Uniform text and cap across configs. Default 0 -> no extra
    # turns -> byte-identical to the frozen run. The stop condition is the engine's terminal flags,
    # so an agent that already committed (clean OR admitted-violation) or escalated is not
    # re-prodded.
    continuations = 0
    while continuations < max_continuation_turns and not (state.submitted or state.escalated):
        turn = len(messages) + continuations + 1
        state = agent.run_turn(state, CONTINUATION_MESSAGE, turn=turn)
        continuations += 1
    return RunResult(event_log=tuple(sink.records), final_state=state)
