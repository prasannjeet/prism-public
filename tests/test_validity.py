"""Unit test for the correction-before-completion scenario validity guard."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from bench.scenario import Beat, Expected, Part, Scenario, load_scenarios
from bench.validity import correction_lands_before_completion
from prism.engine import StateEngine
from prism.schema import parse_workflow
from prism.validators import load_validated_workflow


def _engine(workflow: str) -> StateEngine:
    return StateEngine(load_validated_workflow(Path("workflows") / f"{workflow}.yaml"))


def test_every_correction_branch_switch_scenario_corrects_before_completion() -> None:
    scenarios = load_scenarios("scenarios", workflows_dir=Path("workflows"))
    targets = [s for s in scenarios if s.category == "correction_branch_switch"]
    assert targets, "expected correction_branch_switch scenarios"
    for s in targets:
        assert correction_lands_before_completion(s, _engine(s.workflow)), (
            f"{s.id}: workflow is already complete when the correction arrives"
        )


_TWO_FIELD = {
    "workflow": "tiny",
    "profile": "express",
    "fields": [
        {"name": "a", "type": "text", "required": True},
        {"name": "b", "type": "text", "required": True},
    ],
}


def _scenario(beats: tuple[Beat, ...]) -> Scenario:
    return Scenario(
        id="synthetic",
        workflow="tiny",
        category="correction_branch_switch",
        beats=beats,
        expected=Expected(final_state=MappingProxyType({}), valid_completion=("committed",)),
    )


def test_correction_after_completion_is_detected_as_invalid() -> None:
    engine = StateEngine(parse_workflow(_TWO_FIELD))
    # Both mandatory fields are provided before the correction: the workflow is already
    # complete, so the guard must reject it.
    beats = (
        Beat(parts=(Part(say="a", field="a", value="x"), Part(say="b", field="b", value="y"))),
        Beat(intent="correct", parts=(Part(say="fix a", field="a", value="z"),)),
    )
    assert correction_lands_before_completion(_scenario(beats), engine) is False


def test_correction_before_completion_passes() -> None:
    engine = StateEngine(parse_workflow(_TWO_FIELD))
    # Field b is withheld until after the correction: the workflow is still incomplete.
    beats = (
        Beat(parts=(Part(say="a", field="a", value="x"),)),
        Beat(intent="correct", parts=(Part(say="fix a", field="a", value="z"),)),
        Beat(parts=(Part(say="b", field="b", value="y"),)),
    )
    assert correction_lands_before_completion(_scenario(beats), engine) is True
