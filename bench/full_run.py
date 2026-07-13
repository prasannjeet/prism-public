"""The canonical Plan C full-run grid: the seed-blocked RunPlans the live run executes and the
status tool measures against. One source of truth so the launch and the progress percent agree.

Seed plan: 3 seeds on the small/mid workflows
(W1-W3 + T4), 2 seeds on the two large/deep workflows (T5, T6). The model lineup's per-model
allowed_configs already restrict Sonnet to its C1/C5 subset, so each block carries every model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date

from bench.runner import CellKey, ModelSpec, RunPlan, enumerate_cells, live_models
from bench.scenario import Scenario

# Plan C seed levers (hard-workflows-budget.md, "Recommended plan + levers").
SMALL_SEEDS = 3
LARGE_SEEDS = 2

# The two large (deep) workflows get fewer seeds; every other workflow is "small".
LARGE_WORKFLOWS: frozenset[str] = frozenset({"incident_response", "clinical_trial_eligibility"})

_ALL_CONFIGS = ("C1", "C2", "C3", "C4", "C5")


def is_large_workflow(workflow: str) -> bool:
    return workflow in LARGE_WORKFLOWS


def seeds_for(workflow: str) -> int:
    return LARGE_SEEDS if is_large_workflow(workflow) else SMALL_SEEDS


def build_run_plans(
    scenarios: Sequence[Scenario],
    *,
    run_id: str,
    today: date,
    budget_usd: float,
    models: tuple[ModelSpec, ...] | None = None,
    max_continuation_turns: int = 0,
) -> list[RunPlan]:
    """The seed-blocked plans (small block seeds=3, large block seeds=2) sharing one run_id.

    Returns one plan per non-empty block; enumerate_cells filters Sonnet to its C1/C5 subset."""
    lineup = models if models is not None else live_models()
    small = tuple(s for s in scenarios if not is_large_workflow(s.workflow))
    large = tuple(s for s in scenarios if is_large_workflow(s.workflow))
    plans: list[RunPlan] = []
    for block, seeds in ((small, SMALL_SEEDS), (large, LARGE_SEEDS)):
        if block:
            plans.append(
                RunPlan(
                    run_id, lineup, block, _ALL_CONFIGS, seeds, today, budget_usd,
                    max_continuation_turns,
                )
            )
    return plans


def expected_cells(plans: Sequence[RunPlan]) -> Iterator[CellKey]:
    """Every grid cell across the seed-blocks (the status-percent denominator)."""
    for plan in plans:
        yield from enumerate_cells(plan)
