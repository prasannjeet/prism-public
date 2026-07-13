"""The canonical Plan C full-run grid: seed split + the exact expected cell count."""

from __future__ import annotations

from datetime import date

from bench.full_run import (
    LARGE_SEEDS,
    SMALL_SEEDS,
    build_run_plans,
    expected_cells,
    is_large_workflow,
    seeds_for,
)
from bench.scenario import load_scenarios


def _scenarios():
    return load_scenarios("scenarios")


def test_seed_split_small_vs_large() -> None:
    assert SMALL_SEEDS == 3
    assert LARGE_SEEDS == 2
    assert seeds_for("form_booking") == 3
    assert seeds_for("travel_insurance_claim") == 3  # T4 is small
    assert seeds_for("incident_response") == 2  # T5 is large
    assert seeds_for("clinical_trial_eligibility") == 2  # T6 is large
    assert is_large_workflow("clinical_trial_eligibility") is True
    assert is_large_workflow("form_booking") is False


def test_build_run_plans_splits_into_two_seed_blocks() -> None:
    plans = build_run_plans(_scenarios(), run_id="x", today=date(2026, 1, 15), budget_usd=300.0)
    by_seeds = {p.seeds: p for p in plans}
    assert set(by_seeds) == {3, 2}
    small, large = by_seeds[3], by_seeds[2]
    assert all(not is_large_workflow(s.workflow) for s in small.scenarios)
    assert all(is_large_workflow(s.workflow) for s in large.scenarios)
    # Every scenario is covered exactly once across the two blocks.
    assert len(small.scenarios) == 57
    assert len(large.scenarios) == 31
    assert len(small.scenarios) + len(large.scenarios) == 88


def test_expected_cell_count_pins_plan_c() -> None:
    # 3 cheap models x 5 configs x (57x3 + 31x2) + Sonnet C1/C5 x (57x3 + 31x2).
    # cheap = 3*(57*3 + 31*2)*5 = 3*(171+62)*5 = 3*233*5 = 3495; sonnet = (171+62)*2 = 466.
    plans = build_run_plans(_scenarios(), run_id="x", today=date(2026, 1, 15), budget_usd=300.0)
    assert len(list(expected_cells(plans))) == 3961


def test_build_run_plans_threads_continuation_cap() -> None:
    scenarios = _scenarios()
    default = build_run_plans(scenarios, run_id="x", today=date(2026, 1, 15), budget_usd=300.0)
    assert all(p.max_continuation_turns == 0 for p in default)
    capped = build_run_plans(
        scenarios, run_id="x", today=date(2026, 1, 15), budget_usd=300.0,
        max_continuation_turns=5,
    )
    assert capped  # non-empty
    assert all(p.max_continuation_turns == 5 for p in capped)
