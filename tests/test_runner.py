"""Batch runner: grid enumeration, files-as-ledger resume, serial archived run."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from bench.engine_cache import build_engine
from bench.oracle import OracleAdapter
from bench.prices import UnknownModelPriceError
from bench.run import CONTINUATION_MESSAGE
from bench.runner import (
    CellKey,
    ModelSpec,
    RunPlan,
    archive_cell,
    enumerate_cells,
    is_done,
    live_models,
    run_grid,
    select_models,
    select_scenarios,
)
from bench.scenario import load_scenarios
from prism.agent import CONFIGS
from prism.models import ModelResponse, ModelSettings, Usage


def _two_scenarios():
    # Two scenarios from one workflow keeps the test small and deterministic.
    scenarios = load_scenarios("scenarios")
    picked = [s for s in scenarios if s.workflow == "form_booking"][:2]
    assert len(picked) == 2
    return tuple(picked)


def _plan(scenarios) -> RunPlan:
    cheap = ModelSpec(
        name="oracle-cheap",
        adapter_factory=lambda: OracleAdapter,  # placeholder; replaced per-cell in D4
        allowed_configs=("C1", "C2", "C3", "C4", "C5"),
    )
    subset = ModelSpec(
        name="oracle-subset",
        adapter_factory=lambda: OracleAdapter,
        allowed_configs=("C1", "C5"),
    )
    return RunPlan(
        run_id="t",
        models=(cheap, subset),
        scenarios=scenarios,
        configs=("C1", "C2", "C3", "C4", "C5"),
        seeds=2,
        today=date(2026, 1, 15),
        budget_usd=1000.0,
    )


def test_enumerate_filters_by_allowed_configs_and_counts() -> None:
    scenarios = _two_scenarios()
    cells = list(enumerate_cells(_plan(scenarios)))
    # cheap: 2 scenarios * 5 configs * 2 seeds = 20; subset: 2 * 2 * 2 = 8.
    assert len(cells) == 28
    subset_cells = [c for c in cells if c.model == "oracle-subset"]
    assert {c.config for c in subset_cells} == {"C1", "C5"}


def test_enumerate_orders_configs_by_priority() -> None:
    scenarios = _two_scenarios()
    cheap_cells = [c for c in enumerate_cells(_plan(scenarios)) if c.model == "oracle-cheap"]
    # First config emitted is the headline C5, last is C3 (headline-first priority).
    assert cheap_cells[0].config == "C5"
    seen_order = [c.config for c in cheap_cells]
    assert seen_order.index("C5") < seen_order.index("C3")


def test_cell_key_as_dict() -> None:
    cell = CellKey("anthropic:claude-haiku-4-5", "form_booking", "w1-happy-01", "C5", 0)
    assert cell.as_dict() == {
        "model": "anthropic:claude-haiku-4-5",
        "workflow": "form_booking",
        "scenario": "w1-happy-01",
        "config": "C5",
        "seed": 0,
    }


def test_cell_key_path_sanitizes_model_slug() -> None:
    cell = CellKey("anthropic:claude-haiku-4-5", "form_booking", "w1-happy-01", "C5", 2)
    path = cell.path(Path("results/run-1"))
    assert path == Path(
        "results/run-1/anthropic_claude-haiku-4-5/form_booking/w1-happy-01/C5/seed-2"
    )


def test_archive_writes_three_files_marker_last_and_is_done(tmp_path) -> None:
    cell = CellKey("oracle-cheap", "form_booking", "w1-happy-01", "C5", 0)
    event_log = ({"type": "user_message", "turn": 1, "text": "hi"},)
    usage_sidecar = {"cell": cell.as_dict(), "calls": [], "totals": {}}
    meta = {"cell": cell.as_dict(), "status": "completed", "attempts": 1}

    assert is_done(tmp_path, cell) is False
    archive_cell(tmp_path, cell, event_log, usage_sidecar, meta)

    cell_dir = cell.path(tmp_path)
    assert (cell_dir / "events.jsonl").exists()
    assert (cell_dir / "usage.json").exists()
    assert (cell_dir / "meta.json").exists()
    # events.jsonl is one JSON object per line.
    lines = (cell_dir / "events.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[0])["type"] == "user_message"
    assert is_done(tmp_path, cell) is True


def test_is_done_false_when_marker_missing_even_if_events_present(tmp_path) -> None:
    cell = CellKey("oracle-cheap", "form_booking", "w1-happy-01", "C5", 0)
    cell_dir = cell.path(tmp_path)
    cell_dir.mkdir(parents=True)
    (cell_dir / "events.jsonl").write_text('{"type": "user_message"}\n')
    assert is_done(tmp_path, cell) is False  # half-written cell -> not done


def test_is_done_false_when_status_not_completed(tmp_path) -> None:
    cell = CellKey("oracle-cheap", "form_booking", "w1-happy-01", "C5", 0)
    archive_cell(tmp_path, cell, (), {}, {"status": "failed_infra", "attempts": 3})
    assert is_done(tmp_path, cell) is False


def _scenarios_by_id(scenarios):
    return {s.id: s for s in scenarios}


def _oracle_build_adapter(cell, plan):
    # Build a fresh OracleAdapter for the cell's scenario (deterministic, zero API).
    by_id = _scenarios_by_id(plan.scenarios)
    scenario = by_id[cell.scenario]
    engine = build_engine(scenario.workflow)
    return OracleAdapter(engine, scenario)


def _small_plan():
    scenarios = tuple(s for s in load_scenarios("scenarios") if s.workflow == "form_booking")[:2]
    cheap = ModelSpec("oracle-cheap", adapter_factory=lambda: None, allowed_configs=("C1", "C5"))
    return RunPlan(
        run_id="t",
        models=(cheap,),
        scenarios=scenarios,
        configs=("C1", "C5"),
        seeds=2,
        today=date(2026, 1, 15),
        budget_usd=1000.0,
    )


def test_run_grid_completes_every_cell_and_writes_manifest(tmp_path) -> None:
    plan = _small_plan()
    summary = run_grid(
        plan,
        tmp_path,
        configs=CONFIGS,
        build_adapter=_oracle_build_adapter,
        estimate_usd=0.0,
    )
    # 2 scenarios * 2 configs * 2 seeds = 8 cells.
    assert summary.completed == 8
    assert summary.skipped == 0
    for cell in enumerate_cells(plan):
        assert is_done(tmp_path, cell) is True
    assert (tmp_path / "manifest.json").exists()


def test_run_grid_resumes_and_skips_done_cells(tmp_path) -> None:
    plan = _small_plan()
    run_grid(plan, tmp_path, configs=CONFIGS, build_adapter=_oracle_build_adapter, estimate_usd=0.0)
    # Second run: everything already done -> all skipped, nothing re-executed.
    summary = run_grid(
        plan, tmp_path, configs=CONFIGS, build_adapter=_oracle_build_adapter, estimate_usd=0.0
    )
    assert summary.completed == 0
    assert summary.skipped == 8


def test_run_grid_stops_when_budget_would_be_exceeded(tmp_path) -> None:
    plan = _small_plan()
    # estimate per cell 10.0, budget 5.0 -> the very first cell would exceed -> stop immediately.
    plan_low = RunPlan(**{**plan.__dict__, "budget_usd": 5.0})
    summary = run_grid(
        plan_low, tmp_path, configs=CONFIGS, build_adapter=_oracle_build_adapter, estimate_usd=10.0
    )
    assert summary.completed == 0
    assert summary.stopped_on_budget is True


class _InfraExc(Exception):
    status_code = 503


def test_run_grid_records_failed_infra_without_marking_done(tmp_path) -> None:
    plan = _small_plan()

    def _explode(cell, _plan):
        class _Boom:
            name = "oracle-cheap"
            usages: list = []  # noqa: RUF012

            def complete(self, *a, **k):  # type: ignore[no-untyped-def]
                raise _InfraExc()

        return _Boom()

    # Disable the circuit breaker here so we exercise the all-cells-recorded path.
    summary = run_grid(
        plan,
        tmp_path,
        configs=CONFIGS,
        build_adapter=_explode,
        estimate_usd=0.0,
        max_consecutive_infra=99,
    )
    assert summary.completed == 0
    assert summary.failed == 8
    # Failed-infra cells are NOT "done" -> a later resume will retry them.
    for cell in enumerate_cells(plan):
        assert is_done(tmp_path, cell) is False
        meta = json.loads((cell.path(tmp_path) / "meta.json").read_text())
        assert meta["status"] == "failed_infra"


def test_run_grid_circuit_breaks_after_consecutive_infra_failures(tmp_path) -> None:
    plan = _small_plan()  # 8 cells

    def _always_infra(cell, _plan):
        class _Boom:
            name = "oracle-cheap"
            usages: list = []  # noqa: RUF012

            def complete(self, *a, **k):  # type: ignore[no-untyped-def]
                raise _InfraExc()

        return _Boom()

    summary = run_grid(
        plan,
        tmp_path,
        configs=CONFIGS,
        build_adapter=_always_infra,
        estimate_usd=0.0,
        max_consecutive_infra=3,
    )
    # Stops cleanly after 3 in a row rather than churning all 8.
    assert summary.stopped_on_infra is True
    assert summary.stopped_on_budget is False
    assert summary.failed == 3
    assert summary.completed == 0
    # Only the 3 attempted cells were archived; the rest are left pending for resume.
    assert len(list(tmp_path.rglob("meta.json"))) == 3


class _BillingExc(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__("You exceeded your current quota; check your plan and billing details.")


def test_run_grid_stops_gracefully_on_billing_error_and_is_resumable(tmp_path) -> None:
    plan = _small_plan()

    def _out_of_credits(cell, _plan):
        class _Broke:
            name = "oracle-cheap"
            usages: list = []  # noqa: RUF012

            def complete(self, *a, **k):  # type: ignore[no-untyped-def]
                raise _BillingExc()

        return _Broke()

    summary = run_grid(
        plan, tmp_path, configs=CONFIGS, build_adapter=_out_of_credits, estimate_usd=0.0
    )
    # Graceful stop on the very first cell: no completions, no infra-failure rows written.
    assert summary.stopped_on_billing is True
    assert summary.stopped_on_budget is False
    assert summary.completed == 0
    assert summary.failed == 0
    # The cell that hit the billing wall is left un-stamped -> a later resume retries it,
    # and no partial/failed marker is written for it.
    for cell in enumerate_cells(plan):
        assert is_done(tmp_path, cell) is False
        assert not (cell.path(tmp_path) / "meta.json").exists()


def test_run_grid_require_prices_raises_for_unpriced_model(tmp_path) -> None:
    plan = _small_plan()  # model name "oracle-cheap" is absent from PRICES
    try:
        run_grid(
            plan,
            tmp_path,
            configs=CONFIGS,
            build_adapter=_oracle_build_adapter,
            estimate_usd=0.0,
            require_prices=True,
        )
        raise AssertionError("expected UnknownModelPriceError")
    except UnknownModelPriceError as e:
        assert "oracle-cheap" in str(e)


def test_run_grid_default_does_not_require_prices(tmp_path) -> None:
    # default require_prices=False -> the oracle path runs untouched
    plan = _small_plan()
    summary = run_grid(
        plan, tmp_path, configs=CONFIGS, build_adapter=_oracle_build_adapter, estimate_usd=0.0
    )
    assert summary.completed == 8


def test_select_models_none_returns_all() -> None:
    models = live_models()
    assert select_models(models, None) == models
    assert select_models(models, []) == models


def test_select_models_subset_preserves_lineup_order() -> None:
    models = live_models()
    # request in a different order than the lineup; result keeps lineup order.
    picked = select_models(models, ["gpt-5.4-mini", "claude-haiku-4-5"])
    assert tuple(m.name for m in picked) == ("claude-haiku-4-5", "gpt-5.4-mini")


def test_select_models_single_pilot_model() -> None:
    picked = select_models(live_models(), ["claude-haiku-4-5"])
    assert len(picked) == 1
    assert picked[0].name == "claude-haiku-4-5"


def test_select_models_unknown_name_raises() -> None:
    try:
        select_models(live_models(), ["claude-haiku-4-5", "no-such-model"])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "no-such-model" in str(e)
        # available names are listed to help the caller
        assert "claude-haiku-4-5" in str(e)


def test_select_scenarios_none_returns_all() -> None:
    scenarios = tuple(load_scenarios("scenarios"))
    assert select_scenarios(scenarios, None) == scenarios
    assert select_scenarios(scenarios, []) == scenarios


def test_select_scenarios_subset_preserves_order() -> None:
    scenarios = tuple(load_scenarios("scenarios"))
    ids = ["w1-happy-windows", "w1-multi-answer-moveout"]
    picked = select_scenarios(scenarios, ids)
    # preserves the order scenarios were loaded in, not the request order.
    expected = tuple(s.id for s in scenarios if s.id in set(ids))
    assert tuple(s.id for s in picked) == expected
    assert set(s.id for s in picked) == set(ids)


def test_select_scenarios_single_id() -> None:
    scenarios = tuple(load_scenarios("scenarios"))
    picked = select_scenarios(scenarios, ["w1-happy-windows"])
    assert len(picked) == 1
    assert picked[0].id == "w1-happy-windows"


def test_select_scenarios_unknown_id_raises() -> None:
    scenarios = tuple(load_scenarios("scenarios"))
    try:
        select_scenarios(scenarios, ["w1-happy-windows", "no-such-scenario"])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "no-such-scenario" in str(e)


def test_run_plan_max_continuation_turns_defaults_to_zero() -> None:
    plan = _small_plan()
    assert plan.max_continuation_turns == 0


class _NeverTerminates:
    name = "oracle-cheap"
    usages: list = []  # noqa: RUF012
    settings = ModelSettings()

    def complete(self, *a, **k):  # type: ignore[no-untyped-def]
        return ModelResponse("ok", (), Usage(0, 0, 0, 0, {}), "end_turn")


def test_run_grid_threads_continuation_cap_into_archived_events(tmp_path) -> None:
    base = _small_plan()
    plan = RunPlan(**{**base.__dict__, "max_continuation_turns": 3})

    def _never(cell, _plan):
        return _NeverTerminates()

    summary = run_grid(
        plan, tmp_path, configs=CONFIGS, build_adapter=_never, estimate_usd=0.0
    )
    assert summary.completed == 8  # 2 scenarios * 2 configs * 2 seeds
    # Every archived cell carries exactly 3 continuation user turns.
    cell = next(enumerate_cells(plan))
    events = [
        json.loads(line)
        for line in (cell.path(tmp_path) / "events.jsonl").read_text().splitlines()
    ]
    nudges = [
        e
        for e in events
        if e.get("type") == "user_message" and e.get("text") == CONTINUATION_MESSAGE
    ]
    assert len(nudges) == 3
