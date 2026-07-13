"""Launch the full Plan C run as one resumable, budget-capped background command.

Runs the seed-blocks in order (3-seed small block, then 2-seed large block), sharing one
results/<run_id>/ dir. The dollar cap is GLOBAL: each block's budget is the remainder after
prior blocks' spend. A billing pause (or budget stop) in any block halts the remaining blocks;
the run is fully resumable, so re-launching the same command continues after a top-up."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

from bench.full_run import build_run_plans
from bench.prices import PRICES, ModelPrice
from bench.runner import (
    CellKey,
    RunPlan,
    RunSummary,
    _live_build_adapter,
    live_models,
    run_grid,
    select_models,
)
from bench.scenario import load_scenarios
from prism.agent import Config, load_configs


def launch_full_run(
    run_dir: Path,
    plans: Sequence[RunPlan],
    *,
    configs: Mapping[str, Config],
    build_adapter: Callable[[CellKey, RunPlan], object],
    budget_usd: float,
    estimate_usd: float,
    prices: Mapping[str, ModelPrice] = PRICES,
    require_prices: bool = True,
) -> list[RunSummary]:
    """Run each seed-block in order; stop the whole run on the first budget/billing pause."""
    summaries: list[RunSummary] = []
    spent = 0.0
    for plan in plans:
        block = replace(plan, budget_usd=budget_usd - spent)
        summary = run_grid(
            block,
            run_dir,
            configs=configs,
            build_adapter=build_adapter,
            prices=prices,
            estimate_usd=estimate_usd,
            require_prices=require_prices,
        )
        summaries.append(summary)
        spent += summary.spent_usd
        if summary.stopped_on_billing or summary.stopped_on_budget or summary.stopped_on_infra:
            break
    return summaries


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the full Plan C run (live, resumable).")
    parser.add_argument("run_id")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--workflows-dir", default="workflows")
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--estimate", type=float, default=0.30, help="per-cell USD estimate")
    parser.add_argument("--max-continuation-turns", type=int, default=0,
                        help="extra uniform nudge turns after the scripted turns run out (0 = off)")
    parser.add_argument("--models", default=None,
                        help="comma-separated model names (default: the full locked lineup)")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios_dir, workflows_dir=Path(args.workflows_dir))
    model_names = args.models.split(",") if args.models else None
    lineup = select_models(live_models(), model_names)
    plans = build_run_plans(
        scenarios, run_id=args.run_id, today=date(2026, 1, 15), budget_usd=args.budget,
        models=lineup, max_continuation_turns=args.max_continuation_turns,
    )
    configs = load_configs(Path(args.prompts_dir))
    summaries = launch_full_run(
        Path(args.results_dir) / args.run_id,
        plans,
        configs=configs,
        build_adapter=_live_build_adapter,
        budget_usd=args.budget,
        estimate_usd=args.estimate,
    )

    completed = sum(s.completed for s in summaries)
    skipped = sum(s.skipped for s in summaries)
    failed = sum(s.failed for s in summaries)
    spent = sum(s.spent_usd for s in summaries)
    stopped_billing = any(s.stopped_on_billing for s in summaries)
    stopped_budget = any(s.stopped_on_budget for s in summaries)
    stopped_infra = any(s.stopped_on_infra for s in summaries)
    print(  # noqa: T201 - CLI entrypoint
        f"completed={completed} skipped={skipped} failed={failed} spent=${spent:.2f} "
        f"stopped_on_budget={stopped_budget} stopped_on_billing={stopped_billing} "
        f"stopped_on_infra={stopped_infra}"
    )
    if stopped_billing or stopped_infra:
        why = "a provider billing/quota rejection" if stopped_billing else (
            "repeated infra failures (provider likely down or out of credit)"
        )
        print(  # noqa: T201
            f"[run_full] paused on {why}. Resolve it, then re-run the SAME command: "
            "completed cells are skipped, failed cells are retried, and the run continues."
        )


if __name__ == "__main__":
    main()
