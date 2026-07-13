"""Resumable, cost-metered, serial batch runner over the experiment grid.

Wraps bench.run.run_scenario unchanged. Files-as-ledger resume: a cell is "done" iff
its meta.json marker (status=completed) exists, written LAST after events.jsonl +
usage.json. prism/ is untouched; cost/usage/retry live in bench/ wrappers."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bench.prices import (
    PRICES,
    CostMeter,
    ModelPrice,
    UnknownModelPriceError,
    cost_usd,
    price_table_dict,
)
from bench.resilience import RetryingAdapter, is_billing_error, is_infra_error
from bench.run import run_scenario
from bench.scenario import Scenario
from bench.usage import UsageRecorder, usage_call_dict, usage_totals
from prism.agent import Config, load_configs
from prism.models import AnthropicAdapter, GeminiAdapter, OpenAIAdapter, Usage


def _slug(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


@dataclass(frozen=True)
class CellKey:
    model: str
    workflow: str
    scenario: str
    config: str
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "workflow": self.workflow,
            "scenario": self.scenario,
            "config": self.config,
            "seed": self.seed,
        }

    def path(self, run_dir: Path) -> Path:
        return (
            run_dir
            / _slug(self.model)
            / self.workflow
            / self.scenario
            / self.config
            / f"seed-{self.seed}"
        )


# Config run priority: headline comparison first so a mid-run budget
# stop still yields C5-vs-C1; C3 (the expensive full-reinjection arm) last.
_CONFIG_PRIORITY = ("C5", "C1", "C2", "C4", "C3")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter_factory: Callable[[], object]
    allowed_configs: tuple[str, ...]


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    models: tuple[ModelSpec, ...]
    scenarios: tuple[Scenario, ...]
    configs: tuple[str, ...]
    seeds: int
    today: date
    budget_usd: float
    max_continuation_turns: int = 0


def _ordered_configs(configs: Sequence[str]) -> list[str]:
    known = [c for c in _CONFIG_PRIORITY if c in configs]
    extra = [c for c in configs if c not in _CONFIG_PRIORITY]
    return known + extra


def enumerate_cells(plan: RunPlan) -> Iterator[CellKey]:
    """Every grid cell, filtered by each model's allowed_configs, configs in priority order."""
    for model in plan.models:
        for config in _ordered_configs(plan.configs):
            if config not in model.allowed_configs:
                continue
            for scenario in plan.scenarios:
                for seed in range(plan.seeds):
                    yield CellKey(model.name, scenario.workflow, scenario.id, config, seed)


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def archive_cell(
    run_dir: Path,
    cell: CellKey,
    event_log: Sequence[dict[str, object]],
    usage_sidecar: dict[str, object],
    meta: dict[str, object],
) -> None:
    """Write events.jsonl, then usage.json, then meta.json LAST (the commit marker)."""
    cell_dir = cell.path(run_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(cell_dir / "events.jsonl", event_log)
    _write_json(cell_dir / "usage.json", usage_sidecar)
    _write_json(cell_dir / "meta.json", meta)  # last == commit


def is_done(run_dir: Path, cell: CellKey) -> bool:
    marker = cell.path(run_dir) / "meta.json"
    if not marker.exists():
        return False
    meta = json.loads(marker.read_text(encoding="utf-8"))
    return bool(meta.get("status") == "completed")


class BillingStop(Exception):
    """Provider rejected a call for billing/quota reasons. The grid stops gracefully and the
    offending cell is left un-stamped, so a resume (after topping up) retries it cleanly."""


@dataclass(frozen=True)
class RunSummary:
    completed: int
    skipped: int
    failed: int
    spent_usd: float
    stopped_on_budget: bool
    stopped_on_billing: bool = False
    stopped_on_infra: bool = False


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_manifest(run_dir: Path, plan: RunPlan, prices: Mapping[str, ModelPrice]) -> None:
    manifest = {
        "run_id": plan.run_id,
        "git_commit": _git_commit(),
        "today": plan.today.isoformat(),
        "seeds": plan.seeds,
        # Variance is estimated by repetition (seeds), not by pinning sampling: every
        # provider runs at its default temperature/top-p with no seed.
        "sampling": {
            "temperature": "provider default",
            "top_p": "provider default",
            "seed": None,
            "note": "Anthropic output capped at 2048 tokens; others at provider defaults",
        },
        "budget_usd": plan.budget_usd,
        "models": [
            {"name": m.name, "allowed_configs": list(m.allowed_configs)} for m in plan.models
        ],
        "scenario_count": len(plan.scenarios),
        "scenario_ids": sorted(s.id for s in plan.scenarios),
        "configs": list(plan.configs),
        "prices": price_table_dict(prices),
    }
    _write_json(run_dir / "manifest.json", manifest)


def run_grid(
    plan: RunPlan,
    run_dir: Path,
    *,
    configs: Mapping[str, Config],
    build_adapter: Callable[[CellKey, RunPlan], object],
    prices: Mapping[str, ModelPrice] = PRICES,
    estimate_usd: float = 0.0,
    max_cell_attempts: int = 3,
    require_prices: bool = False,
    max_consecutive_infra: int = 8,
) -> RunSummary:
    """Serial, resumable run over the grid. Files-as-ledger resume; cost-metered hard stop.

    Set require_prices=True for billed runs (the live CLI does) to fail fast if any model in the
    plan lacks a PRICES entry — otherwise it would silently meter as $0. The oracle/scripted test
    path leaves it False (its model names legitimately have no price).

    Circuit breaker: after max_consecutive_infra cells fail with infra errors in a row (a strong
    signal the provider is down or out of credit), stop cleanly (stopped_on_infra) instead of
    churning the rest of the grid into failed_infra. A success resets the counter; failed cells
    are left for retry on resume."""
    if require_prices:
        missing = sorted({m.name for m in plan.models if m.name not in prices})
        if missing:
            raise UnknownModelPriceError(
                f"models in the run plan have no PRICES entry: {', '.join(missing)}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, plan, prices)
    meter = CostMeter(plan.budget_usd)

    # Recover prior spend on resume (sum costs of already-completed cells).
    for cell in enumerate_cells(plan):
        if is_done(run_dir, cell):
            usage = json.loads((cell.path(run_dir) / "usage.json").read_text(encoding="utf-8"))
            price = prices.get(cell.model)
            if price is not None:
                meter.add(cost_usd(dict(usage.get("totals", {})), price))

    completed = skipped = failed = 0
    consecutive_infra = 0
    for cell in enumerate_cells(plan):
        if is_done(run_dir, cell):
            skipped += 1
            continue
        if meter.would_exceed(estimate_usd):
            return RunSummary(completed, skipped, failed, meter.spent, stopped_on_budget=True)
        try:
            outcome = _run_one_cell(
                plan, run_dir, cell, configs, build_adapter, prices, meter, max_cell_attempts
            )
        except BillingStop:
            # Out of credits: stop the whole grid cleanly. The cell is un-stamped (no marker
            # written), so resuming after a top-up retries it and keeps every prior cell's logs.
            return RunSummary(
                completed,
                skipped,
                failed,
                meter.spent,
                stopped_on_budget=False,
                stopped_on_billing=True,
            )
        if outcome:
            completed += 1
            consecutive_infra = 0
        else:
            failed += 1
            consecutive_infra += 1
            if consecutive_infra >= max_consecutive_infra:
                # Provider likely down/out-of-credit: stop instead of churning the rest into
                # failed_infra. The failed cells are un-completed, so resume retries them.
                return RunSummary(
                    completed,
                    skipped,
                    failed,
                    meter.spent,
                    stopped_on_budget=False,
                    stopped_on_infra=True,
                )
    return RunSummary(completed, skipped, failed, meter.spent, stopped_on_budget=False)


def _run_one_cell(
    plan: RunPlan,
    run_dir: Path,
    cell: CellKey,
    configs: Mapping[str, Config],
    build_adapter: Callable[[CellKey, RunPlan], object],
    prices: Mapping[str, ModelPrice],
    meter: CostMeter,
    max_cell_attempts: int,
) -> bool:
    scenario = next(s for s in plan.scenarios if s.id == cell.scenario)
    config = configs[cell.config]
    adapter = build_adapter(cell, plan)
    try:
        result = run_scenario(
            scenario,
            config,
            adapter,  # type: ignore[arg-type]
            today=plan.today,
            max_continuation_turns=plan.max_continuation_turns,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised below unless infra
        if is_billing_error(exc):
            # Don't archive: leave the cell un-stamped so a post-top-up resume retries it.
            raise BillingStop(str(exc)) from exc
        if not is_infra_error(exc):
            raise  # a real bug -> fail-fast
        archive_cell(
            run_dir,
            cell,
            (),
            {"cell": cell.as_dict(), "calls": [], "totals": {}},
            {
                "cell": cell.as_dict(),
                "status": "failed_infra",
                "attempts": max_cell_attempts,
                "error": str(exc),
            },
        )
        return False

    # usages/settings come off the UsageRecorder + ModelAdapter contract; defaults keep the
    # sidecar best-effort (oracle path -> empty -> zero cost).
    usages: list[Usage] = list(getattr(adapter, "usages", []))
    totals = usage_totals(usages)
    settings = getattr(adapter, "settings", None)
    sidecar: dict[str, object] = {
        "cell": cell.as_dict(),
        "model_id": getattr(adapter, "name", cell.model),
        "settings": dict(getattr(settings, "params", {})),
        "calls": [usage_call_dict(u) for u in usages],
        "n_calls": len(usages),
        "totals": totals,
    }
    meta = {
        "cell": cell.as_dict(),
        "status": "completed",
        "attempts": 1,
        "today": plan.today.isoformat(),
    }
    archive_cell(run_dir, cell, result.event_log, sidecar, meta)

    price = prices.get(cell.model)
    if price is not None:
        meter.add(cost_usd(totals, price))
    return True


# The locked model lineup. adapter_factory builds the BASE adapter; run_grid
# wraps it per cell. Real model ids double as price-table keys.
def live_models() -> tuple[ModelSpec, ...]:
    full = ("C5", "C1", "C2", "C4", "C3")
    return (
        ModelSpec("claude-haiku-4-5", lambda: AnthropicAdapter("claude-haiku-4-5"), full),
        ModelSpec("gpt-5.4-mini", lambda: OpenAIAdapter("gpt-5.4-mini"), full),
        ModelSpec("gemini-3.1-flash-lite", lambda: GeminiAdapter("gemini-3.1-flash-lite"), full),
        ModelSpec("claude-sonnet-4-6", lambda: AnthropicAdapter("claude-sonnet-4-6"), ("C1", "C5")),
    )


def select_models(models: tuple[ModelSpec, ...], names: list[str] | None) -> tuple[ModelSpec, ...]:
    """Filter the lineup to the requested model names, preserving lineup order.

    None/empty -> all models unchanged. Fail-fast on any unknown name
    so a typo can never silently run the wrong or empty set."""
    if not names:
        return models
    requested = set(names)
    available = {m.name for m in models}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            f"unknown model name(s): {', '.join(unknown)}; "
            f"available: {', '.join(m.name for m in models)}"
        )
    return tuple(m for m in models if m.name in requested)


def select_scenarios(
    scenarios: tuple[Scenario, ...], ids: list[str] | None
) -> tuple[Scenario, ...]:
    """Filter scenarios to the requested ids, preserving load order.

    None/empty -> all scenarios unchanged. Fail-fast on any unknown id."""
    if not ids:
        return scenarios
    requested = set(ids)
    available = {s.id for s in scenarios}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"unknown scenario id(s): {', '.join(unknown)}")
    return tuple(s for s in scenarios if s.id in requested)


def _live_build_adapter(cell: CellKey, plan: RunPlan) -> object:
    model = next(m for m in plan.models if m.name == cell.model)
    base = model.adapter_factory()
    return UsageRecorder(RetryingAdapter(base))  # type: ignore[arg-type]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PRISM batch grid runner (live).")
    parser.add_argument("run_id")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--workflows-dir", default="workflows")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--estimate", type=float, default=0.05, help="per-cell USD estimate")
    parser.add_argument("--configs", nargs="*", default=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--workflow", default=None, help="restrict to one workflow")
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model names (default: the full locked lineup)",
    )
    parser.add_argument(
        "--scenario-ids",
        default=None,
        help="comma-separated scenario ids (default: all, after the --workflow filter)",
    )
    args = parser.parse_args()

    from bench.scenario import load_scenarios

    model_names = args.models.split(",") if args.models else None
    scenario_ids = args.scenario_ids.split(",") if args.scenario_ids else None

    scenarios = tuple(
        s
        for s in load_scenarios(args.scenarios_dir, workflows_dir=Path(args.workflows_dir))
        if args.workflow is None or s.workflow == args.workflow
    )
    scenarios = select_scenarios(scenarios, scenario_ids)
    plan = RunPlan(
        run_id=args.run_id,
        models=select_models(live_models(), model_names),
        scenarios=scenarios,
        configs=tuple(args.configs),
        seeds=args.seeds,
        today=date(2026, 1, 15),
        budget_usd=args.budget,
    )
    configs = load_configs(Path(args.prompts_dir))
    summary = run_grid(
        plan,
        Path(args.results_dir) / args.run_id,
        configs=configs,
        build_adapter=_live_build_adapter,
        estimate_usd=args.estimate,
        require_prices=True,  # billed run: fail fast on any unpriced model
    )
    print(  # noqa: T201 - CLI entrypoint, not library code
        f"completed={summary.completed} skipped={summary.skipped} failed={summary.failed} "
        f"spent=${summary.spent_usd:.2f} stopped_on_budget={summary.stopped_on_budget} "
        f"stopped_on_billing={summary.stopped_on_billing} "
        f"stopped_on_infra={summary.stopped_on_infra}"
    )
    if summary.stopped_on_billing:
        print(  # noqa: T201
            "[runner] paused on a provider billing/quota rejection. Top up the account, then "
            "re-run the same command: completed cells are skipped and the run continues."
        )
    if summary.stopped_on_infra:
        print(  # noqa: T201
            "[runner] paused after repeated infra failures (provider likely down or out of "
            "credit). Re-run the same command: failed cells are retried on resume."
        )


if __name__ == "__main__":
    main()
