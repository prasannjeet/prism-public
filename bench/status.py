"""Read-only progress snapshot for a live run directory. Zero model calls: counts the on-disk
'done' markers against the canonical expected grid and sums spend from the usage sidecars.

Safe to run at any time against a run that is still in progress: it only reads files, so it
never touches or disturbs the running job."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bench.full_run import build_run_plans, expected_cells
from bench.prices import PRICES, ModelPrice, cost_usd
from bench.runner import CellKey, RunPlan
from bench.scenario import load_scenarios

# (done, total) counts per dimension value.
Breakdown = dict[str, tuple[int, int]]


@dataclass(frozen=True)
class StatusReport:
    total: int
    completed: int
    failed: int
    pending: int
    spent_usd: float
    percent: float
    by_model: Breakdown
    by_config: Breakdown
    by_workflow: Breakdown


def _cell_status(run_dir: Path, cell: CellKey) -> str | None:
    """'completed' / 'failed_infra' / other from the marker, or None if the cell hasn't run."""
    marker = cell.path(run_dir) / "meta.json"
    if not marker.exists():
        return None
    return str(json.loads(marker.read_text(encoding="utf-8")).get("status") or "")


def _cell_cost(run_dir: Path, cell: CellKey, prices: Mapping[str, ModelPrice]) -> float:
    price = prices.get(cell.model)
    if price is None:
        return 0.0
    usage = json.loads((cell.path(run_dir) / "usage.json").read_text(encoding="utf-8"))
    return cost_usd(dict(usage.get("totals", {})), price)


def summarize(
    run_dir: Path,
    plans: Sequence[RunPlan],
    *,
    prices: Mapping[str, ModelPrice] = PRICES,
) -> StatusReport:
    by_model: Breakdown = {}
    by_config: Breakdown = {}
    by_workflow: Breakdown = {}

    def _bump(table: Breakdown, key: str, done: bool) -> None:
        d, t = table.get(key, (0, 0))
        table[key] = (d + (1 if done else 0), t + 1)

    completed = failed = total = 0
    spent = 0.0
    for cell in expected_cells(plans):
        total += 1
        status = _cell_status(run_dir, cell)
        is_done = status == "completed"
        if is_done:
            completed += 1
            spent += _cell_cost(run_dir, cell, prices)
        elif status == "failed_infra":
            failed += 1
        _bump(by_model, cell.model, is_done)
        _bump(by_config, cell.config, is_done)
        _bump(by_workflow, cell.workflow, is_done)

    pending = total - completed - failed
    percent = (completed / total * 100.0) if total else 0.0
    return StatusReport(
        total=total,
        completed=completed,
        failed=failed,
        pending=pending,
        spent_usd=spent,
        percent=percent,
        by_model=by_model,
        by_config=by_config,
        by_workflow=by_workflow,
    )


def _eta_seconds(run_dir: Path, completed: int, pending: int) -> float | None:
    """Best-effort ETA from marker mtimes (wall-clock spread of completed cells / rate).

    Not part of summarize() so the core stays deterministic; None when it can't be estimated."""
    if completed < 2 or pending == 0:
        return None
    mtimes = [p.stat().st_mtime for p in run_dir.rglob("meta.json")]
    if len(mtimes) < 2:
        return None
    elapsed = max(mtimes) - min(mtimes)
    if elapsed <= 0:
        return None
    rate = completed / elapsed  # cells per second
    return pending / rate if rate else None


def _bar(done: int, total: int, width: int = 24) -> str:
    filled = round(width * done / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_breakdown(title: str, table: Breakdown) -> str:
    rows = [f"  {k:<28} {d:>4}/{t:<4} ({d / t * 100:4.0f}%)" for k, (d, t) in sorted(table.items())]
    return f"{title}:\n" + "\n".join(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PRISM run progress (read-only).")
    parser.add_argument("run_dir", type=Path, help="results/<run_id>")
    parser.add_argument("--scenarios", default="scenarios")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    # run_id/today/budget don't affect the cell enumeration (the status denominator).
    plans = build_run_plans(
        scenarios, run_id="(measured)", today=date(2026, 1, 15), budget_usd=300.0
    )
    report = summarize(args.run_dir, plans)

    print(  # noqa: T201 - CLI entrypoint
        f"{_bar(report.completed, report.total)} {report.percent:5.1f}%  "
        f"{report.completed}/{report.total} done  "
        f"({report.pending} pending, {report.failed} infra-failed)"
    )
    print(f"spent: ${report.spent_usd:.2f}")  # noqa: T201
    eta = _eta_seconds(args.run_dir, report.completed, report.pending)
    if eta is not None:
        print(f"eta:   ~{eta / 3600:.1f}h at the current pace (approximate)")  # noqa: T201
    print(_format_breakdown("by model", report.by_model))  # noqa: T201
    print(_format_breakdown("by config", report.by_config))  # noqa: T201
    print(_format_breakdown("by workflow", report.by_workflow))  # noqa: T201


if __name__ == "__main__":
    main()
