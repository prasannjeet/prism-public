"""Descriptive aggregation over a run archive. Zero model calls:
reads events.jsonl + usage.json per completed cell, scores via bench.evaluator.evaluate,
joins token totals + a derived dollar cost, and emits one master per-conversation table
plus the pass^k point estimate. Inferential stats (CIs/bootstrap) are Phase 3."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path

import pandas as pd

from bench.evaluator import evaluate
from bench.prices import PRICES, ModelPrice, cost_usd
from bench.scenario import WORKFLOWS_DIR, Scenario
from prism.engine import StateEngine
from prism.schema import load_workflow


@dataclass(frozen=True)
class CellRecord:
    model: str
    workflow: str
    scenario: str
    config: str
    seed: int
    task_success: bool
    task_success_refined: bool
    terminal: str
    branch: str | None
    contamination: int
    turns: int
    tool_calls: int
    tool_errors: int
    recovered: bool
    attempted: Mapping[str, int]
    admitted: Mapping[str, int]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int
    cost_usd: float


def _completed_cell_dirs(run_dir: Path) -> Iterator[Path]:
    for meta_path in sorted(run_dir.rglob("meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "completed":
            yield meta_path.parent


def _read_event_log(cell_dir: Path) -> list[dict[str, object]]:
    lines = (cell_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line]


def load_cell_records(
    run_dir: Path,
    scenarios_by_id: Mapping[str, Scenario],
    prices: Mapping[str, ModelPrice] = PRICES,
) -> Iterator[CellRecord]:
    engines: dict[str, StateEngine] = {}

    def _engine_for(workflow: str) -> StateEngine:
        if workflow not in engines:
            engines[workflow] = StateEngine(load_workflow(WORKFLOWS_DIR / f"{workflow}.yaml"))
        return engines[workflow]

    for cell_dir in _completed_cell_dirs(run_dir):
        usage = json.loads((cell_dir / "usage.json").read_text(encoding="utf-8"))
        cell = usage["cell"]
        scenario = scenarios_by_id[cell["scenario"]]
        record = evaluate(_read_event_log(cell_dir), scenario, _engine_for(scenario.workflow))
        totals = {str(k): int(v) for k, v in usage.get("totals", {}).items()}
        # Price key is the bare model name (cell["model"] == ModelSpec.name), the actual
        # PRICES key and the same key the runner's live cost meter uses -- NOT usage's
        # "model_id", which is the wrapped adapter name (e.g. "anthropic:claude-haiku-4-5").
        price = prices.get(str(cell["model"]))
        total_tokens = totals.get("input_tokens", 0) + totals.get("output_tokens", 0)
        yield CellRecord(
            model=str(cell["model"]),
            workflow=str(cell["workflow"]),
            scenario=str(cell["scenario"]),
            config=str(cell["config"]),
            seed=int(cell["seed"]),
            task_success=record.task_success,
            task_success_refined=record.task_success_refined,
            terminal=record.terminal,
            branch=record.branch,
            contamination=record.contamination,
            turns=record.turns,
            tool_calls=record.tool_calls,
            tool_errors=record.tool_errors,
            recovered=record.recovered,
            attempted=dict(record.attempted),
            admitted=dict(record.admitted),
            input_tokens=totals.get("input_tokens", 0),
            output_tokens=totals.get("output_tokens", 0),
            cache_read_tokens=totals.get("cache_read_tokens", 0),
            cache_write_tokens=totals.get("cache_write_tokens", 0),
            total_tokens=total_tokens,
            cost_usd=cost_usd(totals, price) if price is not None else 0.0,
        )


def _flat_row(record: CellRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "model": record.model,
        "workflow": record.workflow,
        "scenario": record.scenario,
        "config": record.config,
        "seed": record.seed,
        "task_success": record.task_success,
        "task_success_refined": record.task_success_refined,
        "terminal": record.terminal,
        "branch": record.branch,
        "contamination": record.contamination,
        "turns": record.turns,
        "tool_calls": record.tool_calls,
        "tool_errors": record.tool_errors,
        "recovered": record.recovered,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "cache_read_tokens": record.cache_read_tokens,
        "cache_write_tokens": record.cache_write_tokens,
        "total_tokens": record.total_tokens,
        "cost_usd": record.cost_usd,
    }
    for reason, count in record.attempted.items():
        row[f"attempted_{reason}"] = count
    for reason, count in record.admitted.items():
        row[f"admitted_{reason}"] = count
    return row


def write_master_table(records: Sequence[CellRecord], out_dir: Path) -> None:
    """The one canonical artifact: one row per conversation -> results.csv + results.json.
    Violation buckets are flattened to attempted_<reason>/admitted_<reason> columns
    (union across rows; missing -> 0)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_flat_row(r) for r in records])
    bucket_cols = [c for c in frame.columns if c.startswith(("attempted_", "admitted_"))]
    if bucket_cols:
        frame[bucket_cols] = frame[bucket_cols].fillna(0).astype(int)
    frame = frame.sort_values(["model", "config", "scenario", "seed"]).reset_index(drop=True)
    frame.to_csv(out_dir / "results.csv", index=False)
    frame.to_json(out_dir / "results.json", orient="records", indent=2)


@dataclass(frozen=True)
class PassKRow:
    model: str
    config: str
    k: int
    passk: float
    n_scenarios: int


def pass_k(records: Sequence[CellRecord], *, refined: bool = False) -> list[PassKRow]:
    """pass^k via the combinatorial (unbiased) estimator C(c,k)/C(n,k) per scenario,
    averaged over scenarios within each (model, config). Never pooled.
    For a scenario with n seeds and c successes, k ranges 1..n.
    Pass refined=True to use task_success_refined instead of task_success."""
    # (model, config, scenario) -> [n_seeds, n_success]
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        g = groups[(r.model, r.config, r.scenario)]
        g[0] += 1
        g[1] += int(r.task_success_refined if refined else r.task_success)

    # (model, config) -> k -> list of per-scenario pass^k values
    by_cfg: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (model, config, _scenario), (n, c) in groups.items():
        for k in range(1, n + 1):
            by_cfg[(model, config)][k].append(comb(c, k) / comb(n, k))

    rows: list[PassKRow] = []
    for (model, config), per_k in sorted(by_cfg.items()):
        for k in sorted(per_k):
            values = per_k[k]
            rows.append(PassKRow(model, config, k, sum(values) / len(values), len(values)))
    return rows


def write_passk(
    rows: Sequence[PassKRow], out_dir: Path, *, filename: str = "passk.csv"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r.__dict__ for r in rows])
    frame.to_csv(out_dir / filename, index=False)


@dataclass(frozen=True)
class ScoringDeltaRow:
    model: str
    config: str
    n_cells: int
    n_pass_strict: int
    n_pass_refined: int
    n_rescued: int
    rate_strict: float
    rate_refined: float


def scoring_delta(records: Sequence[CellRecord]) -> list[ScoringDeltaRow]:
    """Per (model, config): strict vs refined pass counts/rates and the number of cells the
    refinement rescues (refined success while strict fails). A final ALL/ALL totals row is
    appended. This is the paper's "size of the refinement" artifact."""
    groups: dict[tuple[str, str], list[CellRecord]] = defaultdict(list)
    for r in records:
        groups[(r.model, r.config)].append(r)

    def _row(model: str, config: str, rs: Sequence[CellRecord]) -> ScoringDeltaRow:
        n = len(rs)
        n_strict = sum(1 for r in rs if r.task_success)
        n_refined = sum(1 for r in rs if r.task_success_refined)
        n_rescued = sum(1 for r in rs if r.task_success_refined and not r.task_success)
        return ScoringDeltaRow(
            model=model,
            config=config,
            n_cells=n,
            n_pass_strict=n_strict,
            n_pass_refined=n_refined,
            n_rescued=n_rescued,
            rate_strict=n_strict / n if n else 0.0,
            rate_refined=n_refined / n if n else 0.0,
        )

    rows = [_row(model, config, rs) for (model, config), rs in sorted(groups.items())]
    rows.append(_row("ALL", "ALL", list(records)))
    return rows


def write_scoring_delta(rows: Sequence[ScoringDeltaRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(out_dir / "scoring_delta.csv", index=False)


def aggregate_run(run_dir: Path, scenarios_by_id: Mapping[str, Scenario]) -> Path:
    """Build the master table + pass^k under run_dir/aggregate/. Returns that dir."""
    # Local import: bench.stats imports CellRecord from this module, so a top-level
    # import here would be circular (stats loads before CellRecord is defined).
    from bench.stats import write_stats

    records = list(load_cell_records(run_dir, scenarios_by_id))
    out_dir = run_dir / "aggregate"
    write_master_table(records, out_dir)
    write_passk(pass_k(records), out_dir)
    write_passk(pass_k(records, refined=True), out_dir, filename="passk_refined.csv")
    write_scoring_delta(scoring_delta(records), out_dir)
    write_stats(records, out_dir)
    return out_dir


def main() -> None:
    import argparse

    from bench.scenario import load_scenarios

    parser = argparse.ArgumentParser(description="Aggregate a PRISM run archive.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    scenarios_by_id = {s.id: s for s in load_scenarios("scenarios")}
    out_dir = aggregate_run(Path(args.run_dir), scenarios_by_id)
    print(f"wrote {out_dir}/results.csv, results.json, passk.csv")  # noqa: T201


if __name__ == "__main__":
    main()
