"""Figures from the master table. Smoke tests: render to a non-empty file."""

from __future__ import annotations

import pandas as pd

from bench.plots import generate_all


def _master_frame() -> pd.DataFrame:
    rows = []
    for config in ("C1", "C5"):
        for scenario in ("A", "B"):
            for seed in (0, 1):
                ok = not (config == "C1" and scenario == "B" and seed == 1)
                rows.append({
                    "model": "claude-haiku-4-5", "workflow": "form_booking",
                    "scenario": scenario, "config": config, "seed": seed,
                    "task_success": ok, "terminal": "committed", "branch": None,
                    "contamination": 0, "turns": 3, "tool_calls": 2, "tool_errors": 0,
                    "recovered": False,
                    "input_tokens": 1000 if config == "C5" else 3000,
                    "output_tokens": 100, "cache_read_tokens": 0, "cache_write_tokens": 0,
                    "total_tokens": (1100 if config == "C5" else 3100),
                    "cost_usd": 0.01,
                    "attempted_inactive_field": 0 if config == "C5" else 2,
                    "admitted_inactive_field": 0 if config == "C5" else 2,
                })
    return pd.DataFrame(rows)


def test_generate_all_writes_four_nonempty_figures(tmp_path) -> None:
    frame = _master_frame()
    csv_path = tmp_path / "results.csv"
    frame.to_csv(csv_path, index=False)
    out_dir = tmp_path / "figures"

    paths = generate_all(csv_path, out_dir)

    assert len(paths) == 4
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
