from __future__ import annotations

import pandas as pd

from bench.failure_taxonomy import ADMITTED_PREFIX, admitted_total, classify_failures


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "config": "C5",
                "task_success": True,
                "terminal": "committed",
                "contamination": 0,
                "admitted_not_complete": 0,
            },
            {
                "config": "C5",
                "task_success": False,
                "terminal": "committed",
                "contamination": 0,
                "admitted_not_complete": 0,
            },
            {
                "config": "C1",
                "task_success": False,
                "terminal": "none",
                "contamination": 3,
                "admitted_not_complete": 1,
            },
            {
                "config": "C4",
                "task_success": False,
                "terminal": "none",
                "contamination": 0,
                "admitted_not_complete": 2,
            },
            {
                "config": "C5",
                "task_success": False,
                "terminal": "none",
                "contamination": 0,
                "admitted_not_complete": 0,
            },
        ]
    )


def test_admitted_prefix_constant() -> None:
    assert ADMITTED_PREFIX == "admitted_"


def test_admitted_total_sums_admitted_columns_only() -> None:
    df = _frame()
    assert admitted_total(df.iloc[2]) == 1
    assert admitted_total(df.iloc[0]) == 0


def test_classify_partitions_failures_by_precedence() -> None:
    counts = classify_failures(_frame())
    assert counts["finished_wrong"] == 1
    assert counts["contaminated"] == 1
    assert counts["admitted_premature"] == 1
    assert counts["truncated"] == 1
    assert sum(counts.values()) == 4
