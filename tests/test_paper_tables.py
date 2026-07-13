from __future__ import annotations

import pandas as pd

from bench.paper_tables import (
    config_defs,
    enforcement_rows,
    fmt_pct,
    latex_escape,
    main_results_main,
    workflow_stats,
)


def test_fmt_pct_one_dp():
    assert fmt_pct(0.7639484978540773) == "76.4"
    assert fmt_pct(0.0) == "0.0"


def test_latex_escape_underscores():
    assert latex_escape("claude_haiku_4-5") == r"claude\_haiku\_4-5"


def test_enforcement_rows_drops_purely_structural_and_pools_cheap_models():
    df = pd.DataFrame([
        {"model": "claude-haiku-4-5", "config": "C4",
         "attempted_not_complete": 5, "admitted_not_complete": 5,
         "attempted_wrong_kind": 2, "admitted_wrong_kind": 0},
        {"model": "claude-haiku-4-5", "config": "C5",
         "attempted_not_complete": 4, "admitted_not_complete": 0,
         "attempted_wrong_kind": 1, "admitted_wrong_kind": 0},
        {"model": "claude-sonnet-4-6", "config": "C5",
         "attempted_not_complete": 9, "admitted_not_complete": 0,
         "attempted_wrong_kind": 9, "admitted_wrong_kind": 9},
    ])
    rows = enforcement_rows(df)
    assert {r["kind"] for r in rows} == {"not_complete"}
    nc = next(r for r in rows if r["kind"] == "not_complete")
    assert nc == {"kind": "not_complete", "c4_attempted": 5, "c4_admitted": 5,
                  "c5_attempted": 4, "c5_admitted": 0}


def test_main_results_main_pivots_models_to_columns():
    """The condensed body table matches the paper's hand-tuned layout: models as
    columns in fixed order, `--` for absent cells, the C5 row in bold."""
    df = pd.DataFrame([
        {"model": "claude-haiku-4-5", "config": "C1", "task_success": 0.0},
        {"model": "claude-haiku-4-5", "config": "C5", "task_success": 0.764},
        {"model": "claude-sonnet-4-6", "config": "C5", "task_success": 0.670},
    ])
    tex = main_results_main(df)
    assert r"Cfg & haiku & gpt & gemini & sonnet \\" in tex
    assert r"C1 & 0.0 & -- & -- & -- \\" in tex
    assert r"C5 & \textbf{76.4} & -- & -- & \textbf{67.0} \\" in tex
    assert r"\begin{table}[tb]\centering\footnotesize" in tex


def test_static_tables_match_paper_layout():
    """workflow_stats spans both columns (table*) and config_defs uses the
    condensed Shows/Enf. wording; drift here would clobber the checked-in paper
    tables on regeneration."""
    ws = workflow_stats()
    assert ws.startswith(r"\begin{table*}[t]") and ws.rstrip().endswith(r"\end{table*}")
    cd = config_defs()
    assert r"Cfg & Shows & Enf. & Description \\" in cd
    assert r"C5 & PRISM & on & projection; gates enforce \\" in cd
