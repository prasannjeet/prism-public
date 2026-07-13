"""Generate drop-in booktabs LaTeX tables from a frozen run's aggregate CSVs (zero model calls).
T1 main results (full + condensed), T2 enforcement delta, T3 workflow stats, T4 config defs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bench.failure_taxonomy import ENFORCE_SENSITIVE, FULL_GRID_MODELS
from bench.scenario import WORKFLOWS_DIR
from prism.schema import load_workflow, referenced_fields

CONFIGS = ("C1", "C2", "C3", "C4", "C5")
WORKFLOW_ORDER = ("form_booking", "incident_investigation", "support_diagnosis",
                  "travel_insurance_claim", "incident_response", "clinical_trial_eligibility")
CONFIG_DEFS = {
    "C1": ("nothing", "off", "no workflow shown"),
    "C2": ("1 field", "on", "next field only"),
    "C3": ("full YAML", "on", "full spec each turn"),
    "C4": ("PRISM", "off", "projection; gates log"),
    "C5": ("PRISM", "on", "projection; gates enforce"),
}
# Column order + short header names for the condensed body table (T1-main).
MODEL_COLUMNS = (("claude-haiku-4-5", "haiku"), ("gpt-5.4-mini", "gpt"),
                 ("gemini-3.1-flash-lite", "gemini"), ("claude-sonnet-4-6", "sonnet"))


def fmt_pct(x: float) -> str:
    return f"{100 * float(x):.1f}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def enforcement_rows(frame: pd.DataFrame) -> list[dict[str, int | str]]:
    sub = frame[frame["model"].isin(FULL_GRID_MODELS)]
    c4, c5 = sub[sub["config"] == "C4"], sub[sub["config"] == "C5"]
    rows: list[dict[str, int | str]] = []
    for kind in ENFORCE_SENSITIVE:
        a, d = f"attempted_{kind}", f"admitted_{kind}"
        if a not in sub.columns or d not in sub.columns:
            continue
        c4_admitted = int(c4[d].sum())
        if c4_admitted == 0:
            continue
        rows.append({"kind": kind,
                     "c4_attempted": int(c4[a].sum()), "c4_admitted": c4_admitted,
                     "c5_attempted": int(c5[a].sum()), "c5_admitted": int(c5[d].sum())})
    return rows


def _wilson(wilson: pd.DataFrame, model: str, config: str) -> str:
    r = wilson[(wilson.model == model) & (wilson.config == config) & (wilson.scoring == "strict")]
    if r.empty:
        return "--"
    return f"[{fmt_pct(r.ci_low.iloc[0])}, {fmt_pct(r.ci_high.iloc[0])}]"


def _passk(passk: pd.DataFrame, model: str, config: str, k: int) -> str:
    r = passk[(passk.model == model) & (passk.config == config) & (passk.k == k)]
    return fmt_pct(r.passk.iloc[0]) if not r.empty else "--"


def main_results_full(results: pd.DataFrame, wilson: pd.DataFrame, passk: pd.DataFrame) -> str:
    g = results.groupby(["model", "config"])
    body: list[str] = []
    for model in sorted(results.model.unique()):
        body.append(r"\midrule")
        body.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{latex_escape(model)}}}}} \\")
        for config in CONFIGS:
            if (model, config) not in g.groups:
                continue
            rows = g.get_group((model, config))
            wi = _wilson(wilson, model, config)
            line = (f"{config} & {fmt_pct(rows.task_success.mean())} {wi}"
                    f" & {fmt_pct(rows.task_success_refined.mean())}"
                    f" & {_passk(passk, model, config, 1)}"
                    f" & {_passk(passk, model, config, 2)}"
                    f" & {rows.turns.mean():.1f} & {rows.total_tokens.mean():.0f}"
                    f" & {rows.cost_usd.mean():.4f} \\\\")
            body.append(line)
    col_hdr = (
        r"Cfg & Success [95\% CI] & Refined & pass$^1$ & pass$^2$"
        r" & Turns & Tokens & Cost\$ \\"
    )
    header = "\n".join([
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        col_hdr,
    ])
    footer = (r"\bottomrule" "\n" r"\end{tabular}" "\n"
              r"\caption{Per-model results across configurations C1--C5 (strict scoring; refined "
              r"success and pass$^k$ alongside). pass$^1$ and pass$^2$ are over all 88 scenarios "
              r"(2--3 seeds each); pass$^3$ for the 57 three-seed scenarios is in the released "
              r"\texttt{passk.csv}.}" "\n"
              r"\label{tab:main-full}" "\n" r"\end{table*}")
    return "\n".join([header, *body, footer])


def main_results_main(results: pd.DataFrame) -> str:
    """Condensed body table: configurations as rows, models as columns, C5 in bold."""
    g = results.groupby(["model", "config"])
    body: list[str] = []
    for config in CONFIGS:
        cells: list[str] = []
        for model, _short in MODEL_COLUMNS:
            if (model, config) in g.groups:
                val = fmt_pct(g.get_group((model, config)).task_success.mean())
                cells.append(rf"\textbf{{{val}}}" if config == "C5" else val)
            else:
                cells.append("--")
        body.append(f"{config} & " + " & ".join(cells) + r" \\")
    header = (r"\begin{table}[tb]\centering\footnotesize\setlength{\tabcolsep}{5pt}" "\n"
              r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
              "Cfg & " + " & ".join(short for _, short in MODEL_COLUMNS) + r" \\" "\n"
              r"\midrule")
    footer = (r"\bottomrule" "\n" r"\end{tabular}" "\n"
              r"\caption{Task success (\%, strict scoring) by configuration and model (sonnet ran"
              "\n"
              r"the C1/C5 subset only). Wilson 95\% intervals, pass$^k$, tokens, and cost are in"
              "\n"
              r"Appendix Table~\ref{tab:main-full}.}" "\n"
              r"\label{tab:main}" "\n" r"\end{table}")
    return "\n".join([header, *body, footer])


def enforcement_delta(results: pd.DataFrame) -> str:
    rows = enforcement_rows(results)
    assert all(r["c5_admitted"] == 0 for r in rows), "C5 admitted a violation: shield breach!"
    body = [f"{latex_escape(str(r['kind']))} & {r['c4_attempted']} & {r['c4_admitted']} & "
            f"{r['c5_attempted']} & {r['c5_admitted']} \\\\" for r in rows]
    tot = {k: sum(int(r[k]) for r in rows)
           for k in ("c4_attempted", "c4_admitted", "c5_attempted", "c5_admitted")}
    total_row = (f"total & {tot['c4_attempted']} & {tot['c4_admitted']} & "
                 f"{tot['c5_attempted']} & \\textbf{{{tot['c5_admitted']}}} \\\\")
    header = (r"\begin{table}[tb]\centering\footnotesize\setlength{\tabcolsep}{4pt}" "\n"
              r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
              r" & \multicolumn{2}{c}{C4 (log)} & \multicolumn{2}{c}{C5 (enforce)} \\" "\n"
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
              r"Violation & Att. & Adm. & Att. & Adm. \\" "\n" r"\midrule")
    footer = (r"\bottomrule" "\n" r"\end{tabular}" "\n"
              r"\caption{Attempted (Att.) vs.\ admitted (Adm.) violations under identical "
              r"prompting, pooled over the three cheap models; rows are the $\mathcal{V}$ "
              r"reasons of \S\ref{sec:method-gates}. Enforcement off (C4) admits all; "
              r"PRISM (C5) admits none.}" "\n"
              r"\label{tab:enforcement}" "\n" r"\end{table}")
    return "\n".join([header, *body, r"\midrule", total_row, footer])


def _branch_counts(workflow: object) -> tuple[int, int]:
    """(branched fields, distinct branch drivers) for a workflow."""
    fields = workflow.fields  # type: ignore[attr-defined]
    branched = sum(1 for f in fields if f.show_when is not None)
    drivers = {ref for f in fields if f.show_when is not None
               for ref in referenced_fields(f.show_when)}
    return branched, len(drivers)


def workflow_stats() -> str:
    body: list[str] = []
    for name in WORKFLOW_ORDER:
        wf = load_workflow(WORKFLOWS_DIR / f"{name}.yaml")
        n_fields = len(wf.fields)
        n_evidence = sum(1 for f in wf.fields if f.satisfied_by)  # satisfied_by => evidence
        # requires_evidence => conclusion
        n_conclusion = sum(1 for f in wf.fields if f.requires_evidence)
        branched, drivers = _branch_counts(wf)
        body.append(f"{latex_escape(name)} & {n_fields} & {drivers} & {branched} & "
                    f"{n_evidence} & {n_conclusion} \\\\")
    header = (r"\begin{table*}[t]\centering\small" "\n" r"\begin{tabular}{lccccc}" "\n"
              r"\toprule" "\n"
              r"Workflow & Fields & Branch drivers & Branched fields & Evidence & Conclusions \\")
    footer = (r"\bottomrule" "\n" r"\end{tabular}" "\n"
              r"\caption{The six workflows by structural complexity.}" "\n"
              r"\label{tab:workflows}" "\n" r"\end{table*}")
    return "\n".join([header, *body, footer])


def config_defs() -> str:
    body = [f"{c} & {latex_escape(proj)} & {enf} & {desc} \\\\"
            for c, (proj, enf, desc) in CONFIG_DEFS.items()]
    header = (r"\begin{table}[tb]\centering\footnotesize\setlength{\tabcolsep}{4pt}" "\n"
              r"\begin{tabular}{llll}" "\n"
              r"\toprule" "\n" r"Cfg & Shows & Enf. & Description \\" "\n" r"\midrule")
    footer = (r"\bottomrule" "\n" r"\end{tabular}" "\n"
              r"\caption{The five configurations: same engine, tools, and scenarios; only "
              r"information delivery and enforcement (Enf.) vary.}" "\n"
              r"\label{tab:configs}" "\n" r"\end{table}")
    return "\n".join([header, *body, footer])


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit paper LaTeX tables from a run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("paper/sections/tables"))
    args = parser.parse_args()
    agg = args.run_dir / "aggregate"
    results = pd.read_csv(agg / "results.csv")
    wilson = pd.read_csv(agg / "wilson_ci.csv")
    passk = pd.read_csv(agg / "passk.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    full_tex = main_results_full(results, wilson, passk) + "\n"
    main_tex = main_results_main(results) + "\n"
    (args.out / "main_results_full.tex").write_text(full_tex)
    (args.out / "main_results_main.tex").write_text(main_tex)
    (args.out / "enforcement_delta.tex").write_text(enforcement_delta(results) + "\n")
    (args.out / "workflow_stats.tex").write_text(workflow_stats() + "\n")
    (args.out / "config_defs.tex").write_text(config_defs() + "\n")
    print(f"wrote 5 .tex tables to {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
