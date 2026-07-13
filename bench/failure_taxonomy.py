"""Failure-mode taxonomy counts over a frozen run's results.csv (zero model calls).
Buckets every STRICT-failing cell into exactly one mode by precedence; also reports the
attempted-vs-admitted totals per enforce-sensitive violation kind. Feeds the paper's
failure-analysis paragraph + table T2; the qualitative labels live in the taxonomy note."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

ADMITTED_PREFIX = "admitted_"
ENFORCE_SENSITIVE = ("inactive_field", "not_complete", "invalid_value",
                     "not_confirmed", "evidence_missing", "wrong_kind")
FULL_GRID_MODELS = ("claude-haiku-4-5", "gpt-5.4-mini", "gemini-3.1-flash-lite")


def admitted_total(row: pd.Series) -> int:
    return int(sum(int(v) for k, v in row.items() if str(k).startswith(ADMITTED_PREFIX)))


def classify_failure(row: pd.Series) -> str:
    """Bucket one STRICT-failing cell by explicit precedence (partition guarantee)."""
    if str(row["terminal"]) in ("committed", "escalated"):
        return "finished_wrong"
    if int(row["contamination"]) > 0:
        return "contaminated"
    if admitted_total(row) > 0:
        return "admitted_premature"
    return "truncated"


def classify_failures(frame: pd.DataFrame) -> dict[str, int]:
    failing = frame[~frame["task_success"].astype(bool)]
    return dict(Counter(classify_failure(r) for _, r in failing.iterrows()))


def c5_admitted_total(frame: pd.DataFrame) -> int:
    c5 = frame[frame["config"] == "C5"]
    return int(sum(admitted_total(r) for _, r in c5.iterrows()))


def enforcement_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Per config (C4,C5), summed across the full-grid cheap models, attempted vs admitted
    for each enforce-sensitive kind. Single source for table T2 + the note's enforcement row."""
    sub = frame[frame["model"].isin(FULL_GRID_MODELS)]
    rows = []
    for config in ("C4", "C5"):
        cfg = sub[sub["config"] == config]
        for kind in ENFORCE_SENSITIVE:
            a, d = f"attempted_{kind}", f"admitted_{kind}"
            if a not in cfg.columns:
                continue
            rows.append({
                "config": config, "kind": kind,
                "attempted": int(cfg[a].sum()),
                "admitted": int(cfg[d].sum()) if d in cfg.columns else 0,
            })
    return pd.DataFrame(rows)


def _format_enforcement_table(df: pd.DataFrame) -> str:
    """Render enforcement_totals DataFrame as a plain text table (no tabulate needed)."""
    cols = ["config", "kind", "attempted", "admitted"]
    widths = {c: max(len(c), max((len(str(r)) for r in df[c]), default=0)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [header, sep]
    for row in df.itertuples(index=False):
        lines.append("  ".join(str(getattr(row, c)).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _report(run_dir: Path) -> str:
    frame = pd.read_csv(run_dir / "aggregate" / "results.csv")
    counts = classify_failures(frame)
    lines = ["## Failure buckets (strict, all models)"]
    for mode in ("truncated", "admitted_premature", "contaminated", "finished_wrong"):
        lines.append(f"- {mode}: {counts.get(mode, 0)}")
    lines.append(f"- TOTAL failing: {sum(counts.values())}")
    lines.append(f"\nC5 admitted-violation total (must be 0): {c5_admitted_total(frame)}")
    lines.append("\n## Enforcement totals (C4 vs C5, cheap models)")
    lines.append(_format_enforcement_table(enforcement_totals(frame)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-taxonomy counts over a run.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(_report(args.run_dir))  # noqa: T201


if __name__ == "__main__":
    main()
