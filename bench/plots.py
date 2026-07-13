"""Descriptive figures from results.csv. Headless Agg backend; per-model
facets (no cross-model averaging). Point estimates only -- confidence bands
are Phase 3."""

from __future__ import annotations

from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _passk_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (model, config, k) combinatorial pass^k averaged over scenarios."""
    rows = []
    grouped = frame.groupby(["model", "config", "scenario"])["task_success"]
    agg = grouped.agg(["count", "sum"]).reset_index()
    for (model, config), block in agg.groupby(["model", "config"]):
        # Cap k at the smallest seed count so every scenario enters every point
        # (a consistent denominator). Without this, scenarios with fewer seeds drop
        # out as k grows and the averaged curve can rise with k, which is impossible
        # for a fixed scenario set.
        min_n = int(block["count"].min())
        for k in range(1, min_n + 1):
            vals = [
                comb(int(c), k) / comb(int(n), k)
                for n, c in zip(block["count"], block["sum"], strict=True)
            ]
            rows.append({"model": model, "config": config, "k": k, "passk": sum(vals) / len(vals)})
    return pd.DataFrame(rows)


def plot_passk(frame: pd.DataFrame, out_dir: Path) -> Path:
    curve = _passk_curve(frame)
    models = sorted(curve["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        for config, block in curve[curve["model"] == model].groupby("config"):
            ax.plot(block["k"], block["passk"], marker="o", label=config)
        ax.set_title(model)
        ax.set_xticks(sorted(curve["k"].unique()))  # integer k only (repeats are whole)
        ax.set_xlabel("k (repeats)")
        ax.set_ylabel("pass^k")
        ax.set_ylim(0, 1.05)
        ax.legend(title="config")
    fig.tight_layout()
    path = out_dir / "passk.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_token_economics(frame: pd.DataFrame, out_dir: Path) -> Path:
    means = frame.groupby(["model", "config"])["total_tokens"].mean().reset_index()
    models = sorted(means["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        block = means[means["model"] == model].sort_values("config")
        ax.bar(block["config"], block["total_tokens"])
        ax.set_title(model)
        ax.set_xlabel("config")
        ax.set_ylabel("mean tokens / conversation")
    fig.tight_layout()
    path = out_dir / "token_economics.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_attempted_admitted(frame: pd.DataFrame, out_dir: Path) -> Path:
    att_cols = [c for c in frame.columns if c.startswith("attempted_")]
    adm_cols = [c for c in frame.columns if c.startswith("admitted_")]
    work = frame.assign(
        _attempted=frame[att_cols].sum(axis=1) if att_cols else 0,
        _admitted=frame[adm_cols].sum(axis=1) if adm_cols else 0,
    )
    totals = work.groupby("config")[["_attempted", "_admitted"]].sum().reset_index()
    x = range(len(totals))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], totals["_attempted"], width=0.4, label="attempted")
    ax.bar([i + 0.2 for i in x], totals["_admitted"], width=0.4, label="admitted")
    ax.set_xticks(list(x))
    ax.set_xticklabels(totals["config"])
    ax.set_xlabel("config")
    ax.set_ylabel("violations")
    ax.set_title("Attempted vs admitted (T2)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "attempted_admitted.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_success_rates(frame: pd.DataFrame, out_dir: Path) -> Path:
    rates = frame.groupby(["model", "config"])["task_success"].mean().reset_index()
    models = sorted(rates["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        block = rates[rates["model"] == model].sort_values("config")
        ax.bar(block["config"], block["task_success"])
        ax.set_title(model)
        ax.set_xlabel("config")
        ax.set_ylabel("task-success rate")
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = out_dir / "success_rates.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all(csv_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_path)
    return [
        plot_passk(frame, out_dir),
        plot_token_economics(frame, out_dir),
        plot_attempted_admitted(frame, out_dir),
        plot_success_rates(frame, out_dir),
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate PRISM figures from a run aggregate.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    paths = generate_all(run_dir / "aggregate" / "results.csv", run_dir / "figures")
    for path in paths:  # noqa: T201
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
