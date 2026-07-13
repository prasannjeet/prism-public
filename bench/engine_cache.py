"""Tiny memoized workflow->engine loader (one StateEngine per workflow, reused per cell)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from prism.engine import StateEngine
from prism.schema import load_workflow

WORKFLOWS_DIR = Path("workflows")


@cache
def build_engine(workflow: str, workflows_dir: Path = WORKFLOWS_DIR) -> StateEngine:
    return StateEngine(load_workflow(workflows_dir / f"{workflow}.yaml"))
