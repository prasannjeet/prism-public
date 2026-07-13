"""Generate the verbatim projection example + an 8-line YAML excerpt for the paper appendix.
Deterministic prototype output: same workflow + same answer sequence -> same listing."""

from __future__ import annotations

from pathlib import Path

from bench.scenario import WORKFLOWS_DIR
from prism.engine import StateEngine
from prism.projection import render_detail, render_schema, render_state
from prism.schema import load_workflow

OUT = Path("paper/sections/listings")


def render_example() -> str:
    engine = StateEngine(load_workflow(WORKFLOWS_DIR / "form_booking.yaml"))
    state = engine.initial_state()
    state = engine.with_answer(state, "service_type", "move_out")
    state = engine.with_answer(state, "customer_name", "Dana Lee")
    state = engine.with_answer(state, "customer_email", "dana@example.com")
    parts = [
        "# Tier 1: flat schema, injected once into the system prompt",
        render_schema(engine.workflow),
        "",
        "# Tier 2: compact state block, re-rendered every turn",
        render_state(engine, state),
        "",
        "# Tier 3: get_detail(deposit_protection), fetched on demand",
        render_detail(engine.workflow, "deposit_protection"),
    ]
    return "\n".join(parts)


def form_excerpt() -> str:
    lines = (WORKFLOWS_DIR / "form_booking.yaml").read_text(encoding="utf-8").splitlines()
    # the service_type driver + one branched child (the move_out / deposit_protection chain)
    start = next(i for i, ln in enumerate(lines) if "name: service_type" in ln)
    return "\n".join(lines[start - 1 : start + 13])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "projection_example.txt").write_text(render_example() + "\n", encoding="utf-8")
    (OUT / "form_excerpt.yaml").write_text(form_excerpt() + "\n", encoding="utf-8")
    print(f"wrote projection_example.txt + form_excerpt.yaml to {OUT}")  # noqa: T201


if __name__ == "__main__":
    main()
