"""Tripwire: the C5 system block stays BELOW Anthropic's prompt-cache floor, so the
'C5 cheap' result is structural (tiny per-turn payload vs C3 reinjection), not caching.
If a schema grows past the
floor this fails on purpose -> revisit the cost mechanism + paper framing."""

from __future__ import annotations

from pathlib import Path

from prism.projection import render_schema
from prism.schema import load_workflow

# Anthropic minimum cacheable prefix; chars/4 is a deliberately rough token estimate.
_CACHE_FLOOR_TOKENS = 4096
_CHARS_PER_TOKEN = 4
_WORKFLOWS = ("form_booking", "incident_investigation", "support_diagnosis")


def _c5_system_block(workflow: str) -> str:
    prompt = Path("prompts/C5.txt").read_text(encoding="utf-8")
    date_line = "Today's date is 2026-01-15."
    schema = render_schema(load_workflow(Path("workflows") / f"{workflow}.yaml"))
    return "\n\n".join([prompt, date_line, schema])


def test_c5_system_block_is_below_anthropic_cache_floor() -> None:
    for workflow in _WORKFLOWS:
        block = _c5_system_block(workflow)
        est_tokens = len(block) // _CHARS_PER_TOKEN
        assert est_tokens < _CACHE_FLOOR_TOKENS, (
            f"{workflow}: C5 system block ~{est_tokens} tok now clears the "
            f"{_CACHE_FLOOR_TOKENS}-tok cache floor; caching would engage -> revisit the "
            "token-economics mechanism + paper framing."
        )
