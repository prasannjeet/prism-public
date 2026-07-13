"""Prompt-file tests — the shared process-rules block must be identical across C1..C5."""

from __future__ import annotations

from pathlib import Path

PROMPTS = Path("prompts")
MARKER = "Process rules (apply whenever the relevant tools are available):"
RULES_KEYS = [
    "Confirmation means approval.",
    "Submitting or changing any answer clears a prior confirmation.",
    "Corrections overwrite.",
    "Evidence is tool-collected.",
    "Commit is final.",
]


def test_every_prompt_has_the_identical_process_rules_block() -> None:
    blocks = {}
    for name in ("C1", "C2", "C3", "C4", "C5"):
        text = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
        assert MARKER in text, f"{name} missing process-rules block"
        for key in RULES_KEYS:
            assert key in text, f"{name} missing rule: {key}"
        blocks[name] = text[text.index(MARKER) :]
    # the appended block is byte-identical across all five
    assert len(set(blocks.values())) == 1


def test_no_em_dash_in_prompts() -> None:
    for name in ("C1", "C2", "C3", "C4", "C5"):
        assert "—" not in (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
