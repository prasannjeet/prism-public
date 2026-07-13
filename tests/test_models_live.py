"""Live adapter smoke tests — excluded from the default run (need API keys + network).
Run explicitly: `.venv/bin/pytest -m live tests/test_models_live.py`."""

from __future__ import annotations

import os

import pytest

from prism.models import AnthropicAdapter, GeminiAdapter, Message, OpenAIAdapter


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_round_trip_populates_usage() -> None:
    adapter = AnthropicAdapter()
    resp = adapter.complete("You are terse.", [Message(role="user", text="Say 'ok'.")], [])
    assert resp.usage.input_tokens > 0


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_round_trip_populates_usage() -> None:
    adapter = OpenAIAdapter()
    resp = adapter.complete("You are terse.", [Message(role="user", text="Say 'ok'.")], [])
    assert resp.usage.input_tokens > 0


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="no GEMINI_API_KEY/GOOGLE_API_KEY",
)
def test_gemini_round_trip_populates_usage() -> None:
    adapter = GeminiAdapter()
    resp = adapter.complete("You are terse.", [Message(role="user", text="Say 'ok'.")], [])
    assert resp.usage.input_tokens > 0
