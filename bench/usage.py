"""Per-conversation token capture. A thin ModelAdapter wrapper that
records each provider call's Usage WITHOUT touching the byte-identical event log."""

from __future__ import annotations

from collections.abc import Sequence

from prism.models import Message, ModelAdapter, ModelResponse, ModelSettings, ToolSpec, Usage

_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


class UsageRecorder:
    """Wraps a ModelAdapter; appends each complete()'s Usage to `usages`.
    Use one per cell (fresh list per conversation)."""

    name: str
    settings: ModelSettings

    def __init__(self, base: ModelAdapter) -> None:
        self._base = base
        self.name = base.name
        self.settings = base.settings
        self.usages: list[Usage] = []

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        response = self._base.complete(system, messages, tools)
        self.usages.append(response.usage)
        return response


def usage_call_dict(usage: Usage) -> dict[str, object]:
    """Serialize one call's Usage, preserving the provider's verbatim raw payload."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "raw": dict(usage.raw),
    }


def usage_totals(usages: Sequence[Usage]) -> dict[str, int]:
    """Sum each token kind across all calls in a conversation."""
    return {key: sum(getattr(u, key) for u in usages) for key in _KEYS}
