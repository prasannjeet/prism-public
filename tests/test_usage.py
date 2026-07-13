"""UsageRecorder: a ModelAdapter wrapper that captures each call's Usage."""

from __future__ import annotations

from types import MappingProxyType

from bench.usage import UsageRecorder, usage_call_dict, usage_totals
from prism.models import (
    Message,
    ModelResponse,
    ScriptedAdapter,
    Usage,
)


def _resp(inp: int, out: int) -> ModelResponse:
    return ModelResponse(
        text="ok",
        tool_calls=(),
        usage=Usage(input_tokens=inp, output_tokens=out, cache_read_tokens=0, cache_write_tokens=0),
        stop_reason="end_turn",
    )


def test_recorder_collects_one_usage_per_complete_call() -> None:
    base = ScriptedAdapter([_resp(100, 10), _resp(200, 20)])
    rec = UsageRecorder(base)
    rec.complete("sys", [Message(role="user", text="a")], [])
    rec.complete("sys", [Message(role="user", text="b")], [])
    assert [u.input_tokens for u in rec.usages] == [100, 200]
    assert [u.output_tokens for u in rec.usages] == [10, 20]


def test_recorder_exposes_base_name_and_settings() -> None:
    base = ScriptedAdapter([_resp(1, 1)])
    rec = UsageRecorder(base)
    assert rec.name == base.name
    assert rec.settings == base.settings


def test_usage_totals_sums_all_four_kinds() -> None:
    usages = [
        Usage(10, 1, 2, 3),
        Usage(20, 2, 4, 6),
    ]
    assert usage_totals(usages) == {
        "input_tokens": 30,
        "output_tokens": 3,
        "cache_read_tokens": 6,
        "cache_write_tokens": 9,
    }


def test_usage_call_dict_preserves_raw_payload() -> None:
    u = Usage(10, 1, 0, 0, raw=MappingProxyType({"prompt_tokens": 10}))
    d = usage_call_dict(u)
    assert d["input_tokens"] == 10
    assert d["raw"] == {"prompt_tokens": 10}
