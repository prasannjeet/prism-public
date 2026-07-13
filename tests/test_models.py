"""Provider-neutral model types — frozen, deterministic, no SDK import at module load."""

from __future__ import annotations

import google.genai
import pytest
from google.genai import types

from prism.models import (
    GeminiAdapter,
    Message,
    ModelResponse,
    ModelSettings,
    ScriptedAdapter,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    _to_anthropic,
    _to_gemini_contents,
    _to_openai,
)


def test_value_types_are_frozen_and_carry_raw_usage() -> None:
    usage = Usage(
        input_tokens=10,
        output_tokens=3,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw={"provider": "scripted"},
    )
    call = ToolCall(id="t1", name="submit_answer", arguments={"field": "symptom", "value": "down"})
    resp = ModelResponse(text="ok", tool_calls=(call,), usage=usage, stop_reason="tool_use")
    assert resp.tool_calls[0].arguments == {"field": "symptom", "value": "down"}
    assert resp.usage.raw == {"provider": "scripted"}


def test_message_defaults_are_empty_tuples() -> None:
    m = Message(role="user", text="hi")
    assert m.tool_calls == () and m.tool_results == ()


def test_toolspec_holds_provider_neutral_schema() -> None:
    spec = ToolSpec(
        name="confirm",
        description="Confirm the summary.",
        input_schema={"type": "object", "properties": {}},
    )
    assert spec.input_schema["type"] == "object"


def test_model_settings_records_params() -> None:
    assert ModelSettings(params={"temperature": 0.2}).params == {"temperature": 0.2}


def test_tool_result_error_flag_defaults_false() -> None:
    assert ToolResult(call_id="t1", content="x").is_error is False


def _resp(text: str, calls: tuple[ToolCall, ...], stop: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=calls,
        usage=Usage(0, 0, 0, 0, {"provider": "scripted"}),
        stop_reason=stop,
    )


def test_scripted_adapter_returns_scripted_responses_in_order() -> None:
    call = ToolCall(id="t1", name="confirm", arguments={})
    adapter = ScriptedAdapter(
        [
            _resp("", (call,), "tool_use"),
            _resp("all set", (), "end_turn"),
        ]
    )
    first = adapter.complete("sys", [Message(role="user", text="hi")], [])
    second = adapter.complete("sys", [], [])
    assert first.tool_calls == (call,)
    assert second.text == "all set" and second.tool_calls == ()


def test_scripted_adapter_records_received_calls() -> None:
    adapter = ScriptedAdapter([_resp("x", (), "end_turn")])
    adapter.complete("sysprompt", [Message(role="user", text="hello")], [])
    assert adapter.calls[0].system == "sysprompt"
    assert adapter.calls[0].messages[0].text == "hello"


def test_scripted_adapter_raises_when_script_exhausted() -> None:
    adapter = ScriptedAdapter([_resp("x", (), "end_turn")])
    adapter.complete("s", [], [])
    try:
        adapter.complete("s", [], [])
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "script exhausted" in str(e)


# ---------------------------------------------------------------------------
# Task 2 — translator unit tests
# ---------------------------------------------------------------------------


def test_to_anthropic_user_assistant_tool() -> None:
    assert _to_anthropic(Message(role="user", text="hi")) == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }
    call = ToolCall(id="c1", name="confirm", arguments={"k": "v"})
    assert _to_anthropic(Message(role="assistant", text="ok", tool_calls=(call,))) == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "c1", "name": "confirm", "input": {"k": "v"}},
        ],
    }
    res = ToolResult(call_id="c1", content="bad", is_error=True)
    assert _to_anthropic(Message(role="tool", tool_results=(res,))) == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "bad", "is_error": True}
        ],
    }


def test_to_openai_fans_out_multiple_tool_results() -> None:
    assert _to_openai(Message(role="user", text="hi")) == [{"role": "user", "content": "hi"}]
    r1 = ToolResult(call_id="a", content="A")
    r2 = ToolResult(call_id="b", content="B")
    assert _to_openai(Message(role="tool", tool_results=(r1, r2))) == [
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
    ]
    call = ToolCall(id="c1", name="confirm", arguments={"k": "v"})
    out = _to_openai(Message(role="assistant", text="", tool_calls=(call,)))
    assert out[0]["role"] == "assistant" and out[0]["content"] is None
    assert out[0]["tool_calls"][0]["function"]["name"] == "confirm"  # type: ignore[index]
    import json

    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"k": "v"}  # type: ignore[index]


def test_to_gemini_contents_resolves_tool_result_names_by_call_id() -> None:
    call_a = ToolCall(id="c1", name="check_logs", arguments={"q": "errors"})
    call_b = ToolCall(id="c2", name="check_deploys", arguments={})
    messages = [
        Message(role="user", text="investigate"),
        Message(role="assistant", text="working", tool_calls=(call_a, call_b)),
        Message(
            role="tool",
            tool_results=(
                ToolResult(call_id="c1", content="OOM at 16:00"),
                ToolResult(call_id="c2", content="deploy 42"),
            ),
        ),
    ]
    contents = _to_gemini_contents(messages)
    assert contents[0] == {"role": "user", "parts": [{"text": "investigate"}]}
    # assistant -> role "model"; text part + one function_call part per tool call
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"text": "working"},
            {"function_call": {"name": "check_logs", "args": {"q": "errors"}}},
            {"function_call": {"name": "check_deploys", "args": {}}},
        ],
    }
    # tool results -> function_response parts, names resolved from the call ids above
    assert contents[2] == {
        "role": "user",
        "parts": [
            {"function_response": {"name": "check_logs", "response": {"result": "OOM at 16:00"}}},
            {"function_response": {"name": "check_deploys", "response": {"result": "deploy 42"}}},
        ],
    }


def test_to_gemini_contents_replays_thought_signature() -> None:
    # Gemini 3 attaches a thought_signature to each function_call part and REQUIRES it echoed
    # back on replay, or it 400s ("Function call is missing a thought_signature"). The emitted
    # dict must also be a valid Gemini Content (SDK accepts thought_signature on the part).
    call = ToolCall(id="c1", name="submit_answer", arguments={"v": 1}, signature=b"sigblob")
    contents = _to_gemini_contents([Message(role="assistant", tool_calls=(call,))])
    assert contents == [
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {"name": "submit_answer", "args": {"v": 1}},
                    "thought_signature": b"sigblob",
                }
            ],
        }
    ]
    validated = types.Content.model_validate(contents[0])
    assert validated.parts is not None
    assert validated.parts[0].thought_signature == b"sigblob"
    assert validated.parts[0].function_call is not None
    assert validated.parts[0].function_call.name == "submit_answer"


def test_to_gemini_contents_omits_signature_when_absent() -> None:
    # A call with no signature (Anthropic/OpenAI-sourced, or none returned) emits no key.
    call = ToolCall(id="c1", name="confirm", arguments={})
    assert _to_gemini_contents([Message(role="assistant", tool_calls=(call,))]) == [
        {"role": "model", "parts": [{"function_call": {"name": "confirm", "args": {}}}]}
    ]


def test_gemini_complete_captures_thought_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="submit_answer", args={"x": 1}),
                    thought_signature=b"sigblob",
                ),
            ],
        ),
        finish_reason=types.FinishReason.STOP,
    )
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=1, candidates_token_count=1
    )
    response = types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)
    _install_fake_gemini(monkeypatch, response)

    resp = GeminiAdapter().complete("sys", [Message(role="user", text="hi")], [])
    assert resp.tool_calls[0].signature == b"sigblob"


def test_to_gemini_contents_user_only_and_textless_assistant() -> None:
    assert _to_gemini_contents([Message(role="user", text="hi")]) == [
        {"role": "user", "parts": [{"text": "hi"}]}
    ]
    call = ToolCall(id="c1", name="confirm", arguments={})
    # an assistant turn with only a tool call (no text) emits no empty text part
    assert _to_gemini_contents([Message(role="assistant", tool_calls=(call,))]) == [
        {"role": "model", "parts": [{"function_call": {"name": "confirm", "args": {}}}]}
    ]


def _install_fake_gemini(
    monkeypatch: pytest.MonkeyPatch, response: types.GenerateContentResponse
) -> None:
    """Monkeypatch google.genai.Client so GeminiAdapter.__init__ builds offline (no key/network).
    The fake client's models.generate_content() returns the canned `response` regardless of args."""

    class _FakeModels:
        def generate_content(
            self, *, model: str, contents: object, config: object
        ) -> types.GenerateContentResponse:
            return response

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr(google.genai, "Client", _FakeClient)


def test_gemini_complete_parses_text_and_function_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real SDK types validate our field-name assumptions: text part + function-call part.
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Recording that now."),
                types.Part(
                    function_call=types.FunctionCall(
                        name="submit_answer", args={"field": "name", "value": "Alice"}
                    )
                ),
            ],
        ),
        finish_reason=types.FinishReason.STOP,
    )
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=11, candidates_token_count=7, cached_content_token_count=3
    )
    response = types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)
    _install_fake_gemini(monkeypatch, response)

    resp = GeminiAdapter().complete("sys", [Message(role="user", text="hi")], [])

    assert resp.text == "Recording that now."
    assert resp.tool_calls == (
        ToolCall(
            id="submit_answer-1",  # synthesized: function-call part is index 1 (text is 0)
            name="submit_answer",
            arguments={"field": "name", "value": "Alice"},
        ),
    )
    assert resp.stop_reason == str(types.FinishReason.STOP)
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 7
    assert resp.usage.cache_read_tokens == 3
    assert resp.usage.cache_write_tokens == 0
    assert resp.usage.raw  # provider payload preserved, non-empty


def test_gemini_complete_tolerates_blocked_candidate_with_no_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A safety/recitation/length stop yields a candidate whose content is None: must not crash.
    candidate = types.Candidate(content=None, finish_reason=types.FinishReason.SAFETY)
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=5, candidates_token_count=0, cached_content_token_count=0
    )
    response = types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)
    _install_fake_gemini(monkeypatch, response)

    resp = GeminiAdapter().complete("sys", [Message(role="user", text="hi")], [])

    assert resp.text == ""
    assert resp.tool_calls == ()
    assert resp.stop_reason == str(types.FinishReason.SAFETY)
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 0
    assert resp.usage.cache_read_tokens == 0
    assert resp.usage.cache_write_tokens == 0
