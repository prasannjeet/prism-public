"""Provider-neutral model interface. The ONLY module that touches an LLM SDK or the network.

Real adapters (Task 8) and the ScriptedAdapter (Task 3) satisfy the same Protocol; the turn
loop depends on the Protocol, never on a concrete SDK. Each adapter returns its provider's raw
usage payload untouched (`Usage.raw`) so token accounting stays faithful for the paper.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object]
    # Provider-opaque Gemini 3 "thought signature" on a function_call part; must be echoed back
    # verbatim on multi-turn replay (else a 400). None for Anthropic/OpenAI. Never logged.
    signature: bytes | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    raw: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class ModelResponse:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    stop_reason: str


@dataclass(frozen=True)
class ModelSettings:
    """The sampling config actually sent to this model — recorded into the run manifest,
    never assumed. Providers differ (Claude adaptive-thinking vs OpenAI temperature/seed)."""

    params: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


class ModelAdapter(Protocol):
    name: str
    settings: ModelSettings

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse: ...


@dataclass(frozen=True)
class _Call:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]


class ScriptedAdapter:
    """Replays a fixed list of ModelResponses. Satisfies ModelAdapter without any SDK —
    the unit-test oracle for agent.py and the no-API-key reproduction path (spec D-Scripted)."""

    name = "scripted"
    settings = ModelSettings()

    def __init__(self, script: list[ModelResponse]) -> None:
        self._script = list(script)
        self._index = 0
        self.calls: list[_Call] = []

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        self.calls.append(_Call(system, tuple(messages), tuple(tools)))
        if self._index >= len(self._script):
            raise RuntimeError("ScriptedAdapter script exhausted")
        response = self._script[self._index]
        self._index += 1
        return response


class AnthropicAdapter:
    """Anthropic Messages API via the official `anthropic` SDK. Tier-1 system block is cached so
    the 'C5 is cheap' token-economics result is measured, not asserted."""

    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 2048) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        self.settings = ModelSettings(params={"max_tokens": max_tokens})

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=[
                {"name": t.name, "description": t.description, "input_schema": dict(t.input_schema)}
                for t in tools
            ],
            messages=[_to_anthropic(m) for m in messages],  # type: ignore[misc]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = tuple(
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        )
        u = resp.usage
        return ModelResponse(
            text=text,
            tool_calls=calls,
            stop_reason=resp.stop_reason or "end_turn",
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                raw=u.model_dump(),
            ),
        )


def _to_anthropic(m: Message) -> dict[str, object]:
    if m.role == "user":
        return {"role": "user", "content": [{"type": "text", "text": m.text}]}
    if m.role == "assistant":
        content: list[dict[str, object]] = []
        if m.text:
            content.append({"type": "text", "text": m.text})
        for c in m.tool_calls:
            content.append(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": dict(c.arguments)}
            )
        return {"role": "assistant", "content": content}
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in m.tool_results
        ],
    }


class OpenAIAdapter:
    """OpenAI chat-completions via the official `openai` SDK."""

    def __init__(self, model: str = "gpt-5.4-mini") -> None:
        import openai

        self._client = openai.OpenAI()
        self._model = model
        self.name = f"openai:{model}"
        self.settings = ModelSettings(params={"model": model})

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        import json

        import openai

        payload: list[dict[str, object]] = [{"role": "system", "content": system}]
        for m in messages:
            payload.extend(_to_openai(m))
        tool_specs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": dict(t.input_schema),
                },
            }
            for t in tools
        ]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=payload,  # type: ignore[arg-type]
            tools=tool_specs if tool_specs else openai.NOT_GIVEN,  # type: ignore[arg-type]
        )
        choice = resp.choices[0]
        calls = tuple(
            ToolCall(
                id=tc.id,
                name=tc.function.name,  # type: ignore[union-attr]
                arguments=json.loads(tc.function.arguments or "{}"),  # type: ignore[union-attr]
            )
            for tc in (choice.message.tool_calls or [])
        )
        u = resp.usage
        return ModelResponse(
            text=choice.message.content or "",
            tool_calls=calls,
            stop_reason=choice.finish_reason or "stop",
            usage=Usage(
                input_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
                output_tokens=getattr(u, "completion_tokens", 0) if u else 0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                raw=u.model_dump() if u else {},
            ),
        )


def _to_openai(m: Message) -> list[dict[str, object]]:
    """Translate one neutral Message to one-or-more OpenAI chat messages. A tool-role Message
    may carry several results; OpenAI wants exactly one `tool` message per result, so this
    returns a list and `complete` flattens it (handles the multi-tool-call turn correctly)."""
    import json

    if m.role == "user":
        return [{"role": "user", "content": m.text}]
    if m.role == "assistant":
        out: dict[str, object] = {"role": "assistant", "content": m.text or None}
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(dict(c.arguments))},
                }
                for c in m.tool_calls
            ]
        return [out]
    # tool role: one OpenAI `tool` message per result.
    return [
        {"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in m.tool_results
    ]


def _to_gemini_contents(messages: list[Message]) -> list[dict[str, object]]:
    """Translate the neutral message list to Gemini `contents` dicts. Gemini matches a tool
    response to its call by function NAME (not an id), so we resolve each ToolResult.call_id
    back to the function name from the tool_calls seen so far."""
    id_to_name: dict[str, str] = {}
    contents: list[dict[str, object]] = []
    for m in messages:
        if m.role == "user":
            contents.append({"role": "user", "parts": [{"text": m.text}]})
        elif m.role == "assistant":
            parts: list[dict[str, object]] = []
            if m.text:
                parts.append({"text": m.text})
            for c in m.tool_calls:
                id_to_name[c.id] = c.name
                fc_part: dict[str, object] = {
                    "function_call": {"name": c.name, "args": dict(c.arguments)}
                }
                if c.signature is not None:  # Gemini 3 requires the signature echoed back
                    fc_part["thought_signature"] = c.signature
                parts.append(fc_part)
            contents.append({"role": "model", "parts": parts})
        else:  # tool results -> function_response parts, name resolved by call_id
            parts = [
                {
                    "function_response": {
                        "name": id_to_name.get(r.call_id, r.call_id),
                        "response": {"result": r.content},
                    }
                }
                for r in m.tool_results
            ]
            contents.append({"role": "user", "parts": parts})
    return contents


class GeminiAdapter:
    """Google Gemini via the official `google-genai` SDK. System prompt goes in
    `system_instruction`; tool responses are matched by function name (see _to_gemini_contents)."""

    def __init__(self, model: str = "gemini-3.1-flash-lite") -> None:
        from google import genai

        self._client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self._model = model
        self.name = f"gemini:{model}"
        self.settings = ModelSettings(params={"model": model})

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=dict(t.input_schema),  # type: ignore[arg-type]
            )
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=_to_gemini_contents(messages),  # type: ignore[arg-type]
            config=config,
        )
        # A candidate may carry no content (safety/recitation block, or a length cap with no
        # parts) -> content is None; tolerate it rather than crash the run.
        candidate = resp.candidates[0] if resp.candidates else None
        content = candidate.content if candidate else None
        parts = (content.parts or []) if content else []
        text = "".join(p.text for p in parts if p.text)
        calls = tuple(
            ToolCall(
                id=f"{fc.name}-{i}",
                name=fc.name or "",
                arguments=dict(fc.args or {}),
                signature=p.thought_signature,  # carry Gemini 3's signature for replay
            )
            for i, p in enumerate(parts)
            if (fc := p.function_call) is not None
        )
        u = resp.usage_metadata
        return ModelResponse(
            text=text,
            tool_calls=calls,
            stop_reason=str(candidate.finish_reason) if candidate else "stop",
            usage=Usage(
                input_tokens=getattr(u, "prompt_token_count", 0) or 0,
                output_tokens=getattr(u, "candidates_token_count", 0) or 0,
                cache_read_tokens=getattr(u, "cached_content_token_count", 0) or 0,
                cache_write_tokens=0,
                raw=u.model_dump() if u else {},
            ),
        )
