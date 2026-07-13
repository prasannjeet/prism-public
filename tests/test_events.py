"""Event model + sink tests — exact envelopes, monotonic seq, byte-identical output."""

from __future__ import annotations

from io import StringIO

from prism.events import Event, EventType, JsonlEventSink, MemoryEventSink


def _events() -> list[Event]:
    return [
        Event(EventType.ANSWER_SUBMITTED, 1, {"field": "symptom", "value": "down"}),
        Event(EventType.BRANCH_ACTIVATED, 1, {"field": "outage_time"}),
        Event(EventType.COMMIT_NOOP, 2, {}),
    ]


def test_memory_sink_records_exact_envelopes_with_monotonic_seq() -> None:
    sink = MemoryEventSink()
    for event in _events():
        sink.emit(event)
    assert sink.records == [
        {"seq": 0, "turn": 1, "type": "answer_submitted", "field": "symptom", "value": "down"},
        {"seq": 1, "turn": 1, "type": "branch_activated", "field": "outage_time"},
        {"seq": 2, "turn": 2, "type": "commit_noop"},
    ]


def test_envelope_key_order_is_seq_turn_type_then_payload() -> None:
    sink = MemoryEventSink()
    sink.emit(Event(EventType.ANSWER_SUBMITTED, 1, {"field": "symptom", "value": "down"}))
    assert list(sink.records[0]) == ["seq", "turn", "type", "field", "value"]


def test_jsonl_sink_writes_one_line_per_event() -> None:
    buffer = StringIO()
    sink = JsonlEventSink(buffer)
    for event in _events():
        sink.emit(event)
    assert buffer.getvalue() == (
        '{"seq": 0, "turn": 1, "type": "answer_submitted", "field": "symptom", "value": "down"}\n'
        '{"seq": 1, "turn": 1, "type": "branch_activated", "field": "outage_time"}\n'
        '{"seq": 2, "turn": 2, "type": "commit_noop"}\n'
    )


def test_jsonl_sink_output_is_byte_identical_across_runs() -> None:
    first, second = StringIO(), StringIO()
    for buffer in (first, second):
        sink = JsonlEventSink(buffer)
        for event in _events():
            sink.emit(event)
    assert first.getvalue() == second.getvalue()


def test_jsonl_sink_preserves_non_ascii() -> None:
    buffer = StringIO()
    event = Event(EventType.ANSWER_SUBMITTED, 1, {"field": "city", "value": "Göteborg"})
    JsonlEventSink(buffer).emit(event)
    assert buffer.getvalue() == (
        '{"seq": 0, "turn": 1, "type": "answer_submitted", "field": "city", "value": "Göteborg"}\n'
    )


def test_conversation_lifecycle_types_serialize() -> None:
    sink = MemoryEventSink()
    sink.emit(Event(EventType.USER_MESSAGE, 1, {"text": "hi"}))
    sink.emit(
        Event(
            EventType.TOOL_CALLED,
            1,
            {
                "tool": "submit_answer",
                "arguments": {"field": "x"},
            },
        )
    )
    sink.emit(Event(EventType.TOOL_RESULT, 1, {"tool": "submit_answer", "content": "ok"}))
    sink.emit(Event(EventType.TOOL_ERROR, 1, {"tool": "bogus", "reason": "unknown_tool"}))
    sink.emit(Event(EventType.ASSISTANT_MESSAGE, 1, {"text": "done"}))
    sink.emit(Event(EventType.ESCALATED, 2, {"reason": "evidence_insufficient"}))
    assert [r["type"] for r in sink.records] == [
        "user_message",
        "tool_called",
        "tool_result",
        "tool_error",
        "assistant_message",
        "escalated",
    ]
