"""Structured event model + sinks. Pure models; the JSONL sink is the layer's only I/O edge."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TextIO


class EventType(StrEnum):
    ANSWER_SUBMITTED = "answer_submitted"
    ANSWER_REJECTED = "answer_rejected"
    EVIDENCE_REJECTED = "evidence_rejected"
    VIOLATION_ADMITTED = "violation_admitted"
    BRANCH_ACTIVATED = "branch_activated"
    BRANCH_CLEANED = "branch_cleaned"
    EVIDENCE_COLLECTED = "evidence_collected"
    CONFIRMATION_SET = "confirmation_set"
    CONFIRMATION_RESET = "confirmation_reset"
    COMMITTED = "committed"
    COMMIT_BLOCKED = "commit_blocked"
    COMMIT_VIOLATION_ADMITTED = "commit_violation_admitted"
    COMMIT_NOOP = "commit_noop"
    # Conversation / tool lifecycle (Phase 1.2 turn loop)
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class Event:
    type: EventType
    turn: int
    payload: Mapping[str, object]  # constructed with explicit, deterministic key order


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


def _envelope(seq: int, event: Event) -> dict[str, object]:
    # Fixed key order seq, turn, type, then the payload's own order — the byte-identical claim
    # rests on insertion order, so no sort_keys anywhere.
    record: dict[str, object] = {"seq": seq, "turn": event.turn, "type": event.type.value}
    record.update(event.payload)
    return record


class MemoryEventSink:
    """Collects envelopes in memory so tests can assert exact sequences without disk."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def emit(self, event: Event) -> None:
        self.records.append(_envelope(len(self.records), event))


class JsonlEventSink:
    """One JSON line per event to a caller-owned stream — opening the file stays at the runner."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._seq = 0

    def emit(self, event: Event) -> None:
        self._stream.write(json.dumps(_envelope(self._seq, event), ensure_ascii=False) + "\n")
        self._seq += 1
