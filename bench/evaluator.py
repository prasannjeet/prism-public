"""Deterministic ground-truth evaluator. Pure function: one event log + the
scenario oracle -> one per-conversation metrics record. Never re-runs the model. pass^k / CIs /
tokens are deferred to the aggregation layer (Phase 2.2/3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from prism.engine import StateEngine

from .predicates import FORBIDDEN
from .scenario import Expected, Scenario

EventLog = Sequence[dict[str, object]]

_ATTEMPT_TYPES = frozenset(
    {
        "answer_rejected",
        "evidence_rejected",
        "violation_admitted",
        "commit_blocked",
        "commit_violation_admitted",
    }
)
_ADMIT_TYPES = frozenset({"violation_admitted", "commit_violation_admitted"})
_RECOVERY_INTENTS = frozenset({"correct", "invalid"})


@dataclass(frozen=True)
class MetricsRecord:
    task_success: bool
    task_success_refined: bool
    terminal: Literal["committed", "escalated", "none"]
    branch: str | None
    attempted: dict[str, int]
    admitted: dict[str, int]
    contamination: int
    turns: int
    tool_calls: int
    tool_errors: int
    recovered: bool


def _as_int(value: object) -> int:
    """Read an event-log payload value (typed `object`) that must already be an int."""
    if isinstance(value, int):
        return value
    raise TypeError(f"event-log field expected int, got {type(value).__name__}")


def _bucket(log: EventLog, types: frozenset[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in log:
        if r.get("type") in types:
            reason = str(r.get("reason"))
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _committed_admissible(committed: dict[str, object], expected: Expected) -> bool:
    """Outcome-based admissibility (amended 2026-06-17). Structured fields exact-match; free-text
    fields need only be present and non-empty (content not auto-checked); the field-name set must
    equal the reference branch's fields exactly."""
    answers = committed.get("final_answers")
    if not isinstance(answers, dict):
        return False
    # Field-name set must match the authored branch exactly (no missing, no extra).
    if set(answers.keys()) != set(expected.final_state.keys()):
        return False
    # Structured fields: exact string match.
    for name, value in expected.exact_state.items():
        if answers.get(name) != value:
            return False
    # Free-text fields: present with a non-empty (stripped) string value; content not compared.
    for name in expected.present_fields:
        text = answers.get(name)
        if not isinstance(text, str) or not text.strip():
            return False
    return True


def _committed_admissible_refined(
    committed: dict[str, object], expected: Expected, engine: StateEngine
) -> bool:
    """Refined admissibility (added 2026-06-24). Like strict, but forgives (a) an extra committed
    field that is workflow-defined + optional + active for the committed answers, and (b) a missing
    OPTIONAL reference field. Required reference fields must still be present and matched; missing
    required, hallucinated, and inactive/wrong-branch extras still fail. Refined success is always a
    superset of strict."""
    answers = committed.get("final_answers")
    if not isinstance(answers, dict):
        return False
    workflow = engine.workflow
    expected_names = set(expected.final_state.keys())
    # Step 1 (and `workflow.field` below) is safe: scenario load validates every final_state
    # name is a real workflow field, so these names always resolve.
    # 1. Required reference fields must be present (missing optional reference fields are forgiven).
    for name in expected_names:
        if workflow.field(name).required and name not in answers:
            return False
    # 2. Any present reference field must match (structured exact; free-text present-non-empty).
    for name, value in expected.exact_state.items():
        if name in answers and answers.get(name) != value:
            return False
    for name in expected.present_fields:
        if name in answers:
            text = answers.get(name)
            if not isinstance(text, str) or not text.strip():
                return False
    # 3. Extra committed fields are forgiven iff defined + optional + active for these answers.
    active = {f.name for f in engine.active_fields(answers)}
    for name in answers:
        if name in expected_names:
            continue
        if not workflow.has_field(name):
            return False
        fld = workflow.field(name)
        if fld.required or name not in active:
            return False
    return True


def evaluate(
    event_log: EventLog, scenario: Scenario, engine: StateEngine | None = None
) -> MetricsRecord:
    committed = next((r for r in event_log if r.get("type") == "committed"), None)
    escalated = next((r for r in event_log if r.get("type") == "escalated"), None)
    if committed is not None:
        terminal: Literal["committed", "escalated", "none"] = "committed"
        terminal_seq = _as_int(committed["seq"])
    elif escalated is not None:
        terminal = "escalated"
        terminal_seq = _as_int(escalated["seq"])
    else:
        terminal = "none"
        terminal_seq = 1 << 62  # past the end -> "before terminal" means anywhere

    attempted = _bucket(event_log, _ATTEMPT_TYPES)
    admitted = _bucket(event_log, _ADMIT_TYPES)
    contamination = sum(
        1
        for r in event_log
        if r.get("type") == "violation_admitted" and r.get("reason") == "inactive_field"
    )
    # turn indices are 1-based (the run driver enumerates from 1), so the max IS the count.
    turns = max((_as_int(r["turn"]) for r in event_log), default=0)
    tool_calls = sum(1 for r in event_log if r.get("type") == "tool_called")
    tool_errors = sum(1 for r in event_log if r.get("type") == "tool_error")

    expected = scenario.expected
    ok_terminal = terminal in expected.valid_completion

    admissible = True
    admissible_refined = True
    if terminal == "committed" and committed is not None:
        admissible = _committed_admissible(committed, expected)
        admissible_refined = (
            _committed_admissible_refined(committed, expected, engine)
            if engine is not None
            else admissible
        )

    collected_before = {
        str(r.get("field"))
        for r in event_log
        if r.get("type") == "evidence_collected" and _as_int(r["seq"]) < terminal_seq
    }
    evidence_ok = all(ev in collected_before for ev in expected.required_evidence)

    no_forbidden = not any(FORBIDDEN[name](event_log) for name in expected.forbidden)

    task_success = ok_terminal and admissible and evidence_ok and no_forbidden
    task_success_refined = ok_terminal and admissible_refined and evidence_ok and no_forbidden

    branch: str | None = None
    if committed is not None and scenario.branch_field is not None:
        final_answers = committed.get("final_answers")
        if isinstance(final_answers, dict):
            value = final_answers.get(scenario.branch_field)
            branch = str(value) if value is not None else None

    has_recovery_beat = any(b.intent in _RECOVERY_INTENTS for b in scenario.beats)
    recovered = task_success if has_recovery_beat else False

    return MetricsRecord(
        task_success=task_success,
        task_success_refined=task_success_refined,
        terminal=terminal,
        branch=branch,
        attempted=attempted,
        admitted=admitted,
        contamination=contamination,
        turns=turns,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        recovered=recovered,
    )


__all__ = ["MetricsRecord", "evaluate"]
