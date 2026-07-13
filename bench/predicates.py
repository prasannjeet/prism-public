"""Named landed-violation predicates (spec section 6.3). These are the ADMISSIBLE
(enforce-sensitive) violations that can land in unenforced configs; structural ones reject in
both modes and so can never land. Each predicate: event_log -> bool. Strategy + registry
(charter section 3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

EventLog = Sequence[dict[str, object]]
Predicate = Callable[[EventLog], bool]


def _count_admitted(log: EventLog, reason: str) -> int:
    return sum(
        1 for r in log if r.get("type") == "violation_admitted" and r.get("reason") == reason
    )


def _count_commit_admitted(log: EventLog, reason: str) -> int:
    return sum(
        1 for r in log if r.get("type") == "commit_violation_admitted" and r.get("reason") == reason
    )


FORBIDDEN: dict[str, Predicate] = {
    "inactive_branch_contamination": lambda log: _count_admitted(log, "inactive_field") > 0,
    "landed_invalid_value": lambda log: _count_admitted(log, "invalid_value") > 0,
    "landed_conclusion_without_evidence": lambda log: _count_admitted(log, "evidence_missing") > 0,
    "landed_premature_commit": lambda log: _count_commit_admitted(log, "not_complete") > 0,
    "landed_unconfirmed_commit": lambda log: _count_commit_admitted(log, "not_confirmed") > 0,
    # Idempotent commit yields commit_noop in both modes, so a duplicate can never "land".
    "landed_duplicate_commit": lambda log: False,
}

__all__ = ["EventLog", "FORBIDDEN", "Predicate"]
