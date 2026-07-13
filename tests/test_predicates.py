"""Each forbidden predicate is a pure function over the event log, keyed on the admitted reason."""

from __future__ import annotations

from bench.predicates import FORBIDDEN


def _admitted(reason: str) -> dict[str, object]:
    return {
        "seq": 0,
        "turn": 1,
        "type": "violation_admitted",
        "gate": "answer",
        "field": "x",
        "value": "v",
        "reason": reason,
        "code": None,
    }


def _commit_admitted(reason: str) -> dict[str, object]:
    return {
        "seq": 0,
        "turn": 1,
        "type": "commit_violation_admitted",
        "reason": reason,
        "remaining": [],
    }


def test_contamination_true_only_with_admitted_inactive() -> None:
    assert FORBIDDEN["inactive_branch_contamination"]([_admitted("inactive_field")]) is True
    assert FORBIDDEN["inactive_branch_contamination"]([_admitted("invalid_value")]) is False
    assert FORBIDDEN["inactive_branch_contamination"]([]) is False


def test_landed_premature_and_unconfirmed_commit() -> None:
    assert FORBIDDEN["landed_premature_commit"]([_commit_admitted("not_complete")]) is True
    assert FORBIDDEN["landed_premature_commit"]([_commit_admitted("not_confirmed")]) is False
    assert FORBIDDEN["landed_unconfirmed_commit"]([_commit_admitted("not_confirmed")]) is True


def test_landed_invalid_and_evidence_missing() -> None:
    assert FORBIDDEN["landed_invalid_value"]([_admitted("invalid_value")]) is True
    assert FORBIDDEN["landed_conclusion_without_evidence"]([_admitted("evidence_missing")]) is True
    assert FORBIDDEN["landed_invalid_value"]([_admitted("evidence_missing")]) is False


def test_duplicate_commit_is_never_true_idempotent() -> None:
    assert FORBIDDEN["landed_duplicate_commit"]([_commit_admitted("not_complete")]) is False
