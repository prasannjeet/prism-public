"""Deterministic state engine: active path, branch lifecycle, completion.

Pure functions over an immutable State — same inputs always yield the same outputs.
This module computes the evaluation ground truth, so it has no I/O, no clock, no randomness.
State transitions return a new State (never mutate); branch cleanup cascades to a fixpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .schema import AllOf, AnyOf, CompareOp, Field, FieldType, Leaf, Workflow, coerce_scalar


def _compare(field_type: FieldType, op: CompareOp, answer: str, value: str) -> bool:
    if op is CompareOp.EQ:
        return answer == value
    if op is CompareOp.NE:
        return answer != value
    left, right = coerce_scalar(field_type, answer), coerce_scalar(field_type, value)
    if left is None or right is None or type(left) is not type(right):
        return False
    return _compare_ordered(op, left, right)


def _compare_ordered(op: CompareOp, left: Any, right: Any) -> bool:
    # left and right are the same orderable type (both float, both date, or both datetime).
    if op is CompareOp.GT:
        return bool(left > right)
    if op is CompareOp.GE:
        return bool(left >= right)
    if op is CompareOp.LT:
        return bool(left < right)
    return bool(left <= right)  # CompareOp.LE


class UnknownFieldError(Exception):
    """An answer or evidence call referenced a field that is not in the workflow."""


class FieldKindError(Exception):
    """A transition was applied to the wrong kind of field (e.g. evidence on a text field)."""


@dataclass(frozen=True)
class State:
    answers: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    evidence: frozenset[str] = frozenset()
    confirmed: bool = False
    submitted: bool = False
    escalated: bool = False


class StateEngine:
    """Deterministic projection of workflow + answers onto active path and completion."""

    def __init__(self, workflow: Workflow) -> None:
        self._workflow = workflow

    @property
    def workflow(self) -> Workflow:
        return self._workflow

    def initial_state(self) -> State:
        return State()

    # --- active path ---------------------------------------------------------

    def is_active(self, fld: Field, answers: Mapping[str, Any]) -> bool:
        """A field is active iff it has no guard, or its guard evaluates true under the
        current answers. Recursion handles nested branches; load-time acyclicity terminates."""
        if fld.show_when is None:
            return True
        return self._eval(fld.show_when, answers)

    def _eval(self, cond: object, answers: Mapping[str, Any]) -> bool:
        match cond:
            case AllOf(conditions=cs):
                return all(self._eval(c, answers) for c in cs)
            case AnyOf(conditions=cs):
                return any(self._eval(c, answers) for c in cs)
            case Leaf():
                return self._eval_leaf(cond, answers)
        return False

    def _eval_leaf(self, leaf: Leaf, answers: Mapping[str, Any]) -> bool:
        referenced = self._workflow.field(leaf.field)
        if not self.is_active(referenced, answers):
            return False
        answer = answers.get(leaf.field)
        if answer is None:
            return False
        return _compare(referenced.type, leaf.op, str(answer), leaf.value)

    def active_fields(self, answers: Mapping[str, Any]) -> tuple[Field, ...]:
        return tuple(f for f in self._workflow.fields if self.is_active(f, answers))

    # --- transitions (return a new State) ------------------------------------

    def with_answer(self, state: State, name: str, value: Any) -> State:
        if not self._workflow.has_field(name):
            raise UnknownFieldError(name)
        answers = dict(state.answers)
        answers[name] = value
        pruned_answers, pruned_evidence = self._prune(answers, state.evidence)
        return replace(state, answers=MappingProxyType(pruned_answers), evidence=pruned_evidence)

    def with_evidence(self, state: State, name: str) -> State:
        if not self._workflow.has_field(name):
            raise UnknownFieldError(name)
        if self._workflow.field(name).type is not FieldType.EVIDENCE:
            raise FieldKindError(f"field '{name}' is not an evidence field")
        evidence = set(state.evidence)
        evidence.add(name)
        pruned_answers, pruned_evidence = self._prune(dict(state.answers), frozenset(evidence))
        return replace(state, answers=MappingProxyType(pruned_answers), evidence=pruned_evidence)

    def with_confirmation(self, state: State) -> State:
        return replace(state, confirmed=True)

    def with_submission(self, state: State) -> State:
        return replace(state, submitted=True)

    def with_escalation(self, state: State) -> State:
        return replace(state, escalated=True)

    # --- cascading branch cleanup --------------------------------------------

    def _prune(
        self, answers: Mapping[str, Any], evidence: frozenset[str]
    ) -> tuple[dict[str, Any], frozenset[str]]:
        """Drop answers/evidence for fields no longer on the active path, iterating to a
        fixpoint (dropping one answer can deactivate a branch that gated further fields)."""
        current_answers = dict(answers)
        current_evidence = evidence
        while True:
            active = {f.name for f in self.active_fields(current_answers)}
            next_answers = {k: v for k, v in current_answers.items() if k in active}
            next_evidence = frozenset(e for e in current_evidence if e in active)
            unchanged = len(next_answers) == len(current_answers) and len(next_evidence) == len(
                current_evidence
            )
            if unchanged:
                return next_answers, next_evidence
            current_answers, current_evidence = next_answers, next_evidence

    # --- completion ----------------------------------------------------------

    def is_satisfied(self, fld: Field, state: State) -> bool:
        """Whether a single active field has met its requirement."""
        if fld.type is FieldType.EVIDENCE:
            return fld.name in state.evidence
        if fld.type is FieldType.CONCLUSION:
            answered = state.answers.get(fld.name) not in (None, "")
            return answered and all(ev in state.evidence for ev in fld.requires_evidence)
        return state.answers.get(fld.name) not in (None, "")

    def remaining_mandatory(self, state: State) -> tuple[Field, ...]:
        return tuple(
            f
            for f in self.active_fields(state.answers)
            if f.required and not self.is_satisfied(f, state)
        )

    def is_complete(self, state: State) -> bool:
        """Data completeness only. Confirmation/submission are gate concerns, not this."""
        return not self.remaining_mandatory(state)
