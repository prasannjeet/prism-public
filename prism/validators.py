"""Single-field validation: strategy + registry. Pure, locale-free; sits below the gate."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from .schema import Field, FieldType, Workflow, WorkflowConfigError, load_workflow


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    code: str | None = None
    message: str | None = None

    @classmethod
    def success(cls) -> ValidationResult:
        return cls(ok=True)

    @classmethod
    def failure(cls, code: str, message: str) -> ValidationResult:
        return cls(ok=False, code=code, message=message)


class Validator(Protocol):
    def validate(self, value: object, rule_value: object, *, today: date) -> ValidationResult: ...


_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _as_date(value: object) -> date:
    # datetime is a date subclass — check it first.
    if isinstance(value, datetime):
        return value.date()
    return cast(date, value)


class _Length:
    def validate(self, value: object, rule_value: object, *, today: date) -> ValidationResult:
        bounds = cast(Mapping[str, int], rule_value)  # shape guaranteed at load time
        text = cast(str, value)
        minimum, maximum = bounds.get("min"), bounds.get("max")
        if minimum is not None and len(text) < minimum:
            return ValidationResult.failure(
                "validation.length", f"Must be at least {minimum} characters."
            )
        if maximum is not None and len(text) > maximum:
            return ValidationResult.failure(
                "validation.length", f"Must be at most {maximum} characters."
            )
        return ValidationResult.success()


class _Range:
    def validate(self, value: object, rule_value: object, *, today: date) -> ValidationResult:
        bounds = cast(Mapping[str, float], rule_value)
        number = cast(float, value)
        minimum, maximum = bounds.get("min"), bounds.get("max")
        if minimum is not None and maximum is not None and not minimum <= number <= maximum:
            return ValidationResult.failure(
                "validation.range", f"Must be between {minimum} and {maximum}."
            )
        if minimum is not None and number < minimum:
            return ValidationResult.failure("validation.range", f"Must be at least {minimum}.")
        if maximum is not None and number > maximum:
            return ValidationResult.failure("validation.range", f"Must be at most {maximum}.")
        return ValidationResult.success()


class _Pattern:
    def validate(self, value: object, rule_value: object, *, today: date) -> ValidationResult:
        text = cast(str, value)
        spec = cast(str, rule_value)
        if spec == "email":
            if _EMAIL_PATTERN.fullmatch(text) is None:
                return ValidationResult.failure(
                    "validation.email", "Must be a valid email address."
                )
            return ValidationResult.success()
        if re.fullmatch(spec, text) is None:
            return ValidationResult.failure(
                "validation.pattern", f"Must match the pattern '{spec}'."
            )
        return ValidationResult.success()


class _NotFuture:
    def validate(self, value: object, rule_value: object, *, today: date) -> ValidationResult:
        if _as_date(value) > today:
            return ValidationResult.failure(
                "validation.not_future", "Date cannot be in the future."
            )
        return ValidationResult.success()


class _FutureWithinDays:
    def validate(self, value: object, rule_value: object, *, today: date) -> ValidationResult:
        days = cast(int, rule_value)
        when = _as_date(value)
        if when < today:
            return ValidationResult.failure(
                "validation.future_within_days", "Date cannot be in the past."
            )
        if when > today + timedelta(days=days):
            return ValidationResult.failure(
                "validation.future_within_days", f"Date must be within {days} days from today."
            )
        return ValidationResult.success()


VALIDATORS: Mapping[str, Validator] = MappingProxyType(
    {
        "length": _Length(),
        "range": _Range(),
        "pattern": _Pattern(),
        "not_future": _NotFuture(),
        "future_within_days": _FutureWithinDays(),
    }
)

KNOWN_RULES: frozenset[str] = frozenset(VALIDATORS)


def _is_empty(raw: object) -> bool:
    return raw is None or (isinstance(raw, str) and not raw.strip())


def _coerce(field: Field, raw: object) -> tuple[object, ValidationResult]:
    """Per field type: parse + structural check. The stored answer stays raw (D-Values)."""
    text = str(raw).strip()
    match field.type:
        case FieldType.NUMBER:
            try:
                number = float(text)
            except ValueError:
                return raw, ValidationResult.failure("validation.type", "Must be a number.")
            if not isfinite(number):
                return raw, ValidationResult.failure("validation.type", "Must be a number.")
            return number, ValidationResult.success()
        case FieldType.DATE:
            try:
                return date.fromisoformat(text), ValidationResult.success()
            except ValueError:
                return raw, ValidationResult.failure(
                    "validation.type", "Must be a date in YYYY-MM-DD format."
                )
        case FieldType.DATETIME:
            try:
                return datetime.fromisoformat(text), ValidationResult.success()
            except ValueError:
                return raw, ValidationResult.failure(
                    "validation.type", "Must be a date and time in ISO format."
                )
        case FieldType.EMAIL:
            if _EMAIL_PATTERN.fullmatch(text) is None:
                return raw, ValidationResult.failure(
                    "validation.email", "Must be a valid email address."
                )
            return text, ValidationResult.success()
        case FieldType.SELECT:
            if text not in field.options:
                return raw, ValidationResult.failure(
                    "validation.option", f"Must be one of: {', '.join(field.options)}."
                )
            return text, ValidationResult.success()
        case _:
            return str(raw), ValidationResult.success()


def canonical_value(field: Field, raw: object) -> object:
    """Canonical stored form for a validated answer: ISO string for DATE/DATETIME, raw otherwise.

    Pure string parsing (no clock, no I/O). DATE/DATETIME accept a space or `T` separator on
    input; re-emitting via `isoformat` makes the stored value representation-independent, so the
    outcome scorer's structured exact-match holds regardless of what the model typed. Every other
    type (and empty/None) is returned unchanged — canonicalizing NUMBER/TEXT/SELECT/EMAIL would
    risk altering values the scorer compares verbatim (e.g. `12` vs `12.0`).
    """
    if _is_empty(raw) or not isinstance(raw, str):
        return raw
    text = raw.strip()
    match field.type:
        case FieldType.DATE:
            return date.fromisoformat(text).isoformat()
        case FieldType.DATETIME:
            return datetime.fromisoformat(text).isoformat()
        case _:
            return raw


def validate_field(field: Field, raw_value: object, *, today: date) -> ValidationResult:
    """The gate's single entry point: presence, then coercion, then rules; first failure wins."""
    if _is_empty(raw_value):
        if field.required:
            return ValidationResult.failure("validation.required", "This field is required.")
        return ValidationResult.success()  # empty optional: nothing to check
    value, coercion = _coerce(field, raw_value)
    if not coercion.ok:
        return coercion
    for rule_name, rule_value in field.validators.items():  # YAML order — deterministic
        result = VALIDATORS[rule_name].validate(value, rule_value, today=today)
        if not result.ok:
            return result
    return ValidationResult.success()


_RULE_FIELD_TYPES: Mapping[str, frozenset[FieldType]] = MappingProxyType(
    {
        "length": frozenset({FieldType.TEXT, FieldType.EMAIL, FieldType.CONCLUSION}),
        "range": frozenset({FieldType.NUMBER}),
        "pattern": frozenset({FieldType.TEXT, FieldType.EMAIL}),
        "not_future": frozenset({FieldType.DATE, FieldType.DATETIME}),
        "future_within_days": frozenset({FieldType.DATE, FieldType.DATETIME}),
    }
)


def validate_workflow_rules(workflow: Workflow) -> None:
    """Fail-fast check of every declared rule: known name, applicable type, well-shaped value."""
    for fld in workflow.fields:
        for rule_name, rule_value in fld.validators.items():
            if rule_name not in KNOWN_RULES:
                raise WorkflowConfigError(
                    f"field '{fld.name}': unknown validation rule '{rule_name}' "
                    f"(known: {', '.join(sorted(KNOWN_RULES))})"
                )
            if fld.type not in _RULE_FIELD_TYPES[rule_name]:
                raise WorkflowConfigError(
                    f"field '{fld.name}': rule '{rule_name}' does not apply to "
                    f"type '{fld.type.value}'"
                )
            _check_rule_value(fld.name, rule_name, rule_value)


def _check_rule_value(field_name: str, rule_name: str, rule_value: object) -> None:
    if rule_name in ("length", "range"):
        if (
            not isinstance(rule_value, Mapping)
            or not rule_value
            or not set(rule_value) <= {"min", "max"}
        ):
            raise WorkflowConfigError(
                f"field '{field_name}': '{rule_name}' needs a mapping with 'min' and/or 'max'"
            )
        for key, bound in rule_value.items():
            if isinstance(bound, bool) or not isinstance(bound, int | float):
                raise WorkflowConfigError(
                    f"field '{field_name}': '{rule_name}.{key}' must be a number"
                )
        if "min" in rule_value and "max" in rule_value:
            lo, hi = rule_value["min"], rule_value["max"]
            if isinstance(lo, int | float) and isinstance(hi, int | float) and lo > hi:
                raise WorkflowConfigError(
                    f"field '{field_name}': '{rule_name}.min' ({lo}) exceeds 'max' ({hi})"
                )
    elif rule_name == "pattern":
        if not isinstance(rule_value, str) or not rule_value:
            raise WorkflowConfigError(f"field '{field_name}': 'pattern' must be a non-empty string")
        if rule_value != "email":
            try:
                re.compile(rule_value)
            except re.error as exc:
                raise WorkflowConfigError(
                    f"field '{field_name}': invalid 'pattern' regex: {exc}"
                ) from None
    elif rule_name == "not_future":
        if rule_value is not True:
            raise WorkflowConfigError(
                f"field '{field_name}': 'not_future' must be true "
                f"(omit the rule instead of setting false)"
            )
    elif rule_name == "future_within_days":
        if isinstance(rule_value, bool) or not isinstance(rule_value, int) or rule_value <= 0:
            raise WorkflowConfigError(
                f"field '{field_name}': 'future_within_days' must be a positive integer"
            )


def load_validated_workflow(path: str | Path) -> Workflow:
    """Loader entry point for runner/tests: schema load + rule check, both fail-fast."""
    workflow = load_workflow(path)
    validate_workflow_rules(workflow)
    return workflow
