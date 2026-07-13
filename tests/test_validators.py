"""Validator tests — every rule's pass and fail branches, exact codes and messages."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import MappingProxyType

import pytest

from prism.schema import Field, FieldType, WorkflowConfigError, parse_workflow
from prism.validators import (
    ValidationResult,
    load_validated_workflow,
    validate_field,
    validate_workflow_rules,
)

_TODAY = date(2026, 1, 15)
WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"


def _field(
    field_type: FieldType,
    *,
    required: bool = True,
    options: tuple[str, ...] = (),
    validators: dict[str, object] | None = None,
) -> Field:
    return Field(
        name="probe",
        type=field_type,
        required=required,
        options=options,
        validators=MappingProxyType(validators or {}),
    )


def _check(result: ValidationResult, code: str, message: str) -> None:
    assert (result.ok, result.code, result.message) == (False, code, message)


# --- required / empty ---------------------------------------------------------


def test_required_rejects_none() -> None:
    result = validate_field(_field(FieldType.TEXT), None, today=_TODAY)
    _check(result, "validation.required", "This field is required.")


def test_required_rejects_blank_string() -> None:
    result = validate_field(_field(FieldType.TEXT), "   ", today=_TODAY)
    _check(result, "validation.required", "This field is required.")


def test_empty_optional_short_circuits_to_success() -> None:
    # An empty optional number must NOT fall through to coercion ("" is not a number).
    field = _field(FieldType.NUMBER, required=False, validators={"range": {"min": 1}})
    assert validate_field(field, "", today=_TODAY) == ValidationResult.success()


def test_text_with_no_rules_passes() -> None:
    assert validate_field(_field(FieldType.TEXT), "anything", today=_TODAY).ok is True


# --- coercion -----------------------------------------------------------------


def test_number_coercion_rejects_non_numeric() -> None:
    result = validate_field(_field(FieldType.NUMBER), "abc", today=_TODAY)
    _check(result, "validation.type", "Must be a number.")


def test_number_coercion_rejects_non_finite() -> None:
    for raw in ("nan", "inf"):
        result = validate_field(_field(FieldType.NUMBER), raw, today=_TODAY)
        _check(result, "validation.type", "Must be a number.")


def test_number_coercion_accepts_numeric_string() -> None:
    field = _field(FieldType.NUMBER, validators={"range": {"min": 1, "max": 5}})
    assert validate_field(field, "3.5", today=_TODAY).ok is True


def test_date_coercion_rejects_malformed() -> None:
    result = validate_field(_field(FieldType.DATE), "Jan 10", today=_TODAY)
    _check(result, "validation.type", "Must be a date in YYYY-MM-DD format.")


def test_datetime_coercion_rejects_malformed() -> None:
    result = validate_field(_field(FieldType.DATETIME), "yesterday", today=_TODAY)
    _check(result, "validation.type", "Must be a date and time in ISO format.")


def test_select_rejects_value_not_in_options() -> None:
    field = _field(FieldType.SELECT, options=("down", "slow"))
    result = validate_field(field, "banana", today=_TODAY)
    _check(result, "validation.option", "Must be one of: down, slow.")


def test_select_accepts_declared_option() -> None:
    field = _field(FieldType.SELECT, options=("down", "slow"))
    assert validate_field(field, "down", today=_TODAY).ok is True


def test_email_field_type_rejects_bad_shape() -> None:
    result = validate_field(_field(FieldType.EMAIL), "nope", today=_TODAY)
    _check(result, "validation.email", "Must be a valid email address.")


def test_email_field_type_accepts_valid_address() -> None:
    assert validate_field(_field(FieldType.EMAIL), "user@example.com", today=_TODAY).ok is True


# --- length -------------------------------------------------------------------


def test_length_min_fails_below_bound() -> None:
    field = _field(FieldType.TEXT, validators={"length": {"min": 5}})
    _check(
        validate_field(field, "abcd", today=_TODAY),
        "validation.length",
        "Must be at least 5 characters.",
    )


def test_length_max_fails_above_bound() -> None:
    field = _field(FieldType.TEXT, validators={"length": {"max": 3}})
    _check(
        validate_field(field, "abcd", today=_TODAY),
        "validation.length",
        "Must be at most 3 characters.",
    )


def test_length_within_bounds_passes() -> None:
    field = _field(FieldType.TEXT, validators={"length": {"min": 2, "max": 5}})
    assert validate_field(field, "abc", today=_TODAY).ok is True


# --- range --------------------------------------------------------------------


def test_range_uses_between_message_when_both_bounds_set() -> None:
    field = _field(FieldType.NUMBER, validators={"range": {"min": 1, "max": 5}})
    _check(
        validate_field(field, "7", today=_TODAY),
        "validation.range",
        "Must be between 1 and 5.",
    )


def test_range_min_only_message() -> None:
    field = _field(FieldType.NUMBER, validators={"range": {"min": 3}})
    _check(validate_field(field, "2", today=_TODAY), "validation.range", "Must be at least 3.")


def test_range_max_only_message() -> None:
    field = _field(FieldType.NUMBER, validators={"range": {"max": 5}})
    _check(validate_field(field, "9", today=_TODAY), "validation.range", "Must be at most 5.")


def test_range_in_bounds_passes() -> None:
    field = _field(FieldType.NUMBER, validators={"range": {"min": 1, "max": 5}})
    assert validate_field(field, "5", today=_TODAY).ok is True


# --- pattern ------------------------------------------------------------------


def test_pattern_regex_fail() -> None:
    field = _field(FieldType.TEXT, validators={"pattern": r"\d{4}"})
    _check(
        validate_field(field, "12a4", today=_TODAY),
        "validation.pattern",
        r"Must match the pattern '\d{4}'.",
    )


def test_pattern_regex_pass() -> None:
    field = _field(FieldType.TEXT, validators={"pattern": r"\d{4}"})
    assert validate_field(field, "1234", today=_TODAY).ok is True


def test_pattern_email_alias_fail() -> None:
    field = _field(FieldType.TEXT, validators={"pattern": "email"})
    _check(
        validate_field(field, "not-an-email", today=_TODAY),
        "validation.email",
        "Must be a valid email address.",
    )


def test_pattern_email_alias_pass() -> None:
    field = _field(FieldType.TEXT, validators={"pattern": "email"})
    assert validate_field(field, "a@b.se", today=_TODAY).ok is True


# --- date window --------------------------------------------------------------


def test_not_future_accepts_today_and_past() -> None:
    field = _field(FieldType.DATE, validators={"not_future": True})
    assert validate_field(field, "2026-01-15", today=_TODAY).ok is True
    assert validate_field(field, "2025-12-31", today=_TODAY).ok is True


def test_not_future_rejects_tomorrow() -> None:
    field = _field(FieldType.DATE, validators={"not_future": True})
    _check(
        validate_field(field, "2026-01-16", today=_TODAY),
        "validation.not_future",
        "Date cannot be in the future.",
    )


def test_not_future_on_datetime_compares_date_part() -> None:
    field = _field(FieldType.DATETIME, validators={"not_future": True})
    assert validate_field(field, "2026-01-15 23:59", today=_TODAY).ok is True


def test_future_within_days_accepts_today_and_window_boundary() -> None:
    field = _field(FieldType.DATE, validators={"future_within_days": 30})
    assert validate_field(field, "2026-01-15", today=_TODAY).ok is True
    assert validate_field(field, "2026-02-14", today=_TODAY).ok is True


def test_future_within_days_rejects_past() -> None:
    field = _field(FieldType.DATE, validators={"future_within_days": 30})
    _check(
        validate_field(field, "2026-01-14", today=_TODAY),
        "validation.future_within_days",
        "Date cannot be in the past.",
    )


def test_future_within_days_rejects_beyond_window() -> None:
    field = _field(FieldType.DATE, validators={"future_within_days": 30})
    _check(
        validate_field(field, "2026-02-15", today=_TODAY),
        "validation.future_within_days",
        "Date must be within 30 days from today.",
    )


# --- runner -------------------------------------------------------------------


def test_first_failing_rule_wins_in_declaration_order() -> None:
    field = _field(FieldType.TEXT, validators={"length": {"min": 5}, "pattern": r"\d+"})
    result = validate_field(field, "ab", today=_TODAY)  # violates both rules
    assert result.code == "validation.length"


# --- load-time rule checking ----------------------------------------------------


def _workflow_data(field_type: str, validators: dict[str, object]) -> dict[str, object]:
    return {
        "workflow": "probe",
        "fields": [{"name": "probe", "type": field_type, "required": True, "validate": validators}],
    }


def test_unknown_rule_name_raises() -> None:
    data = _workflow_data("number", {"rnage": {"min": 1}})
    with pytest.raises(WorkflowConfigError, match="unknown validation rule 'rnage'"):
        validate_workflow_rules(parse_workflow(data))


def test_rule_not_applicable_to_field_type_raises() -> None:
    data = _workflow_data("text", {"range": {"min": 1}})
    with pytest.raises(WorkflowConfigError, match="does not apply to type 'text'"):
        validate_workflow_rules(parse_workflow(data))


def test_length_rule_with_wrong_shape_raises() -> None:
    for bad in ({}, {"minimum": 3}, "5"):
        data = _workflow_data("text", {"length": bad})
        with pytest.raises(WorkflowConfigError, match="'min' and/or 'max'"):
            validate_workflow_rules(parse_workflow(data))


def test_length_rule_with_non_numeric_bound_raises() -> None:
    data = _workflow_data("text", {"length": {"min": "five"}})
    with pytest.raises(WorkflowConfigError, match="'length.min' must be a number"):
        validate_workflow_rules(parse_workflow(data))


def test_pattern_rule_with_invalid_regex_raises() -> None:
    data = _workflow_data("text", {"pattern": "("})
    with pytest.raises(WorkflowConfigError, match="invalid 'pattern' regex"):
        validate_workflow_rules(parse_workflow(data))


def test_pattern_rule_with_non_string_raises() -> None:
    data = _workflow_data("text", {"pattern": 5})
    with pytest.raises(WorkflowConfigError, match="'pattern' must be a non-empty string"):
        validate_workflow_rules(parse_workflow(data))


def test_not_future_must_be_true() -> None:
    data = _workflow_data("date", {"not_future": False})
    with pytest.raises(WorkflowConfigError, match="'not_future' must be true"):
        validate_workflow_rules(parse_workflow(data))


def test_future_within_days_must_be_positive_int() -> None:
    for bad in (0, -3, True, "30"):
        data = _workflow_data("date", {"future_within_days": bad})
        with pytest.raises(WorkflowConfigError, match="positive integer"):
            validate_workflow_rules(parse_workflow(data))


def test_well_formed_rules_pass() -> None:
    data = _workflow_data("date", {"not_future": True, "future_within_days": 30})
    assert validate_workflow_rules(parse_workflow(data)) is None


def test_load_validated_workflow_accepts_committed_incident_yaml() -> None:
    workflow = load_validated_workflow(WORKFLOWS / "incident_investigation.yaml")
    assert workflow.name == "incident-investigation"
    assert dict(workflow.field("outage_time").validators) == {"not_future": True}


def test_range_min_greater_than_max_rejected() -> None:
    wf = parse_workflow(
        {
            "workflow": "w",
            "profile": "express",
            "fields": [
                {
                    "name": "n",
                    "type": "number",
                    "required": True,
                    "validate": {"range": {"min": 5, "max": 1}},
                }
            ],
        }
    )
    with pytest.raises(WorkflowConfigError, match="exceeds 'max'"):
        validate_workflow_rules(wf)
