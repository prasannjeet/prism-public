"""Loader + fail-fast validation tests. Every malformed-config branch must raise."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prism.schema import (
    AllOf,
    AnyOf,
    CompareOp,
    FieldType,
    Leaf,
    Profile,
    WorkflowConfigError,
    dump_workflow,
    load_workflow,
    parse_condition,
    parse_workflow,
    referenced_fields,
)

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"


def _valid() -> dict:
    return {
        "workflow": "incident",
        "profile": "express",
        "fields": [
            {"name": "symptom", "type": "select", "required": True, "options": ["a", "b"]},
            {"name": "when_a", "type": "text", "required": True, "show_when": "symptom = a"},
            {
                "name": "ev",
                "type": "evidence",
                "required": True,
                "show_when": "symptom = a",
                "satisfied_by": ["tool_one"],
            },
            {
                "name": "verdict",
                "type": "conclusion",
                "required": True,
                "show_when": "symptom = a",
                "requires_evidence": ["ev"],
            },
        ],
        "tools": [{"name": "tool_one"}],
    }


def test_parse_valid_workflow_builds_exact_model() -> None:
    wf = parse_workflow(_valid())
    assert wf.name == "incident"
    assert wf.profile is Profile.EXPRESS
    assert wf.field_names == ("symptom", "when_a", "ev", "verdict")
    assert wf.field("symptom").type is FieldType.SELECT
    assert wf.field("symptom").options == ("a", "b")
    assert wf.field("when_a").show_when == Leaf("symptom", CompareOp.EQ, "a")
    assert wf.field("ev").satisfied_by == ("tool_one",)
    assert wf.field("verdict").requires_evidence == ("ev",)
    assert wf.tool_names() == frozenset({"tool_one"})


def test_profile_defaults_to_verified() -> None:
    data = _valid()
    del data["profile"]
    assert parse_workflow(data).profile is Profile.VERIFIED


def test_loads_example_workflow_from_disk() -> None:
    wf = load_workflow(WORKFLOWS / "incident_investigation.yaml")
    assert wf.name == "incident-investigation"
    assert wf.field_names == (
        "affected_service",
        "symptom",
        "outage_time",
        "deploy_check",
        "log_check",
        "server_root_cause",
        "latency_window",
        "metrics_check",
        "latency_root_cause",
        "error_sample",
    )
    assert wf.escalation is not None
    assert wf.escalation.action == "escalate_to_engineer"


def test_missing_name_raises() -> None:
    data = _valid()
    del data["workflow"]
    with pytest.raises(WorkflowConfigError, match="workflow"):
        parse_workflow(data)


def test_empty_fields_raises() -> None:
    data = _valid()
    data["fields"] = []
    with pytest.raises(WorkflowConfigError, match="non-empty list"):
        parse_workflow(data)


def test_unknown_field_type_raises() -> None:
    data = _valid()
    data["fields"][0]["type"] = "telepathy"
    with pytest.raises(WorkflowConfigError, match="unknown type 'telepathy'"):
        parse_workflow(data)


def test_unknown_profile_raises() -> None:
    data = _valid()
    data["profile"] = "yolo"
    with pytest.raises(WorkflowConfigError, match="unknown profile 'yolo'"):
        parse_workflow(data)


def test_duplicate_field_name_raises() -> None:
    data = _valid()
    data["fields"].append({"name": "symptom", "type": "text"})
    with pytest.raises(WorkflowConfigError, match="duplicate field name 'symptom'"):
        parse_workflow(data)


def test_dangling_show_when_reference_raises() -> None:
    data = _valid()
    data["fields"][1]["show_when"] = "ghost = a"
    with pytest.raises(WorkflowConfigError, match="unknown field 'ghost'"):
        parse_workflow(data)


def test_show_when_value_not_an_option_raises() -> None:
    data = _valid()
    data["fields"][1]["show_when"] = "symptom = z"
    with pytest.raises(WorkflowConfigError, match="not an option"):
        parse_workflow(data)


def test_malformed_show_when_raises() -> None:
    data = _valid()
    data["fields"][1]["show_when"] = "symptom is a"
    with pytest.raises(WorkflowConfigError, match="must contain one of"):
        parse_workflow(data)


def test_show_when_cycle_raises() -> None:
    data = {
        "workflow": "cyclic",
        "fields": [
            {"name": "a", "type": "text", "show_when": "b = x"},
            {"name": "b", "type": "text", "show_when": "a = x"},
        ],
    }
    with pytest.raises(WorkflowConfigError, match="cycle"):
        parse_workflow(data)


def test_evidence_without_satisfied_by_raises() -> None:
    data = _valid()
    data["fields"][2]["satisfied_by"] = []
    with pytest.raises(WorkflowConfigError, match="must declare 'satisfied_by'"):
        parse_workflow(data)


def test_evidence_unknown_tool_raises() -> None:
    data = _valid()
    data["fields"][2]["satisfied_by"] = ["nonexistent_tool"]
    with pytest.raises(WorkflowConfigError, match="unknown tool 'nonexistent_tool'"):
        parse_workflow(data)


def test_requires_evidence_unknown_field_raises() -> None:
    data = _valid()
    data["fields"][3]["requires_evidence"] = ["ghost_evidence"]
    with pytest.raises(WorkflowConfigError, match="unknown field 'ghost_evidence'"):
        parse_workflow(data)


def test_requires_evidence_non_evidence_field_raises() -> None:
    data = _valid()
    data["fields"][3]["requires_evidence"] = ["when_a"]  # a text field, not evidence
    with pytest.raises(WorkflowConfigError, match="is not an evidence field"):
        parse_workflow(data)


def test_dump_workflow_round_trips_through_parse() -> None:
    original = parse_workflow(_valid())
    assert parse_workflow(yaml.safe_load(dump_workflow(original))) == original


def test_dump_workflow_is_byte_stable_and_declaration_ordered() -> None:
    workflow = load_workflow(WORKFLOWS / "incident_investigation.yaml")
    first, second = dump_workflow(workflow), dump_workflow(workflow)
    assert first == second
    data = yaml.safe_load(first)
    assert [f["name"] for f in data["fields"]] == [
        "affected_service",
        "symptom",
        "outage_time",
        "deploy_check",
        "log_check",
        "server_root_cause",
        "latency_window",
        "metrics_check",
        "latency_root_cause",
        "error_sample",
    ]
    assert data["fields"][2]["validate"] == {"not_future": True}
    assert data["escalation"] == {
        "when": "evidence_insufficient",
        "action": "escalate_to_engineer",
    }


def test_show_when_must_reference_a_select_field() -> None:
    bad = {
        "workflow": "w",
        "profile": "express",
        "fields": [
            {"name": "n", "type": "number", "required": True},
            {"name": "g", "type": "text", "required": False, "show_when": "n = 5"},
        ],
    }
    with pytest.raises(WorkflowConfigError, match="requires a select field"):
        parse_workflow(bad)


def test_conclusion_requiring_mutually_exclusive_evidence_is_rejected() -> None:
    bad = {
        "workflow": "w",
        "profile": "express",
        "tools": [{"name": "t", "description": "x"}],
        "fields": [
            {"name": "s", "type": "select", "required": True, "options": ["a", "b"]},
            {
                "name": "e",
                "type": "evidence",
                "required": True,
                "show_when": "s = a",
                "satisfied_by": ["t"],
            },
            {
                "name": "c",
                "type": "conclusion",
                "required": True,
                "show_when": "s = b",
                "requires_evidence": ["e"],
            },
        ],
    }
    with pytest.raises(WorkflowConfigError, match="can never be co-active"):
        parse_workflow(bad)


def test_parse_condition_string_equality() -> None:
    assert parse_condition("symptom = down", "f") == Leaf("symptom", CompareOp.EQ, "down")


def test_parse_condition_string_all_operators() -> None:
    assert parse_condition("age >= 18", "f") == Leaf("age", CompareOp.GE, "18")
    assert parse_condition("age > 18", "f") == Leaf("age", CompareOp.GT, "18")
    assert parse_condition("age <= 65", "f") == Leaf("age", CompareOp.LE, "65")
    assert parse_condition("age < 65", "f") == Leaf("age", CompareOp.LT, "65")
    assert parse_condition("tier != basic", "f") == Leaf("tier", CompareOp.NE, "basic")


def test_parse_condition_all_of_and_any_of() -> None:
    cond = parse_condition({"all_of": ["a = x", "b = y"]}, "f")
    assert cond == AllOf((Leaf("a", CompareOp.EQ, "x"), Leaf("b", CompareOp.EQ, "y")))
    cond2 = parse_condition({"any_of": ["a = x", {"all_of": ["b = y", "c = z"]}]}, "f")
    assert cond2 == AnyOf(
        (
            Leaf("a", CompareOp.EQ, "x"),
            AllOf((Leaf("b", CompareOp.EQ, "y"), Leaf("c", CompareOp.EQ, "z"))),
        )
    )


def test_parse_condition_in_sugar_desugars_to_any_of() -> None:
    cond = parse_condition({"field": "t", "in": ["a", "b"]}, "f")
    assert cond == AnyOf((Leaf("t", CompareOp.EQ, "a"), Leaf("t", CompareOp.EQ, "b")))


def test_parse_condition_explicit_leaf_mapping() -> None:
    assert parse_condition({"field": "age", "op": ">=", "value": 18}, "f") == Leaf(
        "age", CompareOp.GE, "18"
    )


def test_parse_condition_explicit_mapping_escapes_operator_valued_value() -> None:
    assert parse_condition({"field": "a", "op": "=", "value": "<="}, "f") == Leaf(
        "a", CompareOp.EQ, "<="
    )


def test_referenced_fields_collects_every_leaf_field() -> None:
    cond = parse_condition({"all_of": ["a = x", {"any_of": ["b = y", "c = z"]}]}, "f")
    assert referenced_fields(cond) == frozenset({"a", "b", "c"})


def test_parse_condition_rejects_empty_side() -> None:
    with pytest.raises(WorkflowConfigError, match="empty side"):
        parse_condition("a = ", "f")


def test_parse_condition_rejects_no_operator() -> None:
    with pytest.raises(WorkflowConfigError, match="must contain one of"):
        parse_condition("just text", "f")


def test_parse_condition_rejects_empty_combinator() -> None:
    with pytest.raises(WorkflowConfigError, match="non-empty list"):
        parse_condition({"all_of": []}, "f")


def test_parse_condition_rejects_unknown_mapping() -> None:
    with pytest.raises(WorkflowConfigError, match="all_of"):
        parse_condition({"foo": "bar"}, "f")


def test_dump_workflow_round_trips_compound_guards() -> None:
    src = _wf_with_guard({"any_of": ["symptom = a", {"all_of": ["symptom = b", "age >= 18"]}]})
    wf = parse_workflow(src)
    redumped = parse_workflow(yaml.safe_load(dump_workflow(wf)))
    assert redumped.field("guarded").show_when == wf.field("guarded").show_when


def test_dump_workflow_keeps_simple_equality_as_string() -> None:
    wf = parse_workflow(_wf_with_guard("symptom = a"))
    assert "show_when: symptom = a" in dump_workflow(wf)


def _wf_with_guard(guard: object, extra: list[dict] | None = None) -> dict:
    fields: list[dict] = [
        {"name": "symptom", "type": "select", "required": True, "options": ["a", "b"]},
        {"name": "age", "type": "number", "required": True},
        {"name": "guarded", "type": "text", "required": False, "show_when": guard},
    ]
    fields.extend(extra or [])
    return {"workflow": "w", "profile": "express", "fields": fields}


def test_comparison_on_select_is_rejected() -> None:
    with pytest.raises(WorkflowConfigError, match="requires a number or date"):
        parse_workflow(_wf_with_guard("symptom > 1"))


def test_equality_value_not_an_option_is_rejected() -> None:
    with pytest.raises(WorkflowConfigError, match="not an option"):
        parse_workflow(_wf_with_guard("symptom = zzz"))


def test_comparison_literal_must_coerce() -> None:
    with pytest.raises(WorkflowConfigError, match="not a valid number"):
        parse_workflow(_wf_with_guard("age >= notnum"))


def test_unknown_ref_in_compound_is_rejected() -> None:
    with pytest.raises(WorkflowConfigError, match="unknown field 'ghost'"):
        parse_workflow(_wf_with_guard({"all_of": ["symptom = a", "ghost = x"]}))


def test_compound_guard_cycle_is_rejected() -> None:
    # guarded -> (refs age) ; age2 guarded by guarded ; guarded also guarded by age2 -> cycle
    data = {
        "workflow": "w",
        "profile": "express",
        "fields": [
            {"name": "p", "type": "select", "required": True, "options": ["x"]},
            {
                "name": "q",
                "type": "select",
                "required": True,
                "options": ["y"],
                "show_when": {"any_of": ["p = x", "r = z"]},
            },
            {
                "name": "r",
                "type": "select",
                "required": True,
                "options": ["z"],
                "show_when": "q = y",
            },
        ],
    }
    with pytest.raises(WorkflowConfigError, match="cycle"):
        parse_workflow(data)


def test_valid_compound_guard_loads() -> None:
    wf = parse_workflow(_wf_with_guard({"all_of": ["symptom = a", "age >= 18"]}))
    assert wf.field("guarded").show_when == AllOf(
        (Leaf("symptom", CompareOp.EQ, "a"), Leaf("age", CompareOp.GE, "18"))
    )


def test_coerce_scalar_number_date_datetime_and_failure() -> None:
    from datetime import date, datetime

    from prism.schema import coerce_scalar

    assert coerce_scalar(FieldType.NUMBER, "18") == 18.0
    assert coerce_scalar(FieldType.DATE, "2026-01-15") == date(2026, 1, 15)
    assert coerce_scalar(FieldType.DATETIME, "2026-01-15T10:00") == datetime(2026, 1, 15, 10, 0)
    assert coerce_scalar(FieldType.NUMBER, "not a number") is None
    assert coerce_scalar(FieldType.DATE, "nope") is None
    assert coerce_scalar(FieldType.SELECT, "x") is None
