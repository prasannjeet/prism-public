"""Scenario loading + fail-fast validation against the referenced workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.scenario import ScenarioConfigError, load_scenario, parse_scenario


def _doc() -> dict:
    return {
        "id": "w1-happy-01",
        "workflow": "form_booking",
        "category": "happy_path",
        "beats": [
            {
                "parts": [
                    {"say": "a move-out cleaning", "field": "service_type", "value": "move_out"},
                    {"say": "for Jane Doe", "field": "customer_name", "value": "Jane Doe"},
                ]
            },
            {"intent": "request_commit", "say": "yes, book it"},
        ],
        "expected": {
            "final_state": {"service_type": "move_out", "customer_name": "Jane Doe"},
            "valid_completion": ["committed"],
        },
    }


def test_parse_minimal_scenario() -> None:
    s = parse_scenario(_doc(), source="<test>")
    assert s.id == "w1-happy-01"
    assert s.workflow == "form_booking"
    assert s.beats[0].parts[1].field == "customer_name"
    assert s.beats[1].intent == "request_commit"
    assert s.expected.valid_completion == ("committed",)


def test_unknown_field_rejected() -> None:
    doc = _doc()
    doc["beats"][0]["parts"][0]["field"] = "not_a_field"
    with pytest.raises(ScenarioConfigError, match="not_a_field"):
        parse_scenario(doc, source="<test>")


def test_bad_select_value_rejected() -> None:
    doc = _doc()
    doc["beats"][0]["parts"][0]["value"] = "spaceship_wash"  # service_type is a select
    with pytest.raises(ScenarioConfigError, match="service_type"):
        parse_scenario(doc, source="<test>")


def test_unknown_category_rejected() -> None:
    doc = _doc()
    doc["category"] = "made_up"
    with pytest.raises(ScenarioConfigError, match="category"):
        parse_scenario(doc, source="<test>")


def test_unknown_completion_rejected() -> None:
    doc = _doc()
    doc["expected"]["valid_completion"] = ["teleported"]
    with pytest.raises(ScenarioConfigError, match="valid_completion"):
        parse_scenario(doc, source="<test>")


def test_workflows_dir_is_injectable() -> None:
    s = parse_scenario(_doc(), source="<test>", workflows_dir=Path("workflows"))
    assert s.workflow == "form_booking"
    with pytest.raises(ScenarioConfigError, match="unknown workflow"):
        parse_scenario(_doc(), source="<test>", workflows_dir=Path("nonexistent"))


def test_final_state_partitioned_by_field_type() -> None:
    # A SELECT field is exact-matched; a CONCLUSION field is presence-only. final_state keeps both.
    doc = {
        "id": "w2-outcome-01",
        "workflow": "incident_investigation",
        "category": "happy_path",
        "beats": [{"say": "go", "intent": "request_commit"}],
        "expected": {
            "final_state": {
                "symptom": "server_down",
                "server_root_cause": "a bad deploy",
            },
            "valid_completion": ["committed"],
        },
    }
    s = parse_scenario(doc, source="<test>")
    assert dict(s.expected.exact_state) == {"symptom": "server_down"}
    assert s.expected.present_fields == ("server_root_cause",)
    # final_state is kept as authored (all fields) for the key-set check + manifests.
    assert set(s.expected.final_state.keys()) == {"symptom", "server_root_cause"}


def test_id_must_match_filename(tmp_path: Path) -> None:
    p = tmp_path / "different-name.yaml"
    p.write_text(
        "id: w1-happy-01\nworkflow: form_booking\ncategory: happy_path\n"
        "beats: []\nexpected: {final_state: {}, valid_completion: [committed]}\n"
    )
    with pytest.raises(ScenarioConfigError, match="filename"):
        load_scenario(p)
