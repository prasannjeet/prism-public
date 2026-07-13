"""W1 form_booking — the workflow loads through every guard and matches the design spec."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from prism.engine import StateEngine
from prism.gates import answer_gate, commit_gate, confirm
from prism.schema import Leaf, Profile
from prism.validators import load_validated_workflow

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"

TODAY = date(2026, 1, 15)


def _workflow():
    return load_validated_workflow(WORKFLOWS / "form_booking.yaml")


def _engine() -> StateEngine:
    return StateEngine(_workflow())


def test_form_booking_loads_validates_and_is_verified() -> None:
    workflow = _workflow()  # raises WorkflowConfigError if any schema/rule guard fails
    assert workflow.profile is Profile.VERIFIED


def test_field_names_in_declaration_order() -> None:
    assert [f.name for f in _workflow().fields] == [
        "service_type",
        "customer_name",
        "customer_email",
        "preferred_date",
        "preferred_time",
        "access_notes",
        "num_rooms",
        "deposit_protection",
        "landlord_email",
        "home_size_sqm",
        "appliances_inside",
        "pet_on_site",
        "num_windows",
        "floors_above_ground",
    ]


def test_optional_fields_are_exactly_two() -> None:
    optional = {f.name for f in _workflow().fields if not f.required}
    assert optional == {"access_notes", "pet_on_site"}


def test_branch_wiring() -> None:
    workflow = _workflow()
    wiring = {
        f.name: (f.show_when.field, f.show_when.value)
        for f in workflow.fields
        if isinstance(f.show_when, Leaf)
    }
    assert wiring == {
        "num_rooms": ("service_type", "move_out"),
        "deposit_protection": ("service_type", "move_out"),
        "landlord_email": ("deposit_protection", "yes"),
        "home_size_sqm": ("service_type", "deep_clean"),
        "appliances_inside": ("service_type", "deep_clean"),
        "pet_on_site": ("service_type", "deep_clean"),
        "num_windows": ("service_type", "windows"),
        "floors_above_ground": ("service_type", "windows"),
    }


def test_validators_match_design() -> None:
    val = {f.name: dict(f.validators) for f in _workflow().fields if f.validators}
    assert val == {
        "customer_email": {"pattern": "email"},
        "preferred_date": {"future_within_days": 90},
        "landlord_email": {"pattern": "email"},
        "num_rooms": {"range": {"min": 1, "max": 12}},
        "home_size_sqm": {"range": {"min": 10, "max": 500}},
        "num_windows": {"range": {"min": 1, "max": 60}},
        "floors_above_ground": {"range": {"min": 0, "max": 10}},
    }


def test_yes_no_options_are_strings_not_booleans() -> None:
    # Guards the pyyaml YAML-1.1 boolean trap: bare yes/no would load as [True, False].
    workflow = _workflow()
    assert workflow.field("deposit_protection").options == ("yes", "no")
    assert workflow.field("appliances_inside").options == ("yes", "no")
    assert workflow.field("pet_on_site").options == ("yes", "no")


def test_move_out_branch_activation_and_nest() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "service_type", "move_out")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert {"num_rooms", "deposit_protection"} <= active
    assert "home_size_sqm" not in active and "num_windows" not in active
    assert "landlord_email" not in active  # nest stays closed until deposit_protection = yes
    state = engine.with_answer(state, "deposit_protection", "yes")
    assert "landlord_email" in {f.name for f in engine.active_fields(state.answers)}


def test_switching_branch_prunes_move_out_subtree_in_one_transition() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "service_type", "move_out")
    state = engine.with_answer(state, "deposit_protection", "yes")
    state = engine.with_answer(state, "landlord_email", "ll@example.com")
    state = engine.with_answer(state, "service_type", "windows")
    assert "deposit_protection" not in state.answers
    assert "landlord_email" not in state.answers  # nested child pruned too


def test_full_move_out_booking_completes_then_commits_after_confirm() -> None:
    engine = _engine()
    state = engine.initial_state()
    answers = {
        "service_type": "move_out",
        "customer_name": "Sam Lee",
        "customer_email": "sam@example.com",
        "preferred_date": "2026-02-01",  # 17 days after TODAY, within 90
        "preferred_time": "morning",
        "num_rooms": "3",
        "deposit_protection": "yes",
        "landlord_email": "landlord@example.com",
    }
    for name, value in answers.items():
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    assert engine.is_complete(state)
    assert commit_gate(engine, state, enforce=True, turn=2).accepted is False  # needs confirm
    state = confirm(engine, state, turn=2).state
    assert commit_gate(engine, state, enforce=True, turn=3).accepted is True


def test_range_and_date_validators_reject_bad_values() -> None:
    engine = _engine()
    base = engine.with_answer(engine.initial_state(), "service_type", "move_out")
    too_many = answer_gate(engine, base, "num_rooms", "20", enforce=True, turn=1, today=TODAY)
    assert too_many.accepted is False  # range max 12
    past = answer_gate(
        engine,
        engine.initial_state(),
        "preferred_date",
        "2025-12-01",
        enforce=True,
        turn=1,
        today=TODAY,
    )
    assert past.accepted is False  # not within future_within_days window
