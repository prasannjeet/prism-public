"""T4 travel_insurance_claim — loads through every guard (incl. compound show_when) and
matches the design spec. Structure + counts + the three compound guards parsed correctly."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from prism.engine import StateEngine
from prism.gates import answer_gate, commit_gate, confirm, record_evidence
from prism.schema import AllOf, AnyOf, CompareOp, FieldType, Leaf, Profile, referenced_fields
from prism.validators import load_validated_workflow

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"

TODAY = date(2026, 1, 15)


def _workflow():
    return load_validated_workflow(WORKFLOWS / "travel_insurance_claim.yaml")


def _engine() -> StateEngine:
    return StateEngine(_workflow())


def _active(engine: StateEngine, answers: dict[str, str]) -> set[str]:
    return {f.name for f in engine.active_fields(answers)}


def test_loads_validates_and_is_verified() -> None:
    workflow = _workflow()  # raises WorkflowConfigError if any schema/rule guard fails
    assert workflow.profile is Profile.VERIFIED


def test_field_names_in_declaration_order() -> None:
    assert [f.name for f in _workflow().fields] == [
        "policy_number",
        "policy_purchase_date",
        "claimant_email",
        "claim_type",
        "claimed_amount",
        "physician_statement",
        "medical_subtype",
        "medical_care_setting",
        "admit_discharge_dates",
        "evac_necessity_cert",
        "proof_of_payment",
        "cancellation_cause",
        "medical_outcome",
        "death_certificate",
        "cancel_physician_statement",
        "non_medical_reason",
        "named_reason_document",
        "baggage_proof_of_ownership",
        "baggage_subtype",
        "police_report_filed",
        "police_report",
        "damage_severity",
        "repair_estimate",
        "extra_documentation",
        "medical_adjudication",
        "cancellation_adjudication",
        "baggage_adjudication",
    ]


def test_exact_field_count_is_27() -> None:
    assert len(_workflow().fields) == 27


def test_node_kind_counts_and_sets() -> None:
    kinds = {f.name: f.type for f in _workflow().fields}
    evidence = {n for n, t in kinds.items() if t is FieldType.EVIDENCE}
    conclusion = {n for n, t in kinds.items() if t is FieldType.CONCLUSION}
    select = {n for n, t in kinds.items() if t is FieldType.SELECT}
    assert evidence == {
        "physician_statement",
        "evac_necessity_cert",
        "proof_of_payment",
        "death_certificate",
        "cancel_physician_statement",
        "named_reason_document",
        "baggage_proof_of_ownership",
        "police_report",
        "repair_estimate",
        "extra_documentation",
    }
    assert len(evidence) == 10
    assert conclusion == {
        "medical_adjudication",
        "cancellation_adjudication",
        "baggage_adjudication",
    }
    assert len(conclusion) == 3
    # 9 selects: the primary driver + 8 sub-drivers
    assert select == {
        "claim_type",
        "medical_subtype",
        "medical_care_setting",
        "cancellation_cause",
        "medical_outcome",
        "non_medical_reason",
        "baggage_subtype",
        "police_report_filed",
        "damage_severity",
    }
    assert len(select) == 9


def test_select_options_match_spec() -> None:
    opts = {f.name: f.options for f in _workflow().fields if f.type is FieldType.SELECT}
    assert opts == {
        "claim_type": ("medical", "cancellation", "baggage"),
        "medical_subtype": ("expense", "evacuation"),
        "medical_care_setting": ("inpatient", "outpatient"),
        "cancellation_cause": ("medical_reason", "non_medical"),
        "medical_outcome": ("sickness", "injury", "death"),
        "non_medical_reason": ("jury_duty", "employer", "supplier_default", "hurricane_warning"),
        "baggage_subtype": ("lost_in_transit", "stolen", "damaged"),
        "police_report_filed": ("yes", "no"),
        "damage_severity": ("repairable", "total_loss"),
    }


def test_optional_fields_are_exactly_the_sub_gated_evidence() -> None:
    optional = {f.name for f in _workflow().fields if not f.required}
    assert optional == {
        "death_certificate",
        "cancel_physician_statement",
        "named_reason_document",
        "police_report",
        "repair_estimate",
        "extra_documentation",
    }


def test_validators_match_design() -> None:
    val = {f.name: dict(f.validators) for f in _workflow().fields if f.validators}
    assert val == {
        "policy_number": {"length": {"min": 6, "max": 20}},
        "policy_purchase_date": {"not_future": True},
        "claimant_email": {"pattern": "email"},
        "claimed_amount": {"range": {"min": 1, "max": 1000000}},
        "admit_discharge_dates": {"length": {"min": 4, "max": 60}},
    }


def test_evidence_satisfied_by() -> None:
    sat = {f.name: f.satisfied_by for f in _workflow().fields if f.satisfied_by}
    assert sat == {
        "physician_statement": ("upload_physician_statement",),
        "evac_necessity_cert": ("upload_evac_certification",),
        "proof_of_payment": ("upload_proof_of_payment",),
        "death_certificate": ("upload_death_certificate",),
        "cancel_physician_statement": ("upload_physician_statement",),
        "named_reason_document": ("upload_named_reason_document",),
        "baggage_proof_of_ownership": ("upload_ownership_proof",),
        "police_report": ("upload_police_report",),
        "repair_estimate": ("upload_repair_estimate",),
        "extra_documentation": ("upload_supporting_docs",),
    }


def test_physician_statement_tool_is_shared_legally() -> None:
    # The one shared tool: physician_statement (medical) and cancel_physician_statement
    # (cancellation) both use upload_physician_statement; they are never co-active.
    wf = _workflow()
    assert wf.field("physician_statement").satisfied_by == ("upload_physician_statement",)
    assert wf.field("cancel_physician_statement").satisfied_by == ("upload_physician_statement",)


def test_conclusions_require_their_branch_wide_evidence() -> None:
    req = {f.name: f.requires_evidence for f in _workflow().fields if f.requires_evidence}
    assert req == {
        "medical_adjudication": ("physician_statement",),
        "cancellation_adjudication": ("proof_of_payment",),
        "baggage_adjudication": ("baggage_proof_of_ownership",),
    }


def test_simple_equality_guards_parsed_as_leaves() -> None:
    wf = _workflow()
    # spot-check the simple-equality (string shorthand) guards parse to Leaf(=)
    for name, ref, val in [
        ("physician_statement", "claim_type", "medical"),
        ("medical_subtype", "claim_type", "medical"),
        ("medical_care_setting", "medical_subtype", "expense"),
        ("admit_discharge_dates", "medical_care_setting", "inpatient"),
        ("evac_necessity_cert", "medical_subtype", "evacuation"),
        ("baggage_subtype", "claim_type", "baggage"),
        ("police_report_filed", "baggage_subtype", "stolen"),
        ("damage_severity", "baggage_subtype", "damaged"),
        ("repair_estimate", "damage_severity", "repairable"),
        ("medical_adjudication", "claim_type", "medical"),
    ]:
        guard = wf.field(name).show_when
        assert isinstance(guard, Leaf), name
        assert (guard.field, guard.op, guard.value) == (ref, CompareOp.EQ, val), name


def test_police_report_guard_is_yes_string_not_bool() -> None:
    # police_report_filed = "yes": the value must survive as the string "yes" (YAML bool trap).
    guard = _workflow().field("police_report").show_when
    assert isinstance(guard, Leaf)
    assert (guard.field, guard.op, guard.value) == ("police_report_filed", CompareOp.EQ, "yes")


def test_death_certificate_compound_guard_is_all_of_two_equality_leaves() -> None:
    # all_of: [cancellation_cause = medical_reason, medical_outcome = death]
    guard = _workflow().field("death_certificate").show_when
    assert isinstance(guard, AllOf)
    assert len(guard.conditions) == 2
    leaves = guard.conditions
    assert all(isinstance(c, Leaf) for c in leaves)
    assert {(c.field, c.op, c.value) for c in leaves} == {  # type: ignore[union-attr]
        ("cancellation_cause", CompareOp.EQ, "medical_reason"),
        ("medical_outcome", CompareOp.EQ, "death"),
    }
    assert referenced_fields(guard) == frozenset({"cancellation_cause", "medical_outcome"})


def test_cancel_physician_statement_guard_is_all_of_with_in_membership() -> None:
    # all_of: [cancellation_cause = medical_reason,
    #          {field: medical_outcome, in: [sickness, injury]}]
    # the `in` sugar desugars to AnyOf of two `=` leaves.
    guard = _workflow().field("cancel_physician_statement").show_when
    assert isinstance(guard, AllOf)
    assert len(guard.conditions) == 2
    eq_leaf = next(c for c in guard.conditions if isinstance(c, Leaf))
    assert (eq_leaf.field, eq_leaf.op, eq_leaf.value) == (
        "cancellation_cause",
        CompareOp.EQ,
        "medical_reason",
    )
    membership = next(c for c in guard.conditions if isinstance(c, AnyOf))
    assert {(c.field, c.op, c.value) for c in membership.conditions} == {  # type: ignore[union-attr]
        ("medical_outcome", CompareOp.EQ, "sickness"),
        ("medical_outcome", CompareOp.EQ, "injury"),
    }
    assert referenced_fields(guard) == frozenset({"cancellation_cause", "medical_outcome"})


def test_extra_documentation_guard_is_numeric_gt_leaf_on_claimed_amount() -> None:
    # the single comparison leaf: claimed_amount > 10000 (against the NUMBER field).
    guard = _workflow().field("extra_documentation").show_when
    assert isinstance(guard, Leaf)
    assert (guard.field, guard.op, guard.value) == ("claimed_amount", CompareOp.GT, "10000")


def test_all_tools_declared_with_descriptions_and_used() -> None:
    wf = _workflow()
    assert {t.name for t in wf.tools} == {
        "upload_physician_statement",
        "upload_evac_certification",
        "upload_proof_of_payment",
        "upload_death_certificate",
        "upload_named_reason_document",
        "upload_ownership_proof",
        "upload_police_report",
        "upload_repair_estimate",
        "upload_supporting_docs",
    }  # 9 distinct tools (physician statement is shared by two fields)
    assert all(t.description for t in wf.tools)
    used = {tool for f in wf.fields for tool in f.satisfied_by}
    assert used == {t.name for t in wf.tools}  # no declared-but-unused tools


def test_medical_branch_depth4_activation_inpatient_path() -> None:
    # claim_type=medical -> medical_subtype=expense -> medical_care_setting=inpatient
    # -> admit_discharge_dates active. Each level gates the next.
    engine = _engine()
    base = {"claim_type": "medical"}
    active = _active(engine, base)
    assert active == {
        "policy_number",
        "policy_purchase_date",
        "claimant_email",
        "claim_type",
        "claimed_amount",
        "physician_statement",
        "medical_subtype",
        "medical_adjudication",
    }
    assert "medical_care_setting" not in active  # L3 closed until medical_subtype answered
    assert "admit_discharge_dates" not in active  # L4 closed

    expense = {**base, "medical_subtype": "expense"}
    assert "medical_care_setting" in _active(engine, expense)  # L3 opens
    assert "admit_discharge_dates" not in _active(engine, expense)  # L4 still closed
    assert "evac_necessity_cert" not in _active(engine, expense)  # evacuation path closed

    inpatient = {**expense, "medical_care_setting": "inpatient"}
    assert "admit_discharge_dates" in _active(engine, inpatient)  # L4 opens (depth 4)

    outpatient = {**expense, "medical_care_setting": "outpatient"}
    assert "admit_discharge_dates" not in _active(engine, outpatient)  # other L3 value: L4 closed


def test_medical_evacuation_path_closes_expense_subtree() -> None:
    engine = _engine()
    evac = {"claim_type": "medical", "medical_subtype": "evacuation"}
    active = _active(engine, evac)
    assert "evac_necessity_cert" in active
    assert "medical_care_setting" not in active  # expense-only L3
    assert "admit_discharge_dates" not in active  # expense-only L4


def test_cancellation_branch_depth4_death_path() -> None:
    # claim_type=cancellation -> cancellation_cause=medical_reason -> medical_outcome=death
    # -> death_certificate active (depth 4 via single-field chain).
    engine = _engine()
    base = {"claim_type": "cancellation"}
    active = _active(engine, base)
    assert active == {
        "policy_number",
        "policy_purchase_date",
        "claimant_email",
        "claim_type",
        "claimed_amount",
        "proof_of_payment",
        "cancellation_cause",
        "cancellation_adjudication",
    }
    assert "medical_outcome" not in active  # L3 closed
    assert "death_certificate" not in active  # L4 closed

    med = {**base, "cancellation_cause": "medical_reason"}
    assert "medical_outcome" in _active(engine, med)  # L3 opens
    assert "death_certificate" not in _active(engine, med)  # L4 closed until outcome=death
    assert "non_medical_reason" not in _active(engine, med)  # non_medical path closed

    death = {**med, "medical_outcome": "death"}
    death_active = _active(engine, death)
    assert "death_certificate" in death_active  # L4 opens (the AND is satisfied)
    assert "cancel_physician_statement" not in death_active  # the in-membership is false for death


def test_baggage_branch_two_depth4_paths() -> None:
    engine = _engine()
    base = {"claim_type": "baggage"}
    active = _active(engine, base)
    assert active == {
        "policy_number",
        "policy_purchase_date",
        "claimant_email",
        "claim_type",
        "claimed_amount",
        "baggage_proof_of_ownership",
        "baggage_subtype",
        "baggage_adjudication",
    }
    # path A: stolen -> police_report_filed -> police_report
    stolen = {**base, "baggage_subtype": "stolen"}
    assert "police_report_filed" in _active(engine, stolen)  # L3
    assert "police_report" not in _active(engine, stolen)  # L4 closed until filed=yes
    filed = {**stolen, "police_report_filed": "yes"}
    assert "police_report" in _active(engine, filed)  # L4 opens (depth 4)
    not_filed = {**stolen, "police_report_filed": "no"}
    assert "police_report" not in _active(engine, not_filed)  # other L3 value: L4 closed
    # path B: damaged -> damage_severity -> repair_estimate
    damaged = {**base, "baggage_subtype": "damaged"}
    assert "damage_severity" in _active(engine, damaged)  # L3
    assert "police_report_filed" not in _active(engine, damaged)  # stolen-only L3
    assert "repair_estimate" not in _active(engine, damaged)  # L4 closed until repairable
    repairable = {**damaged, "damage_severity": "repairable"}
    assert "repair_estimate" in _active(engine, repairable)  # L4 opens (second depth-4 path)
    total = {**damaged, "damage_severity": "total_loss"}
    assert "repair_estimate" not in _active(engine, total)  # other L3 value: L4 closed


def test_death_certificate_and_guard_both_branches() -> None:
    # AllOf(cancellation_cause = medical_reason, medical_outcome = death): true only when BOTH.
    engine = _engine()
    cert = engine.workflow.field("death_certificate")
    assert (
        engine.is_active(
            cert,
            {
                "claim_type": "cancellation",
                "cancellation_cause": "medical_reason",
                "medical_outcome": "death",
            },
        )
        is True
    )
    # first conjunct false (non_medical) -> the chain that makes medical_outcome active is broken
    assert (
        engine.is_active(
            cert, {"claim_type": "cancellation", "cancellation_cause": "non_medical"}
        )
        is False
    )
    # second conjunct false (sickness, not death)
    assert (
        engine.is_active(
            cert,
            {
                "claim_type": "cancellation",
                "cancellation_cause": "medical_reason",
                "medical_outcome": "sickness",
            },
        )
        is False
    )


def test_cancel_physician_in_membership_both_branches() -> None:
    # AllOf(cause = medical_reason, medical_outcome in [sickness, injury]).
    engine = _engine()
    fld = engine.workflow.field("cancel_physician_statement")
    med = {"claim_type": "cancellation", "cancellation_cause": "medical_reason"}
    assert engine.is_active(fld, {**med, "medical_outcome": "sickness"}) is True
    assert engine.is_active(fld, {**med, "medical_outcome": "injury"}) is True
    assert engine.is_active(fld, {**med, "medical_outcome": "death"}) is False  # not in the set


def test_extra_documentation_numeric_threshold_at_above_below() -> None:
    # Leaf claimed_amount > 10000: false at 10000, true above, false below.
    engine = _engine()
    fld = engine.workflow.field("extra_documentation")
    assert engine.is_active(fld, {"claim_type": "medical", "claimed_amount": "10001"}) is True
    assert engine.is_active(fld, {"claim_type": "medical", "claimed_amount": "10000"}) is False
    assert engine.is_active(fld, {"claim_type": "medical", "claimed_amount": "9999"}) is False


def test_baggage_subtype_switch_prunes_deep_subtree_and_reactivates_other() -> None:
    # The T4 signature: switch damaged -> stolen prunes the damaged answer+evidence subtree in one
    # transition and re-activates the stolen sub-chain.
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "claim_type", "baggage")
    state = engine.with_evidence(state, "baggage_proof_of_ownership")
    state = engine.with_answer(state, "baggage_subtype", "damaged")
    state = engine.with_answer(state, "damage_severity", "repairable")
    state = engine.with_evidence(state, "repair_estimate")
    assert state.answers["damage_severity"] == "repairable"
    assert "repair_estimate" in state.evidence
    state = engine.with_answer(state, "baggage_subtype", "stolen")
    # one transition: the damaged answer AND its evidence are gone; ownership proof survives
    assert "damage_severity" not in state.answers
    assert "repair_estimate" not in state.evidence
    assert state.evidence == frozenset({"baggage_proof_of_ownership"})
    assert state.answers == {"claim_type": "baggage", "baggage_subtype": "stolen"}
    # the stolen sub-chain is now active
    assert "police_report_filed" in {f.name for f in engine.active_fields(state.answers)}


def test_claim_type_switch_prunes_entire_branch_subtree() -> None:
    # Switch medical -> cancellation: the whole medical deep chain + evidence prune at once.
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "claim_type", "medical")
    state = engine.with_evidence(state, "physician_statement")
    state = engine.with_answer(state, "medical_subtype", "expense")
    state = engine.with_answer(state, "medical_care_setting", "inpatient")
    state = engine.with_answer(state, "admit_discharge_dates", "2026-01-02 to 2026-01-05")
    assert state.answers["admit_discharge_dates"] == "2026-01-02 to 2026-01-05"
    assert "physician_statement" in state.evidence
    state = engine.with_answer(state, "claim_type", "cancellation")
    assert state.answers == {"claim_type": "cancellation"}  # all medical answers pruned
    assert state.evidence == frozenset()  # medical evidence pruned
    active = {f.name for f in engine.active_fields(state.answers)}
    assert "proof_of_payment" in active and "cancellation_cause" in active
    assert "medical_subtype" not in active and "admit_discharge_dates" not in active


def test_inactive_field_answer_blocked_under_enforce_admitted_without() -> None:
    # Answer a baggage field while the medical branch is active.
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "claim_type", "medical")
    blocked = answer_gate(
        engine, state, "baggage_subtype", "stolen", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: not on the active path
    admitted = answer_gate(
        engine, state, "baggage_subtype", "stolen", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted + logged (engine prunes the inactive answer)


def test_invalid_values_rejected_under_enforce() -> None:
    engine = _engine()
    state = engine.initial_state()
    # policy_number too short (< 6)
    assert (
        answer_gate(
            engine, state, "policy_number", "AB12", enforce=True, turn=1, today=TODAY
        ).accepted
        is False
    )
    assert (
        answer_gate(
            engine, state, "policy_number", "POL-000123", enforce=True, turn=1, today=TODAY
        ).accepted
        is True
    )
    # policy_purchase_date in the future
    assert (
        answer_gate(
            engine, state, "policy_purchase_date", "2026-02-01", enforce=True, turn=1, today=TODAY
        ).accepted
        is False
    )
    assert (
        answer_gate(
            engine, state, "policy_purchase_date", "2025-12-01", enforce=True, turn=1, today=TODAY
        ).accepted
        is True
    )
    # claimed_amount out of range (> max)
    assert (
        answer_gate(
            engine, state, "claimed_amount", "2000000", enforce=True, turn=1, today=TODAY
        ).accepted
        is False
    )
    assert (
        answer_gate(
            engine, state, "claimed_amount", "4200", enforce=True, turn=1, today=TODAY
        ).accepted
        is True
    )


def test_evidence_field_cannot_be_answered_directly() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "claim_type", "medical")
    blocked = answer_gate(
        engine, state, "physician_statement", "looks fine", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: wrong_kind
    admitted = answer_gate(
        engine, state, "physician_statement", "looks fine", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted + logged


def test_premature_conclusion_blocked_under_enforce_admitted_without() -> None:
    engine = _engine()
    state = engine.with_answer(engine.initial_state(), "claim_type", "medical")
    blocked = answer_gate(
        engine, state, "medical_adjudication", "approved", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: missing physician_statement
    admitted = answer_gate(
        engine, state, "medical_adjudication", "approved", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted + logged


def test_each_conclusion_reachable_with_its_coactive_evidence() -> None:
    engine = _engine()
    for claim, evidence_field, conclusion in [
        ("medical", "physician_statement", "medical_adjudication"),
        ("cancellation", "proof_of_payment", "cancellation_adjudication"),
        ("baggage", "baggage_proof_of_ownership", "baggage_adjudication"),
    ]:
        state = engine.with_answer(engine.initial_state(), "claim_type", claim)
        state = engine.with_evidence(state, evidence_field)
        outcome = answer_gate(
            engine, state, conclusion, "adjudicated", enforce=True, turn=2, today=TODAY
        )
        # evidence present -> allowed
        assert outcome.accepted is True, (conclusion, outcome.message)


def test_full_medical_claim_completes_then_commits_after_confirm() -> None:
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("policy_number", "POL-000123"),
        ("policy_purchase_date", "2025-12-01"),
        ("claimant_email", "claimant@example.com"),
        ("claim_type", "medical"),
        ("claimed_amount", "4200"),
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    outcome = record_evidence(
        engine, state, "physician_statement", "upload_physician_statement", enforce=True, turn=2
    )
    assert outcome.accepted, outcome.message
    state = outcome.state
    for name, value in [
        ("medical_subtype", "expense"),
        ("medical_care_setting", "inpatient"),
        ("admit_discharge_dates", "2026-01-02 to 2026-01-05"),
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=3, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    outcome = answer_gate(
        engine,
        state,
        "medical_adjudication",
        "Inpatient expenses are covered under the policy; the claim is approved for the amount.",
        enforce=True,
        turn=4,
        today=TODAY,
    )
    assert outcome.accepted, outcome.message
    state = outcome.state
    assert engine.is_complete(state)  # optional evidence (extra_documentation etc.) not required
    # verified: commit blocked until confirm
    assert commit_gate(engine, state, enforce=True, turn=5).accepted is False
    state = confirm(engine, state, turn=5).state
    assert commit_gate(engine, state, enforce=True, turn=6).accepted is True
