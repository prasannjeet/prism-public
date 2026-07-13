"""T6 clinical_trial_eligibility — the showpiece. Loads through every guard under the
compound-condition-extended engine and matches the design spec exactly."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from prism.engine import State, StateEngine
from prism.gates import answer_gate
from prism.schema import (
    AllOf,
    AnyOf,
    CompareOp,
    FieldType,
    Leaf,
    Profile,
    referenced_fields,
)
from prism.validators import load_validated_workflow

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"
TODAY = date(2026, 1, 15)


def _workflow():
    return load_validated_workflow(WORKFLOWS / "clinical_trial_eligibility.yaml")


def _engine() -> StateEngine:
    return StateEngine(_workflow())


def test_loads_validates_and_is_verified() -> None:
    workflow = _workflow()  # raises WorkflowConfigError if any schema/rule guard fails
    assert workflow.profile is Profile.VERIFIED


def test_exact_field_count_is_91() -> None:
    assert len(_workflow().fields) == 91


def test_exact_evidence_set_is_12() -> None:
    kinds = {f.name: f.type for f in _workflow().fields}
    evidence = {n for n, t in kinds.items() if t is FieldType.EVIDENCE}
    assert evidence == {
        "dx_confirmed_ev",
        "measurable_disease_ev",
        "mcl_risk_ev",
        "prior_records_ev",
        "cbc_ev",
        "chem_lft_ev",
        "coag_ev",
        "ecg_ev",
        "echo_ev",
        "serology_ev",
        "conmed_ev",
        "pregnancy_ev",
    }
    assert len(evidence) == 12


def test_exact_conclusion_set_is_9() -> None:
    kinds = {f.name: f.type for f in _workflow().fields}
    conclusion = {n for n, t in kinds.items() if t is FieldType.CONCLUSION}
    assert conclusion == {
        "enroll",
        "screen_fail_disease_lymphoma",
        "screen_fail_disease_leukemia",
        "screen_fail_disease_solid",
        "screen_fail_no_measurable",
        "screen_fail_safety",
        "rescreen",
        "refer_waiver",
        "escalate_cohort_slot",
    }
    assert len(conclusion) == 9


def test_answer_field_count_is_70() -> None:
    # 91 total - 12 evidence - 9 conclusions = 70 answer/visible fields.
    kinds = [f.type for f in _workflow().fields]
    answers = [t for t in kinds if t not in (FieldType.EVIDENCE, FieldType.CONCLUSION)]
    assert len(answers) == 70


def test_optional_fields_are_exactly_the_four() -> None:
    optional = {f.name for f in _workflow().fields if not f.required}
    assert optional == {"coag_ev", "rescreen", "refer_waiver", "escalate_cohort_slot"}


def test_no_evidence_field_appears_as_a_guard_leaf() -> None:
    workflow = _workflow()
    evidence = {f.name for f in workflow.fields if f.type is FieldType.EVIDENCE}
    referenced: set[str] = set()
    for f in workflow.fields:
        if f.show_when is not None:
            referenced |= referenced_fields(f.show_when)
    assert evidence.isdisjoint(referenced)  # evidence is referenced only by requires_evidence


def test_any_cancer_guard_parsed_on_dx_confirmed_ev() -> None:
    # The recurring `ANY_CANCER` any_of pattern: three `cancer_type = <type>` leaves.
    guard = _workflow().field("dx_confirmed_ev").show_when
    assert isinstance(guard, AnyOf)
    leaves = guard.conditions
    assert all(isinstance(c, Leaf) for c in leaves)
    assert {(c.field, c.op, c.value) for c in leaves if isinstance(c, Leaf)} == {
        ("cancer_type", CompareOp.EQ, "lymphoma"),
        ("cancer_type", CompareOp.EQ, "leukemia"),
        ("cancer_type", CompareOp.EQ, "solid_tumor"),
    }


def test_cancer_type_all_of_guard_parsed() -> None:
    # cancer_type gated all_of[consent_signed=yes, ecog_status<=2]: AND of `=` and a numeric leaf.
    guard = _workflow().field("cancer_type").show_when
    assert isinstance(guard, AllOf)
    by_field = {c.field: (c.op, c.value) for c in guard.conditions if isinstance(c, Leaf)}
    assert by_field == {
        "consent_signed": (CompareOp.EQ, "yes"),
        "ecog_status": (CompareOp.LE, "2"),
    }


def test_numeric_leaf_guards_parsed() -> None:
    # qtcf_ms > 470 gates qtc_confirmed_2of3; leuk_marrow_blasts >= 20 gates leuk_cytogenetic_risk.
    qtc = _workflow().field("qtc_confirmed_2of3").show_when
    assert isinstance(qtc, Leaf)
    assert (qtc.field, qtc.op, qtc.value) == ("qtcf_ms", CompareOp.GT, "470")
    cyto = _workflow().field("leuk_cytogenetic_risk").show_when
    assert isinstance(cyto, Leaf)
    assert (cyto.field, cyto.op, cyto.value) == ("leuk_marrow_blasts", CompareOp.GE, "20")


def test_cardiac_or_converging_guard_parsed() -> None:
    # cardiac_cleared active on any_of[uncontrolled_arrhythmia, recent_mi_acs].
    guard = _workflow().field("cardiac_cleared").show_when
    assert isinstance(guard, AnyOf)
    assert {(c.field, c.op, c.value) for c in guard.conditions if isinstance(c, Leaf)} == {
        ("cardiac_history", CompareOp.EQ, "uncontrolled_arrhythmia"),
        ("cardiac_history", CompareOp.EQ, "recent_mi_acs"),
    }


def test_enroll_requires_exactly_five_evidence() -> None:
    enroll = _workflow().field("enroll")
    assert enroll.requires_evidence == (
        "dx_confirmed_ev",
        "measurable_disease_ev",
        "cbc_ev",
        "chem_lft_ev",
        "serology_ev",
    )


def test_enroll_seven_leaf_all_of_guard() -> None:
    guard = _workflow().field("enroll").show_when
    assert isinstance(guard, AllOf)
    assert {(c.field, c.op, c.value) for c in guard.conditions if isinstance(c, Leaf)} == {
        ("measurable_disease_present", CompareOp.EQ, "yes"),
        ("marrow_adequate", CompareOp.EQ, "yes"),
        ("hiv_status", CompareOp.EQ, "negative"),
        ("hbsag", CompareOp.EQ, "negative"),
        ("hcv_ab", CompareOp.EQ, "negative"),
        ("cardiac_history", CompareOp.EQ, "none"),
        ("prohibited_conmed", CompareOp.EQ, "no"),
    }
    assert len(guard.conditions) == 7


def test_screen_fail_safety_requires_evidence() -> None:
    sf = _workflow().field("screen_fail_safety")
    assert sf.requires_evidence == ("cbc_ev", "serology_ev", "ecg_ev", "echo_ev")


def test_validators_present_and_well_shaped() -> None:
    val = {f.name: dict(f.validators) for f in _workflow().fields if f.validators}
    # Spot-check the spec's validator placement (not_future, future_within_days, range).
    assert val["screen_date"] == {"not_future": True}
    assert val["last_dose_date"] == {"not_future": True}
    assert val["planned_c1d1"] == {"future_within_days": 90}
    assert val["ecog_status"] == {"range": {"min": 0, "max": 5}}
    assert val["qtcf_ms"] == {"range": {"min": 200, "max": 700}}
    assert val["target_lesion_count"] == {"range": {"min": 0, "max": 5}}


def test_all_tools_declared_with_descriptions_and_used() -> None:
    workflow = _workflow()
    assert {t.name for t in workflow.tools} == {
        "pathology_assay",
        "imaging_recist",
        "cbc",
        "chem_lft",
        "coag_panel",
        "ecg_12lead",
        "echo_muga",
        "viral_serology",
        "pregnancy_test",
        "prior_records",
        "conmed_recon",
    }
    assert all(t.description for t in workflow.tools)
    used = {tool for f in workflow.fields for tool in f.satisfied_by}
    assert used == {t.name for t in workflow.tools}  # no declared-but-unused tools


def _answers(engine: StateEngine, pairs: list[tuple[str, str]]) -> State:
    state = engine.initial_state()
    for name, value in pairs:
        state = engine.with_answer(state, name, value)
    return state


def _active(engine: StateEngine, state: State) -> set[str]:
    return {f.name for f in engine.active_fields(state.answers)}


# --- depth: the deepest answer-driven chain reached in each of the 6 deep branches ---


def test_lymphoma_spine_active_at_depth_5() -> None:
    # spine: cancer_type -> lymphoma_subtype -> mcl_cyclin_d1 -> mcl_tp53 -> mcl_high_risk_eligible.
    engine = _engine()
    state = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "1"),
            ("cancer_type", "lymphoma"),
            ("lymphoma_subtype", "mcl"),
            ("mcl_cyclin_d1", "positive"),
            ("mcl_tp53", "present"),
        ],
    )
    active = _active(engine, state)
    # the full lymphoma spine is live to its deepest leaf
    assert {
        "lymphoma_subtype",
        "mcl_cyclin_d1",
        "mcl_risk_ev",
        "mcl_tp53",
        "mcl_high_risk_eligible",
    } <= active
    # the other disease spines stay closed
    assert "leukemia_lineage" not in active
    assert "solid_histology" not in active
    # inactive branch of the depth-5 leaf: mcl_tp53=absent closes mcl_high_risk_eligible
    state2 = engine.with_answer(state, "mcl_tp53", "absent")
    assert "mcl_high_risk_eligible" not in _active(engine, state2)


def test_leukemia_spine_active_at_depth_5() -> None:
    engine = _engine()
    state = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "0"),
            ("cancer_type", "leukemia"),
            ("leukemia_lineage", "aml"),
            ("leuk_marrow_blasts", "30"),
            ("leuk_cytogenetic_risk", "adverse"),
        ],
    )
    active = _active(engine, state)
    assert {
        "leukemia_lineage",
        "leuk_marrow_blasts",
        "leuk_cytogenetic_risk",
        "leuk_adverse_cohort_open",
    } <= active
    assert "cll_del17p" not in active  # cll-only leaf stays closed for aml
    # numeric inactive branch: blasts < 20 closes the cytogenetic-risk leaf
    state2 = engine.with_answer(state, "leuk_marrow_blasts", "10")
    inactive = _active(engine, state2)
    assert "leuk_cytogenetic_risk" not in inactive
    assert "leuk_adverse_cohort_open" not in inactive


def test_prior_therapy_cns_spine_active_at_depth_6() -> None:
    # The deepest chain: cancer_type -> prior_therapy_any -> prior_class=radiation
    #  -> cns_directed_prior=yes -> cns_stable_no_disease=yes -> cns_sponsor_approval.
    engine = _engine()
    state = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "1"),
            ("cancer_type", "lymphoma"),
            ("prior_therapy_any", "yes"),
            ("prior_class", "radiation"),
            ("cns_directed_prior", "yes"),
            ("cns_stable_no_disease", "yes"),
        ],
    )
    active = _active(engine, state)
    assert {
        "radiation_field",
        "cns_directed_prior",
        "cns_stable_no_disease",
        "cns_sponsor_approval",
    } <= active
    # AND-guard inactive branch: not stable closes the sponsor-approval leaf
    state2 = engine.with_answer(state, "cns_stable_no_disease", "no")
    assert "cns_sponsor_approval" not in _active(engine, state2)


def test_marrow_spine_active_at_depth_4() -> None:
    engine = _engine()
    state = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "1"),
            ("cancer_type", "leukemia"),
            ("marrow_adequate", "no"),
            ("cytopenia_cause", "tumor_involvement_ge30"),
            ("marrow_tumor_pct", "40"),
        ],
    )
    active = _active(engine, state)
    assert {"cytopenia_cause", "marrow_tumor_pct", "marrow_pi_discussion"} <= active
    assert "transfusion_independent" not in active  # the marrow_adequate=yes leaf is closed
    # numeric inactive branch: pct < 30 closes the PI-discussion leaf
    state2 = engine.with_answer(state, "marrow_tumor_pct", "10")
    assert "marrow_pi_discussion" not in _active(engine, state2)


def test_hbv_reflex_spine_active_at_depth_4() -> None:
    # serology reflex: hbsag=neg -> anti_hbc=positive -> hbv_dna_detectable=no -> hbv_prophylaxis.
    engine = _engine()
    state = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "1"),
            ("cancer_type", "solid_tumor"),
            ("hbsag", "negative"),
            ("anti_hbc", "positive"),
            ("hbv_dna_detectable", "no"),
        ],
    )
    active = _active(engine, state)
    assert {"anti_hbc", "hbv_dna_detectable", "hbv_prophylaxis"} <= active
    # inactive branch: a positive HBsAg closes the whole occult-HBV reflex
    state2 = engine.with_answer(state, "hbsag", "positive")
    closed = _active(engine, state2)
    assert "anti_hbc" not in closed
    assert "hbv_dna_detectable" not in closed
    assert "hbv_prophylaxis" not in closed


def test_solid_spine_active_at_depth_4() -> None:
    engine = _engine()
    state = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "2"),
            ("cancer_type", "solid_tumor"),
            ("solid_stage", "iv"),
            ("solid_lesion_prior_rt", "yes"),
        ],
    )
    active = _active(engine, state)
    assert {
        "solid_histology",
        "solid_stage",
        "target_lesion_count",
        "solid_lesion_prior_rt",
        "solid_new_lesion_confirmed",
    } <= active
    # all_of inactive branch: stage != iv closes the prior-RT sub-chain
    state2 = engine.with_answer(state, "solid_stage", "iii")
    closed = _active(engine, state2)
    assert "solid_lesion_prior_rt" not in closed
    assert "solid_new_lesion_confirmed" not in closed


def test_switching_cancer_type_prunes_whole_disease_spine_to_fixpoint() -> None:
    # The deep cascade: switch cancer_type after a deep lymphoma fill -> the entire lymphoma
    # spine + its evidence prune in one transition; the leukemia spine becomes active instead.
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("consent_signed", "yes"),
        ("ecog_status", "1"),
        ("cancer_type", "lymphoma"),
        ("lymphoma_subtype", "mcl"),
        ("mcl_cyclin_d1", "positive"),
        ("mcl_tp53", "present"),
        ("mcl_high_risk_eligible", "yes"),
    ]:
        state = engine.with_answer(state, name, value)
    state = engine.with_evidence(state, "mcl_risk_ev")
    assert "mcl_high_risk_eligible" in state.answers
    assert "mcl_risk_ev" in state.evidence
    state = engine.with_answer(state, "cancer_type", "leukemia")
    # exact pruned set: the entire lymphoma spine answers are gone
    assert "lymphoma_subtype" not in state.answers
    assert "mcl_cyclin_d1" not in state.answers
    assert "mcl_tp53" not in state.answers
    assert "mcl_high_risk_eligible" not in state.answers
    # ...and its evidence is gone too (heterogeneous prune)
    assert "mcl_risk_ev" not in state.evidence
    assert state.answers["cancer_type"] == "leukemia"
    # the leukemia spine is now the live disease branch
    active = _active(engine, state)
    assert "leukemia_lineage" in active
    assert "lymphoma_subtype" not in active


def test_switching_mid_chain_subdriver_prunes_below_only() -> None:
    # Switch a mid-chain sub-driver (prior_class) and assert ONLY the below-chain prunes,
    # the shared prior-therapy head survives.
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("consent_signed", "yes"),
        ("ecog_status", "1"),
        ("cancer_type", "lymphoma"),
        ("prior_therapy_any", "yes"),
        ("prior_class", "radiation"),
        ("cns_directed_prior", "yes"),
        ("cns_stable_no_disease", "yes"),
        ("cns_sponsor_approval", "yes"),
    ]:
        state = engine.with_answer(state, name, value)
    assert "cns_sponsor_approval" in state.answers
    state = engine.with_answer(state, "prior_class", "targeted")
    # the radiation sub-cascade is gone...
    for gone in [
        "radiation_field",
        "cns_directed_prior",
        "cns_stable_no_disease",
        "cns_sponsor_approval",
    ]:
        assert gone not in state.answers
    # ...the prior-therapy head survives, and the targeted leaf is now active
    assert state.answers["prior_therapy_any"] == "yes"
    assert "targeted_half_life_days" in _active(engine, state)


def test_ecog_gate_both_branches() -> None:
    # cancer_type gated all_of[consent_signed=yes, ecog_status<=2]: ECOG 2 admits, ECOG 3 closes it.
    engine = _engine()
    fit = _answers(engine, [("consent_signed", "yes"), ("ecog_status", "2")])
    assert "cancer_type" in _active(engine, fit)
    unfit = _answers(engine, [("consent_signed", "yes"), ("ecog_status", "3")])
    assert "cancer_type" not in _active(engine, unfit)


def test_any_cancer_pattern_true_for_each_type() -> None:
    # The ANY_CANCER any_of is true for each of the three options (and false before a type chosen).
    engine = _engine()
    head = _answers(engine, [("consent_signed", "yes"), ("ecog_status", "1")])
    assert "cbc_ev" not in _active(engine, head)  # ANY_CANCER false before a type
    for ctype in ["lymphoma", "leukemia", "solid_tumor"]:
        state = engine.with_answer(head, "cancer_type", ctype)
        active = _active(engine, state)
        # the cross-type, ANY_CANCER-gated fields are live for every cancer type
        for cross in [
            "dx_confirmed_ev",
            "cbc_ev",
            "ecg_ev",
            "serology_ev",
            "conmed_ev",
            "marrow_adequate",
        ]:
            assert cross in active, (ctype, cross)


def test_qtc_screenfail_trigger_reachable_and_enroll_independent() -> None:
    # The prior bug's regression: confirmed prolonged QTc must be reachable (it routes to
    # screen_fail_safety), AND the normal-QTc enroll happy path must stay reachable because
    # QTc is NOT in the enroll guard.
    engine = _engine()
    base = _answers(
        engine, [("consent_signed", "yes"), ("ecog_status", "1"), ("cancer_type", "lymphoma")]
    )
    # prolonged QTc opens the 2-of-3 confirmation leaf
    prolonged = engine.with_answer(base, "qtcf_ms", "500")
    assert "qtc_confirmed_2of3" in _active(engine, prolonged)
    # normal QTc never opens it (the happy path)
    normal = engine.with_answer(base, "qtcf_ms", "420")
    assert "qtc_confirmed_2of3" not in _active(engine, normal)
    # enroll guard references no QTc field
    enroll_guard = engine.workflow.field("enroll").show_when
    assert isinstance(enroll_guard, AllOf)
    enroll_refs = referenced_fields(enroll_guard)
    assert "qtcf_ms" not in enroll_refs
    assert "qtc_confirmed_2of3" not in enroll_refs


def test_each_disease_screenfail_guard_reachable() -> None:
    # Every top-level cancer branch can screen-fail on disease (no branch is a dead end).
    engine = _engine()
    # lymphoma: cyclin-D1 negative
    lym = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "1"),
            ("cancer_type", "lymphoma"),
            ("lymphoma_subtype", "mcl"),
            ("mcl_cyclin_d1", "negative"),
        ],
    )
    assert "screen_fail_disease_lymphoma" in _active(engine, lym)
    # leukemia: favorable cytogenetic risk
    leuk = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "1"),
            ("cancer_type", "leukemia"),
            ("leukemia_lineage", "aml"),
            ("leuk_marrow_blasts", "25"),
            ("leuk_cytogenetic_risk", "favorable"),
        ],
    )
    assert "screen_fail_disease_leukemia" in _active(engine, leuk)
    # solid: no eligible target lesion (target_lesion_count < 1)
    solid = _answers(
        engine,
        [
            ("consent_signed", "yes"),
            ("ecog_status", "1"),
            ("cancer_type", "solid_tumor"),
            ("solid_histology", "adeno"),
            ("solid_stage", "iv"),
            ("target_lesion_count", "0"),
        ],
    )
    assert "screen_fail_disease_solid" in _active(engine, solid)


def test_enroll_blocked_until_all_five_evidence_then_allowed() -> None:
    # The headline 5-evidence enroll gate: C5 blocks until ALL five packets collected; C4 admits
    # the premature attempt. Build the full enroll-eligible answer set, then add evidence stepwise.
    engine = _engine()
    state = engine.initial_state()
    enroll_answers = [
        ("consent_signed", "yes"),
        ("screen_date", "2026-01-10"),
        ("planned_c1d1", "2026-02-01"),
        ("patient_age", "61"),
        ("ecog_status", "1"),
        ("life_expectancy_ge_12wk", "yes"),
        ("cancer_type", "lymphoma"),
        ("measurable_disease_present", "yes"),
        ("lymphoma_subtype", "mcl"),
        ("mcl_cyclin_d1", "positive"),
        ("mcl_tp53", "absent"),
        ("prior_therapy_any", "no"),
        ("marrow_adequate", "yes"),
        ("transfusion_independent", "yes"),
        ("crcl_mlmin", "90"),
        ("hepatic_involvement", "no"),
        ("bili_x_uln", "1"),
        ("transaminase_x_uln", "1"),
        ("coag_elevated", "no"),
        ("qtcf_ms", "420"),
        ("lvef_pct", "60"),
        ("cardiac_history", "none"),
        ("hiv_status", "negative"),
        ("hbsag", "negative"),
        ("anti_hbc", "negative"),
        ("hcv_ab", "negative"),
        ("prohibited_conmed", "no"),
        ("steroid_mg_pred_eq", "5"),
        ("second_malignancy", "none"),
        ("childbearing_potential", "no"),
    ]
    for name, value in enroll_answers:
        state = engine.with_answer(state, name, value)
    # collect four of the five required evidence packets
    for ev in ["dx_confirmed_ev", "measurable_disease_ev", "cbc_ev", "chem_lft_ev"]:
        state = engine.with_evidence(state, ev)
    # enroll still blocked under C5 (serology_ev missing), admitted under C4
    blocked = answer_gate(
        engine, state, "enroll", "Enroll the patient.", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # missing serology_ev
    admitted = answer_gate(
        engine, state, "enroll", "Enroll the patient.", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4 admits + logs the premature conclusion
    # collect the fifth -> now allowed under C5
    state = engine.with_evidence(state, "serology_ev")
    allowed = answer_gate(
        engine, state, "enroll", "Enroll the patient.", enforce=True, turn=2, today=TODAY
    )
    assert allowed.accepted is True


def test_inactive_field_answer_blocked_under_enforce_admitted_without() -> None:
    # Wrong-spine temptation: answer a lymphoma-spine field while on the leukemia branch.
    engine = _engine()
    state = _answers(
        engine, [("consent_signed", "yes"), ("ecog_status", "1"), ("cancer_type", "leukemia")]
    )
    blocked = answer_gate(
        engine, state, "mcl_cyclin_d1", "positive", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: not on the active path
    admitted = answer_gate(
        engine, state, "mcl_cyclin_d1", "positive", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted + logged (engine prunes it)


def test_evidence_field_cannot_be_answered_directly() -> None:
    engine = _engine()
    state = _answers(
        engine, [("consent_signed", "yes"), ("ecog_status", "1"), ("cancer_type", "lymphoma")]
    )
    blocked = answer_gate(
        engine, state, "dx_confirmed_ev", "looks confirmed", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # C5: wrong_kind
    admitted = answer_gate(
        engine, state, "dx_confirmed_ev", "looks confirmed", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True  # C4: admitted + logged


def test_range_validators_reject_out_of_range() -> None:
    engine = _engine()
    head = _answers(
        engine, [("consent_signed", "yes"), ("ecog_status", "1"), ("cancer_type", "lymphoma")]
    )
    bad = answer_gate(engine, head, "qtcf_ms", "999", enforce=True, turn=1, today=TODAY)
    assert bad.accepted is False  # range 200-700
    ok = answer_gate(engine, head, "qtcf_ms", "420", enforce=True, turn=1, today=TODAY)
    assert ok.accepted is True
