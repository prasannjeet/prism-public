"""T5 incident_response: loads through every compound-condition guard, matches the design spec."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from prism.engine import StateEngine
from prism.gates import answer_gate, commit_gate, record_evidence
from prism.schema import AllOf, AnyOf, CompareOp, FieldType, Leaf, Profile, referenced_fields
from prism.validators import load_validated_workflow

TODAY = date(2026, 1, 15)

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"


def _workflow():
    return load_validated_workflow(WORKFLOWS / "incident_response.yaml")


FIELD_NAMES = [
    "reporter_channel",
    "alert_summary",
    "triage_validation",
    "alert_disposition",
    "functional_impact",
    "information_impact",
    "affected_host_count",
    "severity",
    "incident_type",
    "asset_class",
    "mass_incident_bridge_note",
    # phishing / bec
    "phish_header_pull",
    "is_malicious_email",
    "spoof_class",
    "recipient_scope_pull",
    "anyone_clicked",
    "payload_exec_scan",
    "entered_credentials",
    "bec_persistence_pull",
    "phish_takeover_confirmed",
    # ransomware
    "variant_ioc_lookup",
    "encryption_scope",
    "ransom_scope_scan",
    "recovery_path",
    "data_impact_scan",
    "backup_freshness",
    "backup_integrity_check",
    "ransom_recovery_decision",
    # compromised credentials
    "compromise_confirm_pull",
    "principal_type",
    "attacker_activity_pull",
    "persistence_established",
    "backdoor_enum_pull",
    "cred_eradication_plan",
    # data exfiltration
    "exfil_confirmed",
    "exfil_egress_pull",
    "exfil_data_class",
    "breach_notification_decision",
    # shared disposition + tail
    "containment_action",
    "disposition",
    "incident_report",
    "lessons_learned",
]

EVIDENCE = {
    "triage_validation",
    "phish_header_pull",
    "recipient_scope_pull",
    "payload_exec_scan",
    "bec_persistence_pull",
    "variant_ioc_lookup",
    "ransom_scope_scan",
    "data_impact_scan",
    "backup_integrity_check",
    "compromise_confirm_pull",
    "attacker_activity_pull",
    "backdoor_enum_pull",
    "exfil_egress_pull",
}

CONCLUSIONS = {
    "phish_takeover_confirmed",
    "ransom_recovery_decision",
    "cred_eradication_plan",
    "breach_notification_decision",
}

OPTIONAL = {"mass_incident_bridge_note", "lessons_learned"}

TOOLS = {
    "ioc_lookup",
    "edr_scan",
    "mailbox_audit",
    "auth_log",
    "network_capture",
    "dlp_log",
    "file_integrity",
    "identity_graph",
    "backup_check",
}


def test_loads_validates_and_is_verified() -> None:
    workflow = _workflow()  # raises WorkflowConfigError if any schema/rule guard fails
    assert workflow.profile is Profile.VERIFIED


def test_field_names_in_declaration_order() -> None:
    assert [f.name for f in _workflow().fields] == FIELD_NAMES


def test_field_count_and_required_split() -> None:
    fields = _workflow().fields
    assert len(fields) == 42
    optional = {f.name for f in fields if not f.required}
    assert optional == OPTIONAL  # exactly 2 optional, the rest required
    assert len([f for f in fields if f.required]) == 40


def test_node_kinds() -> None:
    kinds = {f.name: f.type for f in _workflow().fields}
    assert {n for n, t in kinds.items() if t is FieldType.EVIDENCE} == EVIDENCE
    assert {n for n, t in kinds.items() if t is FieldType.CONCLUSION} == CONCLUSIONS
    assert len(EVIDENCE) == 13
    assert len(CONCLUSIONS) == 4


def test_conclusion_requires_evidence_exact() -> None:
    req = {f.name: f.requires_evidence for f in _workflow().fields if f.requires_evidence}
    assert req == {
        "phish_takeover_confirmed": ("bec_persistence_pull", "payload_exec_scan"),
        "ransom_recovery_decision": (
            "backup_integrity_check",
            "data_impact_scan",
            "ransom_scope_scan",
        ),
        "cred_eradication_plan": (
            "backdoor_enum_pull",
            "attacker_activity_pull",
            "compromise_confirm_pull",
        ),
        "breach_notification_decision": ("exfil_egress_pull",),
    }
    # 3 multi-evidence (2/3/3) + 1 single-evidence by design
    assert sorted(len(v) for v in req.values()) == [1, 2, 3, 3]


def test_validators_match_design() -> None:
    val = {f.name: dict(f.validators) for f in _workflow().fields if f.validators}
    assert val == {
        "alert_summary": {"length": {"min": 10, "max": 500}},
        "affected_host_count": {"range": {"min": 0, "max": 100000}},
        "mass_incident_bridge_note": {"length": {"min": 0, "max": 1000}},
        "incident_report": {"length": {"min": 30, "max": 4000}},
        "lessons_learned": {"length": {"min": 0, "max": 2000}},
    }


def test_all_tools_declared_with_descriptions_and_used() -> None:
    workflow = _workflow()
    assert {t.name for t in workflow.tools} == TOOLS
    assert len(TOOLS) == 9
    assert all(t.description for t in workflow.tools)  # every tool documents itself
    used = {tool for f in workflow.fields for tool in f.satisfied_by}
    assert used == TOOLS  # no declared-but-unused, no used-but-undeclared


def _field(name: str):
    return next(f for f in _workflow().fields if f.name == name)


def test_numeric_guard_parsed_on_mass_incident_bridge_note() -> None:
    # The breach-bridge note is gated by all_of[disposition=confirmed_incident, host_count >= 50].
    guard = _field("mass_incident_bridge_note").show_when
    assert isinstance(guard, AllOf)
    leaves = guard.conditions
    assert isinstance(leaves[0], Leaf)
    assert (leaves[0].field, leaves[0].op, leaves[0].value) == (
        "alert_disposition",
        CompareOp.EQ,
        "confirmed_incident",
    )
    assert isinstance(leaves[1], Leaf)
    assert (leaves[1].field, leaves[1].op, leaves[1].value) == (
        "affected_host_count",
        CompareOp.GE,
        "50",
    )
    assert referenced_fields(guard) == frozenset({"alert_disposition", "affected_host_count"})


def test_nested_all_of_any_of_parsed_on_ransom_recovery_decision() -> None:
    # all_of[type=ransomware, recovery_path=restore_from_backup, any_of[freshness in {24h, 7d}]]
    guard = _field("ransom_recovery_decision").show_when
    assert isinstance(guard, AllOf)
    assert isinstance(guard.conditions[0], Leaf)
    assert isinstance(guard.conditions[1], Leaf)
    nested = guard.conditions[2]
    assert isinstance(nested, AnyOf)
    assert {leaf.value for leaf in nested.conditions} == {"within_24h", "within_7d"}
    assert all(isinstance(c, Leaf) and c.field == "backup_freshness" for c in nested.conditions)
    assert referenced_fields(guard) == frozenset(
        {"incident_type", "recovery_path", "backup_freshness"}
    )


def test_exfil_or_entry_parsed_on_exfil_confirmed() -> None:
    # exfil_confirmed is gated by any_of[type=data_exfiltration, info_impact=confirmed_loss].
    guard = _field("exfil_confirmed").show_when
    assert isinstance(guard, AnyOf)
    refs = {(c.field, c.value) for c in guard.conditions if isinstance(c, Leaf)}
    assert refs == {
        ("incident_type", "data_exfiltration"),
        ("information_impact", "confirmed_loss"),
    }


def test_breach_notification_nested_all_of_any_of() -> None:
    # all_of[any_of[type=data_exfiltration, info_impact=confirmed_loss], exfil_confirmed=yes,
    #        exfil_data_class=regulated] with single-evidence exfil_egress_pull.
    guard = _field("breach_notification_decision").show_when
    assert isinstance(guard, AllOf)
    assert isinstance(guard.conditions[0], AnyOf)  # the OR-entry, nested first
    assert referenced_fields(guard) == frozenset(
        {"incident_type", "information_impact", "exfil_confirmed", "exfil_data_class"}
    )


def _engine() -> StateEngine:
    return StateEngine(_workflow())


def _confirmed(engine: StateEngine):
    """A confirmed incident with severity/classification answered (the common stem)."""
    state = engine.initial_state()
    for name, value in [
        ("reporter_channel", "siem_alert"),
        ("alert_summary", "Suspicious alert observed on the network."),
        ("alert_disposition", "confirmed_incident"),
        ("functional_impact", "high"),
        ("information_impact", "none"),
        ("affected_host_count", "3"),
        ("severity", "high"),
        ("asset_class", "endpoint"),
    ]:
        state = engine.with_answer(state, name, value)
    return state


def test_false_positive_closes_with_no_branch_fields() -> None:
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("reporter_channel", "siem_alert"),
        ("alert_summary", "Benign scanner traffic, no real incident."),
        ("alert_disposition", "false_positive"),
    ]:
        state = engine.with_answer(state, name, value)
    active = {f.name for f in engine.active_fields(state.answers)}
    # Only the always-active triage stem is active; fields 5-42 are all dormant.
    assert active == {"reporter_channel", "alert_summary", "triage_validation", "alert_disposition"}


def test_phishing_active_set_at_depth() -> None:
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "phishing_bec")
    state = engine.with_answer(state, "is_malicious_email", "yes")
    state = engine.with_answer(state, "anyone_clicked", "yes")
    state = engine.with_answer(state, "entered_credentials", "yes")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert {
        "phish_header_pull",
        "is_malicious_email",
        "spoof_class",
        "recipient_scope_pull",
        "anyone_clicked",
        "payload_exec_scan",
        "entered_credentials",
        "bec_persistence_pull",
        "phish_takeover_confirmed",
    } <= active
    # no other branch leaks in
    assert "variant_ioc_lookup" not in active
    assert "compromise_confirm_pull" not in active
    assert "exfil_confirmed" not in active  # info_impact=none, type=phishing


def test_ransomware_active_set_full_restore_path() -> None:
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "ransomware")
    state = engine.with_answer(state, "encryption_scope", "full")
    state = engine.with_answer(state, "recovery_path", "restore_from_backup")
    state = engine.with_answer(state, "backup_freshness", "within_24h")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert {
        "variant_ioc_lookup",
        "encryption_scope",
        "ransom_scope_scan",
        "recovery_path",
        "data_impact_scan",
        "backup_freshness",
        "backup_integrity_check",
        "ransom_recovery_decision",
    } <= active


def test_credentials_active_set_persistence_yes() -> None:
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "compromised_credentials")
    state = engine.with_answer(state, "persistence_established", "yes")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert {
        "compromise_confirm_pull",
        "principal_type",
        "attacker_activity_pull",
        "persistence_established",
        "backdoor_enum_pull",
        "cred_eradication_plan",
    } <= active


def test_exfiltration_active_set_regulated_to_breach_conclusion() -> None:
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "data_exfiltration")
    state = engine.with_answer(state, "exfil_confirmed", "yes")
    state = engine.with_answer(state, "exfil_data_class", "regulated")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert {
        "exfil_confirmed",
        "exfil_egress_pull",
        "exfil_data_class",
        "breach_notification_decision",
    } <= active


def test_exfiltration_reachable_via_information_impact_or_entry() -> None:
    # The any_of OR-entry: exfil branch opens on info_impact=confirmed_loss for a non-exfil type.
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("reporter_channel", "siem_alert"),
        ("alert_summary", "Credential abuse with confirmed data loss."),
        ("alert_disposition", "confirmed_incident"),
        ("functional_impact", "high"),
        ("information_impact", "confirmed_loss"),
        ("affected_host_count", "3"),
        ("severity", "critical"),
        ("incident_type", "compromised_credentials"),
        ("asset_class", "identity"),
    ]:
        state = engine.with_answer(state, name, value)
    active = {f.name for f in engine.active_fields(state.answers)}
    assert "exfil_confirmed" in active  # OR-entry: confirmed_loss opens exfil on a non-exfil type
    # and with info_impact=none it would NOT (the other leg)
    state2 = engine.with_answer(state, "information_impact", "none")
    active2 = {f.name for f in engine.active_fields(state2.answers)}
    assert "exfil_confirmed" not in active2


def test_incident_type_switch_prunes_whole_subtree_in_one_transition() -> None:
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "phishing_bec")
    state = engine.with_answer(state, "is_malicious_email", "yes")
    state = engine.with_answer(state, "anyone_clicked", "yes")
    state = engine.with_evidence(state, "phish_header_pull")
    assert "is_malicious_email" in state.answers
    assert state.evidence == frozenset({"phish_header_pull"})
    # Switch the master driver: the entire phishing subtree (answers + evidence) prunes at once.
    state = engine.with_answer(state, "incident_type", "ransomware")
    assert "is_malicious_email" not in state.answers
    assert "anyone_clicked" not in state.answers
    assert state.evidence == frozenset()


def test_sub_driver_switch_prunes_deep_cascade() -> None:
    # Flipping recovery_path away from restore_from_backup deactivates fields 25-28 to fixpoint.
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "ransomware")
    state = engine.with_answer(state, "encryption_scope", "full")
    state = engine.with_answer(state, "recovery_path", "restore_from_backup")
    state = engine.with_answer(state, "backup_freshness", "within_24h")
    assert "backup_freshness" in state.answers
    state = engine.with_answer(state, "recovery_path", "rebuild")
    after = {f.name for f in engine.active_fields(state.answers)}
    assert "data_impact_scan" not in after
    assert "backup_freshness" not in after
    assert "backup_integrity_check" not in after
    assert "ransom_recovery_decision" not in after
    assert "backup_freshness" not in state.answers  # the deactivated answer was pruned


def test_numeric_guard_active_at_or_above_50_inactive_below() -> None:
    engine = _engine()
    base = engine.initial_state()
    for name, value in [
        ("reporter_channel", "siem_alert"),
        ("alert_summary", "Widespread compromise across the fleet."),
        ("alert_disposition", "confirmed_incident"),
    ]:
        base = engine.with_answer(base, name, value)
    at = engine.with_answer(base, "affected_host_count", "50")
    above = engine.with_answer(base, "affected_host_count", "120")
    below = engine.with_answer(base, "affected_host_count", "49")
    assert "mass_incident_bridge_note" in {f.name for f in engine.active_fields(at.answers)}
    assert "mass_incident_bridge_note" in {f.name for f in engine.active_fields(above.answers)}
    assert "mass_incident_bridge_note" not in {f.name for f in engine.active_fields(below.answers)}


def test_nested_all_of_any_of_both_ways() -> None:
    # backup_integrity_check active when freshness in {24h, 7d}, inactive when older_or_unknown.
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "ransomware")
    state = engine.with_answer(state, "encryption_scope", "partial")
    state = engine.with_answer(state, "recovery_path", "restore_from_backup")
    fresh = engine.with_answer(state, "backup_freshness", "within_7d")
    stale = engine.with_answer(state, "backup_freshness", "older_or_unknown")
    assert "backup_integrity_check" in {f.name for f in engine.active_fields(fresh.answers)}
    assert "ransom_recovery_decision" in {f.name for f in engine.active_fields(fresh.answers)}
    assert "backup_integrity_check" not in {f.name for f in engine.active_fields(stale.answers)}
    assert "ransom_recovery_decision" not in {f.name for f in engine.active_fields(stale.answers)}


def test_phish_conclusion_blocked_until_both_evidence_then_allowed() -> None:
    engine = _engine()
    state = _confirmed(engine)
    for name, value in [
        ("incident_type", "phishing_bec"),
        ("is_malicious_email", "yes"),
        ("anyone_clicked", "yes"),
        ("entered_credentials", "yes"),
    ]:
        state = engine.with_answer(state, name, value)
    # C5: blocked while either evidence is missing.
    blocked = answer_gate(
        engine, state, "phish_takeover_confirmed", "takeover", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False  # missing bec_persistence_pull, payload_exec_scan
    # C4: identical attempt admitted + logged.
    admitted = answer_gate(
        engine, state, "phish_takeover_confirmed", "takeover", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True
    # Collect both required evidence, then C5 allows it.
    state = engine.with_evidence(state, "payload_exec_scan")
    state = engine.with_evidence(state, "bec_persistence_pull")
    allowed = answer_gate(
        engine,
        state,
        "phish_takeover_confirmed",
        "Mailbox-rule persistence plus payload execution confirm account takeover.",
        enforce=True,
        turn=2,
        today=TODAY,
    )
    assert allowed.accepted is True


def test_ransom_conclusion_requires_all_three_evidence() -> None:
    engine = _engine()
    state = _confirmed(engine)
    for name, value in [
        ("incident_type", "ransomware"),
        ("encryption_scope", "full"),
        ("recovery_path", "restore_from_backup"),
        ("backup_freshness", "within_24h"),
    ]:
        state = engine.with_answer(state, name, value)
    state = engine.with_evidence(state, "ransom_scope_scan")
    state = engine.with_evidence(state, "data_impact_scan")
    # Still missing backup_integrity_check -> C5 blocks.
    blocked = answer_gate(
        engine, state, "ransom_recovery_decision", "restore", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False
    state = engine.with_evidence(state, "backup_integrity_check")
    allowed = answer_gate(
        engine,
        state,
        "ransom_recovery_decision",
        "Clean 24h backup verified; restore rather than pay.",
        enforce=True,
        turn=2,
        today=TODAY,
    )
    assert allowed.accepted is True


def test_ransomware_escalate_by_design_conclusion_unreachable_on_rebuild() -> None:
    # recovery_path=rebuild leaves the recovery conclusion forever inactive -> escalate is terminal.
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "ransomware")
    state = engine.with_answer(state, "encryption_scope", "full")
    state = engine.with_answer(state, "recovery_path", "rebuild")
    active = {f.name for f in engine.active_fields(state.answers)}
    assert "ransom_recovery_decision" not in active  # never activates on the rebuild sub-path
    assert "backup_integrity_check" not in active
    # The stale-backup sub-path is the other escalate trigger.
    state2 = engine.with_answer(
        engine.with_answer(state, "recovery_path", "restore_from_backup"),
        "backup_freshness",
        "older_or_unknown",
    )
    active2 = {f.name for f in engine.active_fields(state2.answers)}
    assert "ransom_recovery_decision" not in active2


def test_inactive_field_answer_rejected_under_enforce_admitted_without() -> None:
    # Answering a dormant branch field is an admissible inactive_field violation.
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "ransomware")
    # is_malicious_email belongs to the (dormant) phishing branch.
    blocked = answer_gate(
        engine, state, "is_malicious_email", "yes", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False
    admitted = answer_gate(
        engine, state, "is_malicious_email", "yes", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True


def test_evidence_field_cannot_be_answered_directly() -> None:
    # Evidence-faking via submit_answer is an admissible wrong_kind violation.
    engine = _engine()
    state = _confirmed(engine)
    state = engine.with_answer(state, "incident_type", "ransomware")
    blocked = answer_gate(
        engine, state, "variant_ioc_lookup", "clean", enforce=True, turn=1, today=TODAY
    )
    assert blocked.accepted is False
    admitted = answer_gate(
        engine, state, "variant_ioc_lookup", "clean", enforce=False, turn=1, today=TODAY
    )
    assert admitted.accepted is True


def test_affected_host_count_range_rejected_out_of_band() -> None:
    engine = _engine()
    state = _confirmed(engine)  # already sets affected_host_count="3"; re-answer with a bad value
    bad = answer_gate(
        engine, state, "affected_host_count", "100001", enforce=True, turn=1, today=TODAY
    )
    assert bad.accepted is False  # range max is 100000


def test_unconfirmed_commit_blocked_under_verified_admitted_without() -> None:
    # The verified-profile consequence boundary: commit without confirm.
    engine = _engine()
    state = engine.initial_state()
    for name, value in [
        ("reporter_channel", "siem_alert"),
        ("alert_summary", "Benign scanner traffic, no real incident."),
        ("alert_disposition", "false_positive"),
    ]:
        outcome = answer_gate(engine, state, name, value, enforce=True, turn=1, today=TODAY)
        assert outcome.accepted, (name, outcome.message)
        state = outcome.state
    state = record_evidence(
        engine, state, "triage_validation", "ioc_lookup", enforce=True, turn=2
    ).state
    assert engine.is_complete(state)  # false_positive close: only the stem is mandatory
    blocked = commit_gate(engine, state, enforce=True, turn=3)
    assert blocked.accepted is False  # verified: needs confirm first
    admitted = commit_gate(engine, state, enforce=False, turn=3)
    assert admitted.accepted is True  # C1/C4: admitted + logged as not_confirmed
