"""evaluate() is a pure function of (event_log, scenario). Tested against hand-crafted logs:
a clean committed log scores success; logs with landed violations / wrong final state fail."""

from __future__ import annotations

from pathlib import Path

from bench.evaluator import evaluate
from bench.scenario import Beat, Expected, Part, Scenario
from prism.engine import StateEngine
from prism.schema import load_workflow


def _scenario(**over: object) -> Scenario:
    base: dict[str, object] = dict(
        id="s",
        workflow="form_booking",
        category="happy_path",
        beats=(),
        expected=Expected(
            final_state={"service_type": "move_out"},
            valid_completion=("committed",),
            exact_state={"service_type": "move_out"},  # SELECT -> exact-match
        ),
        branch_field="service_type",
    )
    base.update(over)
    return Scenario(**base)  # type: ignore[arg-type]


def _committed_log() -> list[dict[str, object]]:
    return [
        {"seq": 0, "turn": 1, "type": "user_message", "text": "hi"},
        {
            "seq": 1,
            "turn": 1,
            "type": "tool_called",
            "tool": "submit_answer",
            "arguments": {"field": "service_type", "value": "move_out"},
        },
        {
            "seq": 2,
            "turn": 1,
            "type": "answer_submitted",
            "field": "service_type",
            "value": "move_out",
        },
        {"seq": 3, "turn": 1, "type": "tool_result", "tool": "submit_answer", "content": "ok"},
        {
            "seq": 4,
            "turn": 1,
            "type": "committed",
            "final_answers": {"service_type": "move_out"},
            "evidence": [],
        },
    ]


def test_clean_committed_log_scores_success() -> None:
    m = evaluate(_committed_log(), _scenario())
    assert m.task_success is True
    assert m.terminal == "committed"
    assert m.branch == "move_out"
    assert m.turns == 1
    assert m.tool_calls == 1
    assert m.tool_errors == 0
    assert m.admitted == {}
    assert m.contamination == 0


def test_wrong_final_state_fails() -> None:
    log = _committed_log()
    log[-1]["final_answers"] = {"service_type": "window_clean"}
    m = evaluate(log, _scenario())
    assert m.task_success is False
    assert m.terminal == "committed"


def test_forbidden_landed_violation_fails_and_counts() -> None:
    log = _committed_log()
    log.insert(
        2,
        {
            "seq": 99,
            "turn": 1,
            "type": "violation_admitted",
            "gate": "answer",
            "field": "ghost",
            "value": "v",
            "reason": "inactive_field",
            "code": None,
        },
    )
    scenario = _scenario(
        expected=Expected(
            final_state={"service_type": "move_out"},
            valid_completion=("committed",),
            forbidden=("inactive_branch_contamination",),
        ),
        branch_field="service_type",
    )
    m = evaluate(log, scenario)
    assert m.task_success is False
    assert m.contamination == 1
    assert m.attempted.get("inactive_field") == 1
    assert m.admitted.get("inactive_field") == 1


def test_escalated_terminal_scores_success_when_expected() -> None:
    log = [
        {
            "seq": 0,
            "turn": 1,
            "type": "evidence_collected",
            "field": "deploy_check",
            "via": "check_deployments",
        },
        {"seq": 1, "turn": 1, "type": "escalated", "action": "escalate_to_engineer"},
    ]
    scenario = _scenario(
        workflow="incident_investigation",
        category="insufficient_evidence_escalate",
        expected=Expected(
            final_state={},
            valid_completion=("escalated",),
            required_evidence=("deploy_check",),
        ),
        branch_field=None,
    )
    m = evaluate(log, scenario)
    assert m.terminal == "escalated"
    assert m.task_success is True


def test_required_evidence_missing_before_terminal_fails() -> None:
    log = [{"seq": 0, "turn": 1, "type": "escalated", "action": "x"}]  # no evidence_collected
    scenario = _scenario(
        workflow="incident_investigation",
        category="insufficient_evidence_escalate",
        expected=Expected(
            final_state={}, valid_completion=("escalated",), required_evidence=("deploy_check",)
        ),
        branch_field=None,
    )
    assert evaluate(log, scenario).task_success is False


def test_recovered_true_for_correction_scenario_that_succeeds() -> None:
    scenario = _scenario(
        beats=(
            Beat(intent="correct", parts=(Part("make it move out", "service_type", "move_out"),)),
        )
    )
    assert evaluate(_committed_log(), scenario).recovered is True


# --- outcome-based admissibility (amended 2026-06-17): structured exact, free-text presence ---

# server_down branch of incident_investigation: structured {symptom, outage_time} + free-text
# {affected_service (TEXT), server_root_cause (CONCLUSION)}.
_EXPECTED_FINAL = {
    "affected_service": "checkout-api",
    "symptom": "server_down",
    "outage_time": "2026-06-10T14:00",
    "server_root_cause": "Deploy d-42 introduced a null-pointer regression seen in the logs.",
}


def _outcome_scenario(**over: object) -> Scenario:
    base: dict[str, object] = dict(
        id="s",
        workflow="incident_investigation",
        category="happy_path",
        beats=(),
        expected=Expected(
            final_state=dict(_EXPECTED_FINAL),
            valid_completion=("committed",),
            # Load-time partition; the evaluator reads these, not final_state, for admissibility.
            exact_state={"symptom": "server_down", "outage_time": "2026-06-10T14:00"},
            present_fields=("affected_service", "server_root_cause"),
        ),
        branch_field=None,
    )
    base.update(over)
    return Scenario(**base)  # type: ignore[arg-type]


def _outcome_log(final_answers: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"seq": 0, "turn": 1, "type": "user_message", "text": "incident"},
        {"seq": 1, "turn": 1, "type": "committed", "final_answers": final_answers, "evidence": []},
    ]


def test_free_text_wording_differs_still_success() -> None:
    # THE fix: the committed conclusion is worded completely differently from the reference, but
    # structured fields match and the free-text fields are non-empty -> task_success True.
    answers = dict(_EXPECTED_FINAL)
    answers["server_root_cause"] = "Root cause: a bad deploy. Totally different phrasing here."
    answers["affected_service"] = "the checkout API service (rephrased)"
    m = evaluate(_outcome_log(answers), _outcome_scenario())
    assert m.task_success is True


def test_wrong_structured_value_fails() -> None:
    answers = dict(_EXPECTED_FINAL)
    answers["outage_time"] = "2026-06-10T15:00"  # structured field, wrong value
    assert evaluate(_outcome_log(answers), _outcome_scenario()).task_success is False


def test_missing_required_field_fails() -> None:
    answers = dict(_EXPECTED_FINAL)
    del answers["affected_service"]  # key-set mismatch (missing)
    assert evaluate(_outcome_log(answers), _outcome_scenario()).task_success is False


def test_empty_free_text_field_fails() -> None:
    answers = dict(_EXPECTED_FINAL)
    answers["server_root_cause"] = "   "  # present but whitespace-only
    assert evaluate(_outcome_log(answers), _outcome_scenario()).task_success is False


def test_extra_unexpected_field_fails() -> None:
    answers = dict(_EXPECTED_FINAL)
    answers["latency_window"] = "10:00-11:00"  # key-set mismatch (extra)
    assert evaluate(_outcome_log(answers), _outcome_scenario()).task_success is False


# --- refined ("forgiving") admissibility (added 2026-06-24, Phase 3 task 1) -------------------
# Strict requires the committed field-set to equal the reference exactly. Refined forgives an
# extra committed field iff it is workflow-defined + optional + active, and forgives a missing
# OPTIONAL reference field. Missing required, hallucinated, and inactive/wrong-branch extras
# still fail.


def _form_engine() -> StateEngine:
    return StateEngine(load_workflow(Path("workflows/form_booking.yaml")))


def _form_scenario(**over: object) -> Scenario:
    base: dict[str, object] = dict(
        id="s",
        workflow="form_booking",
        category="happy_path",
        beats=(),
        expected=Expected(
            final_state={"service_type": "move_out"},
            valid_completion=("committed",),
            exact_state={"service_type": "move_out"},  # SELECT -> exact-match
        ),
        branch_field="service_type",
    )
    base.update(over)
    return Scenario(**base)  # type: ignore[arg-type]


def _form_committed_log(final_answers: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"seq": 0, "turn": 1, "type": "user_message", "text": "hi"},
        {"seq": 1, "turn": 1, "type": "committed", "final_answers": final_answers, "evidence": []},
    ]


def test_refined_rescues_optional_active_extra_field() -> None:
    # access_notes is defined, optional (required=False), and always active (no show_when).
    log = _form_committed_log({"service_type": "move_out", "access_notes": "gate code 1234"})
    m = evaluate(log, _form_scenario(), _form_engine())
    assert m.task_success is False  # strict: key-set mismatch
    assert m.task_success_refined is True  # refined: extra is optional + active + defined


def test_refined_still_fails_hallucinated_extra() -> None:
    log = _form_committed_log({"service_type": "move_out", "ghost_field": "x"})
    m = evaluate(log, _form_scenario(), _form_engine())
    assert m.task_success is False
    assert m.task_success_refined is False  # not a workflow field


def test_refined_still_fails_inactive_optional_extra() -> None:
    # pet_on_site is optional but only active when service_type == deep_clean; here it is move_out.
    log = _form_committed_log({"service_type": "move_out", "pet_on_site": "yes"})
    m = evaluate(log, _form_scenario(), _form_engine())
    assert m.task_success_refined is False  # defined + optional but INACTIVE -> still fails


def test_refined_still_fails_required_extra() -> None:
    # num_rooms is active for move_out but is required, not optional.
    # A required extra is rejected even when active+valid: committing a required field the
    # reference omits means the model expanded scope beyond the branch, so it is not forgiven.
    log = _form_committed_log({"service_type": "move_out", "num_rooms": "3"})
    m = evaluate(log, _form_scenario(), _form_engine())
    assert m.task_success_refined is False  # defined + active but REQUIRED -> still fails


def test_refined_still_fails_missing_required_reference_field() -> None:
    # service_type (required) is absent; an optional active extra does not rescue it.
    log = _form_committed_log({"access_notes": "gate code"})
    m = evaluate(log, _form_scenario(), _form_engine())
    assert m.task_success_refined is False


def test_refined_forgives_missing_optional_reference_field() -> None:
    # E lists an optional field (access_notes); committing without it is forgiven under refined.
    scenario = _form_scenario(
        expected=Expected(
            final_state={"service_type": "move_out", "access_notes": "gate code"},
            valid_completion=("committed",),
            exact_state={"service_type": "move_out"},
            present_fields=("access_notes",),
        ),
        branch_field="service_type",
    )
    log = _form_committed_log({"service_type": "move_out"})
    m = evaluate(log, scenario, _form_engine())
    assert m.task_success is False  # strict: missing key
    assert m.task_success_refined is True  # refined: optional reference field, absent is OK


def test_refined_equals_strict_when_field_sets_match() -> None:
    log = _form_committed_log({"service_type": "move_out"})
    m = evaluate(log, _form_scenario(), _form_engine())
    assert m.task_success is True
    assert m.task_success_refined is True


def test_refined_falls_back_to_strict_without_engine() -> None:
    log = _form_committed_log({"service_type": "move_out", "access_notes": "x"})
    m = evaluate(log, _form_scenario())  # no engine
    assert m.task_success is False
    assert m.task_success_refined == m.task_success  # fallback: refined == strict
