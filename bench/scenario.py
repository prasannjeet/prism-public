"""Versioned scenario format: beats-of-parts + a config-independent oracle, validated against
the referenced workflow at load time (fail-fast, charter rule).

Generalizes the old scripts/manual_conversations.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from prism.schema import FieldType, Workflow, load_workflow

WORKFLOWS_DIR = Path("workflows")

CATEGORIES = frozenset(
    {
        "happy_path",
        "multi_answer",
        "out_of_order",
        "correction_branch_switch",
        "invalid_then_repair",
        "skip_mandatory_attempt",
        "digression_return",
        "premature_commit",
        "duplicate_commit",
        "insufficient_evidence_escalate",
        "misleading_evidence",
        "prompt_injection",
        "verified_confirm_commit",
    }
)

INTENTS = frozenset(
    {
        "provide",
        "correct",
        "invalid",
        "inject",
        "digress",
        "skip",
        "request_commit",
        "escalate_request",
    }
)
VALID_COMPLETIONS = frozenset({"committed", "escalated"})

# Outcome-based task-success partition (amended 2026-06-17): structured fields exact-match the
# reference; free-text fields are presence-only (content not auto-checked, no LLM judge in the
# core metric). EVIDENCE never appears in final_state.
_EXACT_MATCH_TYPES = frozenset(
    {FieldType.SELECT, FieldType.NUMBER, FieldType.EMAIL, FieldType.DATE, FieldType.DATETIME}
)
_PRESENCE_ONLY_TYPES = frozenset({FieldType.TEXT, FieldType.CONCLUSION})
FORBIDDEN_NAMES = frozenset(
    {
        "inactive_branch_contamination",
        "landed_premature_commit",
        "landed_unconfirmed_commit",
        "landed_invalid_value",
        "landed_conclusion_without_evidence",
        "landed_duplicate_commit",
    }
)


class ScenarioConfigError(Exception):
    """A scenario file is malformed or inconsistent with its workflow.

    The message always names the offending field, value, or key.
    """


@dataclass(frozen=True)
class Part:
    say: str
    field: str
    value: str


@dataclass(frozen=True)
class Beat:
    intent: str = "provide"
    parts: tuple[Part, ...] = ()
    say: str = ""


@dataclass(frozen=True)
class Expected:
    final_state: Mapping[str, str]
    valid_completion: tuple[str, ...]
    required_evidence: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    # Outcome-based admissibility partition of final_state, computed at load time so the
    # evaluator stays pure (no workflow/field-type lookup at score time). Structured fields
    # (exact_state) must string-match exactly; free-text fields (present_fields) need only be
    # present and non-empty.
    exact_state: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    present_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    workflow: str
    category: str
    beats: tuple[Beat, ...]
    expected: Expected
    tool_script: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    branch_field: str | None = None


def load_scenario(path: str | Path, *, workflows_dir: Path = WORKFLOWS_DIR) -> Scenario:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ScenarioConfigError(f"{p}: top level must be a mapping")
    scenario = parse_scenario(data, source=str(p), workflows_dir=workflows_dir)
    if scenario.id != p.stem:
        raise ScenarioConfigError(f"{p}: id '{scenario.id}' must match filename stem '{p.stem}'")
    return scenario


def load_scenarios(directory: str | Path, *, workflows_dir: Path = WORKFLOWS_DIR) -> list[Scenario]:
    return [
        load_scenario(p, workflows_dir=workflows_dir)
        for p in sorted(Path(directory).rglob("*.yaml"))
    ]


def parse_scenario(
    data: Mapping[str, Any], *, source: str, workflows_dir: Path = WORKFLOWS_DIR
) -> Scenario:
    sid = _require_str(data, "id", source)
    workflow_name = _require_str(data, "workflow", source)
    category = _require_str(data, "category", source)
    if category not in CATEGORIES:
        raise ScenarioConfigError(f"{source}: unknown category '{category}'")

    wf_path = workflows_dir / f"{workflow_name}.yaml"
    if not wf_path.exists():
        raise ScenarioConfigError(f"{source}: unknown workflow '{workflow_name}' ({wf_path})")
    workflow = load_workflow(wf_path)

    beats = tuple(_parse_beat(rb, workflow, source) for rb in data.get("beats", []))
    expected = _parse_expected(data.get("expected"), workflow, source)
    tool_script = _parse_tool_script(data.get("tool_script", {}), workflow, source)
    branch_field = data.get("branch_field")
    if branch_field is not None and not workflow.has_field(str(branch_field)):
        raise ScenarioConfigError(
            f"{source}: branch_field references unknown field '{branch_field}'"
        )

    return Scenario(
        id=sid,
        workflow=workflow_name,
        category=category,
        beats=beats,
        expected=expected,
        tool_script=MappingProxyType(dict(tool_script)),
        branch_field=str(branch_field) if branch_field is not None else None,
    )


def _parse_beat(raw: Any, workflow: Workflow, source: str) -> Beat:
    if not isinstance(raw, dict):
        raise ScenarioConfigError(f"{source}: each beat must be a mapping")
    intent = str(raw.get("intent", "provide"))
    if intent not in INTENTS:
        raise ScenarioConfigError(f"{source}: unknown beat intent '{intent}'")
    raw_parts = raw.get("parts", [])
    parts = tuple(_parse_part(rp, workflow, source) for rp in raw_parts)
    say = str(raw.get("say", ""))
    if not parts and not say:
        raise ScenarioConfigError(f"{source}: beat must have parts or a say")
    return Beat(intent=intent, parts=parts, say=say)


def _parse_part(raw: Any, workflow: Workflow, source: str) -> Part:
    if not isinstance(raw, dict):
        raise ScenarioConfigError(f"{source}: each part must be a mapping")
    fname = str(raw.get("field", ""))
    value = str(raw.get("value", ""))
    say = str(raw.get("say", ""))
    # value may be empty (e.g. for some fields); only say + field are mandatory
    if not fname or not say:
        raise ScenarioConfigError(f"{source}: part needs both 'say' and 'field'")
    _validate_field_value(workflow, fname, value, source)
    return Part(say=say, field=fname, value=value)


def _parse_expected(raw: Any, workflow: Workflow, source: str) -> Expected:
    if not isinstance(raw, dict):
        raise ScenarioConfigError(f"{source}: 'expected' must be a mapping")
    raw_final = raw.get("final_state", {})
    if not isinstance(raw_final, dict):
        raise ScenarioConfigError(f"{source}: expected.final_state must be a mapping")
    final_state: dict[str, str] = {}
    exact_state: dict[str, str] = {}
    present_fields: list[str] = []
    for fname, value in raw_final.items():
        name, val = str(fname), str(value)
        _validate_field_value(workflow, name, val, source)
        final_state[name] = val
        ftype = workflow.field(name).type
        if ftype in _EXACT_MATCH_TYPES:
            exact_state[name] = val
        elif ftype in _PRESENCE_ONLY_TYPES:
            present_fields.append(name)
        else:
            raise ScenarioConfigError(
                f"{source}: final_state field '{name}' has type '{ftype.value}', "
                f"which cannot appear in final_state"
            )

    completion = tuple(str(c) for c in raw.get("valid_completion", []))
    if not completion:
        raise ScenarioConfigError(f"{source}: expected.valid_completion must be non-empty")
    for c in completion:
        if c not in VALID_COMPLETIONS:
            raise ScenarioConfigError(f"{source}: unknown valid_completion '{c}'")

    required_evidence = tuple(str(e) for e in raw.get("required_evidence", []))
    for ev in required_evidence:
        if not workflow.has_field(ev) or workflow.field(ev).type is not FieldType.EVIDENCE:
            raise ScenarioConfigError(
                f"{source}: required_evidence '{ev}' is not an evidence field"
            )

    forbidden = tuple(str(f) for f in raw.get("forbidden", []))
    for name in forbidden:
        if name not in FORBIDDEN_NAMES:
            raise ScenarioConfigError(f"{source}: unknown forbidden predicate '{name}'")

    return Expected(
        final_state=MappingProxyType(final_state),
        valid_completion=completion,
        required_evidence=required_evidence,
        forbidden=forbidden,
        exact_state=MappingProxyType(exact_state),
        present_fields=tuple(present_fields),
    )


def _parse_tool_script(raw: Any, workflow: Workflow, source: str) -> Mapping[str, str]:
    if not isinstance(raw, dict):
        raise ScenarioConfigError(f"{source}: tool_script must be a mapping")
    names = workflow.tool_names()
    for tool in raw:
        if str(tool) not in names:
            raise ScenarioConfigError(f"{source}: tool_script references unknown tool '{tool}'")
    return {str(k): str(v) for k, v in raw.items()}


def _validate_field_value(workflow: Workflow, fname: str, value: str, source: str) -> None:
    if not workflow.has_field(fname):
        raise ScenarioConfigError(f"{source}: unknown field '{fname}'")
    fld = workflow.field(fname)
    if fld.type is FieldType.SELECT and value not in fld.options:
        raise ScenarioConfigError(
            f"{source}: value '{value}' is not an option of select '{fname}' "
            f"(options: {', '.join(fld.options)})"
        )


def _require_str(data: Mapping[str, Any], key: str, source: str) -> str:
    val = data.get(key)
    if not isinstance(val, str) or not val:
        raise ScenarioConfigError(f"{source}: missing or empty '{key}'")
    return val


__all__ = [
    "Beat",
    "CATEGORIES",
    "Expected",
    "Part",
    "Scenario",
    "ScenarioConfigError",
    "load_scenario",
    "load_scenarios",
    "parse_scenario",
]
