"""Tier-1/2/3 projection of engine state for the LLM — pure rendering, declaration order only."""

from __future__ import annotations

import json

from .engine import State, StateEngine
from .schema import AllOf, AnyOf, Field, Leaf, Profile, Workflow, dump_workflow, referenced_fields


def _render_condition(cond: object) -> str:
    match cond:
        case Leaf(field=fld, op=op, value=val):
            return f"{fld} {op.value} {val}"
        case AllOf(conditions=cs):
            return "all(" + ", ".join(_render_condition(c) for c in cs) + ")"
        case AnyOf(conditions=cs):
            return "any(" + ", ".join(_render_condition(c) for c in cs) + ")"
    return ""


def render_schema(workflow: Workflow) -> str:
    """Tier 1 — static structure block, rendered once into the system prompt."""
    mandatory = [f.name for f in workflow.fields if f.required and f.show_when is None]
    lines = ["=== WORKFLOW STRUCTURE ===", f"Mandatory: {_join(mandatory)}"]
    lines.extend(_schema_line(f) for f in workflow.fields)
    return "\n".join(lines)


def render_state(engine: StateEngine, state: State) -> str:
    """Tier 2 — compact progress block, re-rendered every turn."""
    workflow = engine.workflow
    answered = [f.name for f in workflow.fields if f.name in state.answers]
    answered_text = _join([f'{name} = "{state.answers[name]}"' for name in answered])
    remaining = [f.name for f in engine.remaining_mandatory(state)]
    evidence = [f.name for f in workflow.fields if f.name in state.evidence]
    lines = [
        f"Answered ({len(answered)}): {answered_text}",
        f"Active Branch: {_active_branch(engine, state)}",
        f"Remaining Mandatory ({len(remaining)}): {_join(remaining)}",
        f"Evidence collected ({len(evidence)}): {_join(evidence)}",
    ]
    if workflow.profile is Profile.VERIFIED:
        lines.append(f"Confirmed: {'Yes' if state.confirmed else 'No'}")
    lines.append(f"Complete: {'Yes' if engine.is_complete(state) else 'No'}")
    return "\n".join(lines)


def render_detail(workflow: Workflow, field_name: str) -> str:
    """Tier 3 — full detail for one field; raises KeyError on an unknown name."""
    fld = workflow.field(field_name)
    lines = [
        f"Field: {fld.name}",
        f"Type: {fld.type.value.upper()}",
        f"Required: {'yes' if fld.required else 'no'}",
    ]
    if fld.prompt is not None:
        lines.append(f'Prompt: "{fld.prompt}"')
    if fld.description is not None:
        lines.append(f"Description: {fld.description}")
    if fld.options:
        lines.append(f"Options: {' | '.join(fld.options)}")
    if fld.show_when is not None:
        lines.append(f"Show when: {_render_condition(fld.show_when)}")
    if fld.validators:
        rules = "; ".join(f"{name} = {json.dumps(value)}" for name, value in fld.validators.items())
        lines.append(f"Validators: {rules}")
    if fld.satisfied_by:
        lines.append(f"Satisfied by: {', '.join(fld.satisfied_by)}")
    if fld.requires_evidence:
        lines.append(f"Requires evidence: {', '.join(fld.requires_evidence)}")
    return "\n".join(lines)


def render_current_field(engine: StateEngine, state: State) -> str:
    """C2's whole world: the single next outstanding mandatory field, nothing else."""
    remaining = engine.remaining_mandatory(state)
    if not remaining:
        return "All mandatory fields are complete."
    return "Next required field:\n" + render_detail(engine.workflow, remaining[0].name)


def render_raw(workflow: Workflow) -> str:
    """C3's per-turn payload: the whole workflow, deterministically re-serialized."""
    return dump_workflow(workflow)


def _schema_line(fld: Field) -> str:
    parts = [fld.type.value.upper(), "required" if fld.required else "optional"]
    if fld.options:
        parts.append(f"options: {'|'.join(fld.options)}")
    if fld.show_when is not None:
        parts.append(f"show_when: {_render_condition(fld.show_when)}")
    if fld.satisfied_by:
        parts.append(f"via: {'|'.join(fld.satisfied_by)}")
    if fld.requires_evidence:
        parts.append(f"needs: {'|'.join(fld.requires_evidence)}")
    line = f"- {fld.name} [{', '.join(parts)}]"
    if fld.prompt is not None:
        line += f' - "{fld.prompt}"'
    return line


def _active_branch(engine: StateEngine, state: State) -> str:
    """Answered fields whose value currently gates at least one active child field."""
    workflow = engine.workflow
    active = engine.active_fields(state.answers)
    segments: list[str] = []
    for fld in workflow.fields:
        if fld.name not in state.answers:
            continue
        gates_active_child = any(
            child.show_when is not None
            and fld.name in referenced_fields(child.show_when)
            and child in active
            for child in workflow.fields
        )
        if gates_active_child:
            segments.append(f"{fld.name} → {state.answers[fld.name]}")
    return ", ".join(segments) or "none"


def _join(items: list[str]) -> str:
    return ", ".join(items) or "—"
