"""Workflow domain model + fail-fast loader. Pure: stdlib + pyyaml only."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class WorkflowConfigError(Exception):
    """A workflow YAML is malformed. Message names the offending field."""


class FieldType(Enum):
    TEXT = "text"
    SELECT = "select"
    NUMBER = "number"
    EMAIL = "email"
    DATE = "date"
    DATETIME = "datetime"
    # Two additions a plain form doesn't have — guarded by the node-kind smell test.
    EVIDENCE = "evidence"  # satisfied by tool calls, not user answers
    CONCLUSION = "conclusion"  # terminal node; may require evidence before it can be reached

    @classmethod
    def parse(cls, raw: str, field_name: str) -> FieldType:
        try:
            return cls(raw)
        except ValueError:
            known = ", ".join(t.value for t in cls)
            raise WorkflowConfigError(
                f"field '{field_name}': unknown type '{raw}' (known: {known})"
            ) from None


class Profile(Enum):
    VERIFIED = "verified"  # deterministic summary, wait for an explicit confirm before commit
    EXPRESS = "express"  # commit as soon as the engine reports complete

    @classmethod
    def parse(cls, raw: str) -> Profile:
        try:
            return cls(raw)
        except ValueError:
            known = ", ".join(p.value for p in cls)
            raise WorkflowConfigError(f"unknown profile '{raw}' (known: {known})") from None


class CompareOp(Enum):
    EQ = "="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="

    @classmethod
    def parse(cls, raw: str, owner: str) -> CompareOp:
        try:
            return cls(raw)
        except ValueError:
            known = ", ".join(o.value for o in cls)
            raise WorkflowConfigError(
                f"field '{owner}': unknown operator '{raw}' (known: {known})"
            ) from None


@dataclass(frozen=True)
class Leaf:
    """A single predicate: <field> <op> <value>. The value is stored raw (a string)."""

    field: str
    op: CompareOp
    value: str


@dataclass(frozen=True)
class AllOf:
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class AnyOf:
    conditions: tuple[Condition, ...]


Condition = Leaf | AllOf | AnyOf

# Longest-first so ">=" / "<=" / "!=" are matched before ">" / "<" / "=".
_OPERATORS: tuple[str, ...] = (">=", "<=", "!=", "=", ">", "<")


def parse_condition(raw: object, owner: str) -> Condition:
    """Parse a show_when guard: the "field op value" string shorthand, or a structured
    all_of / any_of / in / {field,op,value} mapping. Fail-fast, naming the owner field.

    A value that contains an operator token (e.g. value "<=") must use the explicit
    {field, op, value} mapping form rather than the string shorthand."""
    if isinstance(raw, str):
        return _parse_leaf_string(raw, owner)
    if isinstance(raw, Mapping):
        if "all_of" in raw:
            return AllOf(_parse_condition_list(raw["all_of"], owner, "all_of"))
        if "any_of" in raw:
            return AnyOf(_parse_condition_list(raw["any_of"], owner, "any_of"))
        if "in" in raw:
            return _parse_in(raw, owner)
        if "field" in raw and "op" in raw and "value" in raw:
            return Leaf(
                str(raw["field"]), CompareOp.parse(str(raw["op"]), owner), str(raw["value"])
            )
        raise WorkflowConfigError(
            f"field '{owner}': show_when mapping must have 'all_of', 'any_of', 'in', "
            f"or 'field'+'op'+'value'"
        )
    raise WorkflowConfigError(
        f"field '{owner}': show_when must be a string or mapping, got {type(raw).__name__}"
    )


def _parse_leaf_string(raw: str, owner: str) -> Leaf:
    for op in _OPERATORS:
        if op in raw:
            lhs, _, rhs = raw.partition(op)
            lhs, rhs = lhs.strip(), rhs.strip()
            if not lhs or not rhs:
                raise WorkflowConfigError(
                    f"field '{owner}': show_when '{raw}' has an empty side"
                )
            return Leaf(lhs, CompareOp(op), rhs)
    raise WorkflowConfigError(
        f"field '{owner}': show_when '{raw}' must contain one of {', '.join(_OPERATORS)}"
    )


def _parse_condition_list(raw: object, owner: str, key: str) -> tuple[Condition, ...]:
    if not isinstance(raw, list) or not raw:
        raise WorkflowConfigError(f"field '{owner}': '{key}' must be a non-empty list")
    return tuple(parse_condition(item, owner) for item in raw)


def _parse_in(raw: Mapping[str, Any], owner: str) -> AnyOf:
    field_name = raw.get("field")
    values = raw.get("in")
    if not isinstance(field_name, str) or not field_name:
        raise WorkflowConfigError(f"field '{owner}': 'in' form needs a 'field' name")
    if not isinstance(values, list) or not values:
        raise WorkflowConfigError(f"field '{owner}': 'in' must be a non-empty list of values")
    return AnyOf(tuple(Leaf(field_name, CompareOp.EQ, str(v)) for v in values))


def _iter_leaves(cond: Condition) -> Iterator[Leaf]:
    match cond:
        case Leaf():
            yield cond
        case AllOf(conditions=cs) | AnyOf(conditions=cs):
            for c in cs:
                yield from _iter_leaves(c)


def referenced_fields(cond: Condition) -> frozenset[str]:
    """Every field name a condition tree references (for ref/cycle validation + projection)."""
    return frozenset(leaf.field for leaf in _iter_leaves(cond))


def coerce_scalar(field_type: FieldType, text: str) -> float | date | datetime | None:
    """Pure coercion for comparison-leaf evaluation/validation. None if it does not parse
    (or the type is not orderable). No clock, no I/O."""
    try:
        if field_type is FieldType.NUMBER:
            return float(text)
        if field_type is FieldType.DATE:
            return date.fromisoformat(text)
        if field_type is FieldType.DATETIME:
            return datetime.fromisoformat(text)
    except ValueError:
        return None
    return None


@dataclass(frozen=True)
class Field:
    name: str
    type: FieldType
    required: bool = False
    prompt: str | None = None
    description: str | None = None
    options: tuple[str, ...] = ()
    show_when: Condition | None = None
    validators: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    satisfied_by: tuple[str, ...] = ()  # tool names that satisfy an EVIDENCE field
    requires_evidence: tuple[str, ...] = ()  # evidence fields a CONCLUSION needs first


@dataclass(frozen=True)
class Tool:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Escalation:
    when: str
    action: str


@dataclass(frozen=True)
class Workflow:
    name: str
    profile: Profile
    fields: tuple[Field, ...]
    tools: tuple[Tool, ...] = ()
    escalation: Escalation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", MappingProxyType({f.name: f for f in self.fields}))

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def field(self, name: str) -> Field:
        by_name: Mapping[str, Field] = self._by_name  # type: ignore[attr-defined]
        if name not in by_name:
            raise KeyError(name)
        return by_name[name]

    def has_field(self, name: str) -> bool:
        return name in self._by_name  # type: ignore[attr-defined]

    def tool_names(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tools)


# --- loading -----------------------------------------------------------------


def load_workflow(path: str | Path) -> Workflow:
    """Read a workflow YAML from disk, parse, and validate. Raises WorkflowConfigError."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise WorkflowConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return parse_workflow(data)


def parse_workflow(data: Mapping[str, Any]) -> Workflow:
    """Build and validate a Workflow from an already-parsed mapping."""
    name = data.get("workflow")
    if not isinstance(name, str) or not name:
        raise WorkflowConfigError("missing or empty 'workflow' name")

    profile = Profile.parse(str(data.get("profile", Profile.VERIFIED.value)))

    raw_fields = data.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise WorkflowConfigError(f"workflow '{name}': 'fields' must be a non-empty list")
    fields = tuple(_parse_field(rf) for rf in raw_fields)

    tools = tuple(_parse_tool(rt) for rt in data.get("tools", []))

    escalation = _parse_escalation(data.get("escalation"))

    workflow = Workflow(
        name=name, profile=profile, fields=fields, tools=tools, escalation=escalation
    )
    _validate(workflow)
    return workflow


def _parse_field(raw: Any) -> Field:
    if not isinstance(raw, dict):
        raise WorkflowConfigError(f"each field must be a mapping, got {type(raw).__name__}")
    fname = raw.get("name")
    if not isinstance(fname, str) or not fname:
        raise WorkflowConfigError("a field is missing its 'name'")
    ftype = FieldType.parse(str(raw.get("type", "")), fname)

    show_when_raw = raw.get("show_when")
    show_when = parse_condition(show_when_raw, fname) if show_when_raw is not None else None

    validators_raw = raw.get("validate", {})
    if not isinstance(validators_raw, dict):
        raise WorkflowConfigError(f"field '{fname}': 'validate' must be a mapping")

    return Field(
        name=fname,
        type=ftype,
        required=bool(raw.get("required", False)),
        prompt=raw.get("prompt"),
        description=raw.get("description"),
        options=tuple(str(o) for o in raw.get("options", [])),
        show_when=show_when,
        validators=MappingProxyType(dict(validators_raw)),
        satisfied_by=tuple(str(t) for t in raw.get("satisfied_by", [])),
        requires_evidence=tuple(str(e) for e in raw.get("requires_evidence", [])),
    )


def _parse_tool(raw: Any) -> Tool:
    if not isinstance(raw, dict):
        raise WorkflowConfigError(f"each tool must be a mapping, got {type(raw).__name__}")
    tname = raw.get("name")
    if not isinstance(tname, str) or not tname:
        raise WorkflowConfigError("a tool is missing its 'name'")
    return Tool(name=tname, description=raw.get("description"))


def _parse_escalation(raw: Any) -> Escalation | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or "when" not in raw or "action" not in raw:
        raise WorkflowConfigError("'escalation' must be a mapping with 'when' and 'action'")
    return Escalation(when=str(raw["when"]), action=str(raw["action"]))


# --- validation (fail-fast; each check raises with a precise message) --------


def _validate(workflow: Workflow) -> None:
    _validate_unique_names(workflow)
    _validate_show_when_refs(workflow)
    _validate_no_show_when_cycles(workflow)
    _validate_show_when_field_types(workflow)
    _validate_evidence_and_conclusion(workflow)


def _validate_unique_names(workflow: Workflow) -> None:
    seen: set[str] = set()
    for f in workflow.fields:
        if f.name in seen:
            raise WorkflowConfigError(f"duplicate field name '{f.name}'")
        seen.add(f.name)


def _validate_show_when_refs(workflow: Workflow) -> None:
    for f in workflow.fields:
        if f.show_when is None:
            continue
        for ref in referenced_fields(f.show_when):
            if not workflow.has_field(ref):
                raise WorkflowConfigError(
                    f"field '{f.name}': show_when references unknown field '{ref}'"
                )


def _validate_show_when_field_types(workflow: Workflow) -> None:
    for f in workflow.fields:
        if f.show_when is None:
            continue
        for leaf in _iter_leaves(f.show_when):
            ref = workflow.field(leaf.field)
            if leaf.op in (CompareOp.EQ, CompareOp.NE):
                if ref.type is not FieldType.SELECT:
                    raise WorkflowConfigError(
                        f"field '{f.name}': show_when '{leaf.op.value}' on '{leaf.field}' "
                        f"requires a select field, but it is type '{ref.type.value}'"
                    )
                if leaf.value not in ref.options:
                    raise WorkflowConfigError(
                        f"field '{f.name}': show_when value '{leaf.value}' is not an option of "
                        f"select '{leaf.field}' (options: {', '.join(ref.options)})"
                    )
            else:
                if ref.type not in (FieldType.NUMBER, FieldType.DATE, FieldType.DATETIME):
                    raise WorkflowConfigError(
                        f"field '{f.name}': show_when comparison '{leaf.op.value}' requires a "
                        f"number or date field, but '{leaf.field}' is type '{ref.type.value}'"
                    )
                if coerce_scalar(ref.type, leaf.value) is None:
                    raise WorkflowConfigError(
                        f"field '{f.name}': show_when value '{leaf.value}' is not a valid "
                        f"{ref.type.value}"
                    )


def _validate_no_show_when_cycles(workflow: Workflow) -> None:
    deps = {
        f.name: referenced_fields(f.show_when)
        for f in workflow.fields
        if f.show_when is not None
    }
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        visiting.add(node)
        for nxt in deps.get(node, frozenset()):
            if nxt in visiting:
                raise WorkflowConfigError(f"show_when dependency cycle through field '{nxt}'")
            if nxt not in done and nxt in deps:
                visit(nxt)
        visiting.discard(node)
        done.add(node)

    for start in deps:
        if start not in done:
            visit(start)


def _validate_evidence_and_conclusion(workflow: Workflow) -> None:
    tool_names = workflow.tool_names()
    for f in workflow.fields:
        if f.type is FieldType.EVIDENCE:
            if not f.satisfied_by:
                raise WorkflowConfigError(f"evidence field '{f.name}' must declare 'satisfied_by'")
            for tool in f.satisfied_by:
                if tool not in tool_names:
                    raise WorkflowConfigError(
                        f"evidence field '{f.name}': satisfied_by references unknown tool '{tool}'"
                    )
        for ev in f.requires_evidence:
            if not workflow.has_field(ev):
                raise WorkflowConfigError(
                    f"field '{f.name}': requires_evidence references unknown field '{ev}'"
                )
            if workflow.field(ev).type is not FieldType.EVIDENCE:
                raise WorkflowConfigError(
                    f"field '{f.name}': requires_evidence '{ev}' is not an evidence field"
                )
            ev_field = workflow.field(ev)
            # Single-level same-field/different-value mutual-exclusion check.
            # A transitive guard-chain version is a follow-up if multi-level gating appears.
            if (
                isinstance(f.show_when, Leaf)
                and isinstance(ev_field.show_when, Leaf)
                and f.show_when.op is CompareOp.EQ
                and ev_field.show_when.op is CompareOp.EQ
                and f.show_when.field == ev_field.show_when.field
                and f.show_when.value != ev_field.show_when.value
            ):
                raise WorkflowConfigError(
                    f"field '{f.name}': requires_evidence '{ev}' can never be co-active "
                    f"(guarded '{ev_field.show_when.field} = {ev_field.show_when.value}' "
                    f"vs '{f.show_when.field} = {f.show_when.value}')"
                )


# --- deterministic re-serialization (C3's render_raw delegates here) ----------


def dump_workflow(workflow: Workflow) -> str:
    """YAML re-serialization of the parsed model, declaration order preserved; round-trips
    through parse_workflow. Same Workflow -> byte-identical string."""
    data: dict[str, Any] = {
        "workflow": workflow.name,
        "profile": workflow.profile.value,
        "fields": [_dump_field(f) for f in workflow.fields],
    }
    if workflow.tools:
        data["tools"] = [_dump_tool(t) for t in workflow.tools]
    if workflow.escalation is not None:
        escalation = workflow.escalation
        data["escalation"] = {"when": escalation.when, "action": escalation.action}
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _dump_field(fld: Field) -> dict[str, Any]:
    out: dict[str, Any] = {"name": fld.name, "type": fld.type.value}
    if fld.required:
        out["required"] = True
    if fld.prompt is not None:
        out["prompt"] = fld.prompt
    if fld.description is not None:
        out["description"] = fld.description
    if fld.options:
        out["options"] = list(fld.options)
    if fld.show_when is not None:
        out["show_when"] = _dump_condition(fld.show_when)
    if fld.validators:
        out["validate"] = dict(fld.validators)
    if fld.satisfied_by:
        out["satisfied_by"] = list(fld.satisfied_by)
    if fld.requires_evidence:
        out["requires_evidence"] = list(fld.requires_evidence)
    return out


def _dump_condition(cond: Condition) -> object:
    match cond:
        case Leaf(field=fld, op=op, value=val):
            return f"{fld} {op.value} {val}"
        case AllOf(conditions=cs):
            return {"all_of": [_dump_condition(c) for c in cs]}
        case AnyOf(conditions=cs):
            return {"any_of": [_dump_condition(c) for c in cs]}
    return ""


def _dump_tool(tool: Tool) -> dict[str, Any]:
    out: dict[str, Any] = {"name": tool.name}
    if tool.description is not None:
        out["description"] = tool.description
    return out
