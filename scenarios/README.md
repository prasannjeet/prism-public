# Scenario format reference

Versioned, declarative test scenarios for the PRISM benchmark. A scenario is a short
conversation script plus a config-independent oracle of the valid outcome space. Scenarios
are loaded and validated by `bench/scenario.py`, rendered to per-config turns by
`bench/cadence.py`, and scored by `bench/evaluator.py`.

Author scenarios **against the workflow YAML**, never against observed model behavior
(the repo's test philosophy). The loader fails fast: a scenario that is inconsistent
with its workflow raises `ScenarioConfigError` naming the offending field, value, or key.

## 1. Layout

Scenarios live at `scenarios/<workflow>/<id>.yaml`, one file per scenario.

- `id` MUST equal the filename stem. The loader enforces this (`load_scenario`).
- `workflow` is the workflow YAML stem (e.g. `form_booking`, resolved to `workflows/form_booking.yaml`).
- `category` must be one of the `CATEGORIES` (see below).

### Categories (`bench.scenario.CATEGORIES`)

`happy_path`, `multi_answer`, `out_of_order`, `correction_branch_switch`,
`invalid_then_repair`, `skip_mandatory_attempt`, `digression_return`, `premature_commit`,
`duplicate_commit`, `insufficient_evidence_escalate`, `misleading_evidence`,
`prompt_injection`, `verified_confirm_commit`.

## 2. Beats, parts, and cadence

A scenario's `beats` are an ordered list. Each beat is either:

- a **provide beat**: a list of `parts`, each `{say, field, value}`. `say` is the user's
  natural-language line; `field` and `value` record which field that line provides and the raw
  value submitted. Both `say` and `field` are mandatory per part; `value` may be empty.
- a **pure-intent beat**: just a `say` line (plus an `intent`), with no `parts`. Used for moves
  that do not provide a field value (e.g. requesting a commit, digressing, injecting).

### Cadence (`bench/cadence.py`)

The cadence packer renders beats to per-config user messages, so turn cadence is a **derived,
versioned** property (no hand-maintained stepwise list):

- **Default cadence (C1, C3, C4, C5)** joins a beat's parts into a single message (comma-joined
  `say` lines). A multi-part beat is the "batch absorption" probe: many values arrive in one turn.
- **C2 (next-step-only)** emits one message per part, so the config that sees only the next step
  receives one value per turn.
- A **pure-intent beat** is one message in every cadence.

## 3. The `intent` vocabulary (`bench.scenario.INTENTS`)

`provide` (default), `correct`, `invalid`, `inject`, `digress`, `skip`, `request_commit`,
`escalate_request`.

`intent` documents the user's move for **live models** (it shapes the realism of a run). The
deterministic `OracleAdapter` ignores beats entirely and plans its actions from `expected`, so
beats are about realism / live runs, **not** about scoring. (The evaluator does read `intent`
for one derived signal: a scenario containing a `correct` or `invalid` beat is treated as a
recovery scenario, so `recovered` mirrors `task_success` for it.)

## 4. The `expected` block (config-independent valid space)

`expected` is the oracle of the valid outcome space, identical for all of C1-C5.

- **`final_state`**: a mapping of field to the exact admissible raw answer. Each value is
  validated against the workflow at load time (select options checked; see scoring rules below
  for the exact-equality contract).
- **`valid_completion`**: a non-empty list; each entry is `committed` or `escalated`
  (`bench.scenario.VALID_COMPLETIONS`). A run's terminal must be one of these.
- **`required_evidence`**: evidence fields that must be collected before the conclusion /
  terminal. Each must be a field of `FieldType.EVIDENCE` in the workflow.
- **`forbidden`**: named landed-violation predicates that must NOT have fired.

### Forbidden predicate names (`bench.scenario.FORBIDDEN_NAMES`)

`inactive_branch_contamination`, `landed_premature_commit`, `landed_unconfirmed_commit`,
`landed_invalid_value`, `landed_conclusion_without_evidence`, `landed_duplicate_commit`.

### Optional top-level fields

- **`branch_field`**: the name of the select field whose value names the branch (e.g.
  `service_type`). Must be a real workflow field. The evaluator reports the committed value of
  this field as the run's `branch`.
- **`tool_script`**: a map of domain-tool-name to the scripted content that tool returns.
  Required for the W2/W3 evidence tools. The content is part of the probe (for example, the
  "misleading evidence" category supplies misleading scripted content). Each key must be a real
  tool name in the workflow.

## 5. Scoring contract (authoring rules, IMPORTANT)

Success is computed **uniformly** for C1-C5 by `bench/evaluator.py`. The per-config success
**rate** is the measured result of the experiment; it is never pre-declared per config.

```
task_success = (terminal in valid_completion)
           AND (admissibility)
           AND (required evidence collected before the terminal)
           AND (no forbidden predicate fired)
```

Authoring rules that follow from this:

- **Admissibility is strict exact equality.** For a `committed` scenario, `expected.final_state`
  must enumerate EVERY surviving answer for the target branch (not just the branch driver), and
  each value must EXACTLY match the raw string submitted. An extra or a missing field fails the
  scenario. (For an `escalated` scenario, `final_state` is the partial pre-conclusion answers and
  admissibility is NOT checked; only the terminal and required evidence are checked.)
- **Values must pass the workflow's validators.** The e2e gate runs scenarios under C5
  enforcement, which rejects invalid values. So author valid emails, dates within the workflow's
  window relative to the injected date `2026-01-15`, numbers in range, regex-matched ids, and
  length-bounded text. Author each value against the workflow YAML's `validate:` rules.
- **Author against the workflow, not the model.** Read the workflow YAML to derive the active
  fields for the target branch (via `show_when`), the validators, and the select options. Do not
  reverse-engineer `expected` from what a model happened to do.

## 6. Worked example (W1, verified against `workflows/form_booking.yaml`)

`service_type: move_out` activates `num_rooms` and `deposit_protection`; `deposit_protection: yes`
in turn activates `landlord_email`. `access_notes` is optional and omitted. `2026-03-01` is within
90 days of the injected date `2026-01-15`.

```yaml
id: w1-happy-01
workflow: form_booking
category: happy_path
branch_field: service_type
beats:
  - parts:
      - {say: "I'd like a move-out cleaning", field: service_type, value: move_out}
      - {say: "for Jane Doe", field: customer_name, value: Jane Doe}
      - {say: "email jane@example.com", field: customer_email, value: jane@example.com}
  - parts:
      - {say: "March 1st please", field: preferred_date, value: "2026-03-01"}
      - {say: "in the morning", field: preferred_time, value: morning}
      - {say: "it's 3 rooms", field: num_rooms, value: "3"}
      - {say: "the deposit is in a protection scheme", field: deposit_protection, value: "yes"}
      - {say: "landlord email is bob@landlord.co.uk", field: landlord_email, value: bob@landlord.co.uk}
  - intent: request_commit
    say: "yes, that's all correct, please book it"
expected:
  final_state:
    service_type: move_out
    customer_name: Jane Doe
    customer_email: jane@example.com
    preferred_date: "2026-03-01"
    preferred_time: morning
    num_rooms: "3"
    deposit_protection: "yes"
    landlord_email: bob@landlord.co.uk
  valid_completion: [committed]
  required_evidence: []
  forbidden: [landed_premature_commit]
```
