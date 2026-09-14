# PRISM: Bounded Autonomy for LLM Agents via Workflow Projection and Deterministic Consequence Boundaries

Research artifact for the paper *PRISM: Bounded Autonomy for LLM Agents via Workflow
Projection and Deterministic Consequence Boundaries* (Prasannjeet Singh, School of
Informatics, University of Skövde), accepted at the [REALM
workshop](https://openreview.net/forum?id=ckKOi3zWcY) at EMNLP 2026. It contains the
complete runtime, the six
workflow definitions, the 88 benchmark scenarios, the evaluation harness, and the full frozen
data of the benchmark run reported in the paper (all 3,961 conversation transcripts plus the
aggregate result files), so every number in the paper can be verified from this repository alone.

PRISM gives an LLM workflow agent three things at once: **global process awareness** (a
three-tier workflow projection: a compact schema tier, a live state tier, and an on-demand
detail tool), **local decision autonomy** (the model chooses its own actions each turn; nothing
is hard-scripted), and **deterministic consequence boundaries** (every proposed action passes
through deterministic gates over an explicit state machine; with enforcement on, an action that
violates the workflow is rejected with a reason instead of mutating state). The paper evaluates
five configurations of one and the same engine (C1 to C5, below) across four commercial model
APIs to separate what *delivery* of process information contributes from what *enforcement*
contributes.

## The five configurations

All five share one engine, one tool surface, and one scripted conversation per scenario. A
configuration only changes what the model is shown and whether gates reject or merely record.

| Config | Workflow information shown to the model | Enforcement |
|---|---|---|
| C1 | None (prompt-only baseline) | Off |
| C2 | Only the current field, per turn | On |
| C3 | Raw workflow YAML plus live state, per turn | On |
| C4 | Full three-tier projection (schema tier, state tier, detail tool) | Off (monitor-only: violations are recorded and admitted) |
| C5 | Full three-tier projection | On (full PRISM) |

## Repository layout

| Path | Contents |
|---|---|
| `prism/` | The runtime: workflow schema loader, pure state engine, projection renderer, validators, gates, turn loop, tool dispatch, model adapters |
| `workflows/` | Six declarative workflow YAMLs (three small/mid, three deep) |
| `scenarios/` | 88 scripted test scenarios, grouped by workflow |
| `prompts/` | The frozen per-configuration system prompts (C1.txt to C5.txt) |
| `bench/` | Harness: single-cell driver, oracle adapter, batch grid runner, deterministic evaluator, aggregation, inferential stats, figure/table generators, `PREREGISTRATION.md` |
| `tests/` | The full pytest suite (engine, harness, scoring; no network needed) |
| `examples/run_oracle.py` | End-to-end offline demo (no API keys) |
| `results/prism-full-01/` | The frozen benchmark run: `manifest.json`, `aggregate/` CSVs, `figures/`, and all raw transcripts as tarballs under `transcripts/` |

## Install

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the checks

```bash
ruff check .
mypy prism bench
pytest
```

The test suite is fully offline and deterministic (tests that would hit a real model API are
marked `live` and excluded by default). It includes an oracle sweep that drives every one of the
88 scenarios to its reference completion, proving each reference outcome reachable by the engine.
(For the four escalate-expected scenarios the oracle drives the dispatcher directly; the two
workflows hosting them never advertise the escalate tool to agents, an affordance gap discussed
in the paper's Results.)

## Run a scenario end to end, offline

The oracle adapter plans the ideal action sequence from the workflow definition and the
scenario's reference answers, so the whole runtime-plus-scoring path runs with no API keys:

```bash
python examples/run_oracle.py
# or any scenario id, e.g.:
python examples/run_oracle.py t6-correction-branch-switch-deep
```

Expected output ends with `OK: oracle drove the scenario to a successful terminal state.`

## The frozen run (`results/prism-full-01`)

One benchmark run, frozen: 3,961 conversations = 88 scenarios x seeds x 17 model-configuration
combinations (233 cells per combination). **User turns are scripted**: every conversation
replays a fixed, versioned user script from `scenarios/`, so configurations and models see
identical inputs. Model turns are live API outputs from four commercial APIs:

- `claude-haiku-4-5` (Anthropic), all five configurations
- `gpt-5.4-mini` (OpenAI), all five configurations
- `gemini-3.1-flash-lite` (Google), all five configurations
- `claude-sonnet-4-6` (Anthropic), C1 and C5 only

`manifest.json` records the grid, the price table, and the budget. `aggregate/` holds the
scored per-conversation table and all derived statistics. `figures/` holds the four descriptive
figures. The raw per-conversation transcripts ship as one tarball per model under
`transcripts/` (about 8 MB compressed, about 150 MB extracted, 11,883 files).

### Unpack and verify the transcripts

```bash
for f in results/prism-full-01/transcripts/*.tgz; do tar xzf "$f" -C results/prism-full-01/; done
cd results/prism-full-01 && sha256sum -c transcripts/raw-manifest.sha256 && cd ../..
```

`raw-manifest.sha256` was produced when the run was frozen; all 11,883 files must check out.
Each conversation directory `<model>/<workflow>/<scenario>/<config>/seed-<n>/` contains:

- `events.jsonl`: the ordered event log (user messages, tool calls and results, gate
  rejections, admitted violations, evidence collection, terminal events). This is the ground
  truth everything else is computed from.
- `meta.json`: the cell key, completion status, attempt count, and the fixed injected date.
- `usage.json`: per-call token usage as reported by the provider, plus totals.

### Reproduce the aggregation, stats, figures, and tables

All of these are deterministic re-computations from the shipped data (no model calls, no keys).
Aggregation reads the extracted transcripts, so unpack them first (above).

```bash
# Re-score every transcript and rebuild aggregate/ (a few minutes; fixed bootstrap seed)
python -m bench.metrics results/prism-full-01
git diff --stat results/prism-full-01/aggregate/   # expect: no changes

# Rebuild the four figures in results/prism-full-01/figures/
# (the CSVs above are byte-identical on re-run; PNG bytes may differ across
# matplotlib versions, the content is the same)
python -m bench.plots results/prism-full-01

# Emit the paper's LaTeX tables (written to paper/sections/tables/, gitignored here)
python -m bench.paper_tables results/prism-full-01

# Print the failure-taxonomy counts used in the paper
python -m bench.failure_taxonomy results/prism-full-01
```

Re-running the full grid live is also possible (`python -m bench.runner <run_id>`) but needs
provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) and spends real
money; it is not required to verify the paper.

## Data dictionary (`results/prism-full-01/aggregate/`)

### `results.csv` (and `results.json`, the same records as JSON)

One row per conversation (cell). Columns:

| Column | Meaning |
|---|---|
| `model`, `workflow`, `scenario`, `config`, `seed` | The cell key |
| `task_success` | Strict success: reached a terminal state the scenario accepts, the committed answers match the reference branch exactly (structured fields exact, free-text fields present and non-empty), all required evidence was collected before the terminal event, and no forbidden predicate fired |
| `task_success_refined` | Forgiving variant: extra committed fields are forgiven iff they are defined, optional, and active for the committed answers. Always a superset of strict |
| `terminal` | `committed`, `escalated`, or `none` (ran out of turns with no terminal action) |
| `branch` | Value of the scenario's branch-discriminating field in the committed answers (empty if not committed or not applicable) |
| `contamination` | Count of admitted inactive-branch writes (`violation_admitted` events with reason `inactive_field`). Non-zero only in enforcement-off configurations; the engine prunes the write within the same transition, so the event, not surviving state, is the record |
| `turns` | User turns consumed (scripted) |
| `tool_calls`, `tool_errors` | Tool invocations and structurally failed invocations |
| `recovered` | True iff the scenario contains a recovery beat (correction, digression, contradiction) and the run still succeeded strictly |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `total_tokens` | Provider-reported token usage summed over the conversation |
| `cost_usd` | Token usage priced with `manifest.json`'s price table |
| `attempted_<reason>` | Number of proposed actions whose gate verdict was `<reason>` (in enforcing configurations these were rejected with feedback; in monitor-only configurations they were recorded and admitted) |
| `admitted_<reason>` | Number of those proposals that were admitted despite the negative verdict (enforcement off). Zero everywhere for C5 by construction |

Violation reasons: `inactive_field`, `not_complete`, `invalid_value`, `not_confirmed`,
`evidence_missing`, `wrong_kind` are the six admissible kinds the paper's V covers;
`unknown_field` and `already_submitted` are structural impossibilities that reject in every
configuration (no `admitted_` counterpart can be non-zero for a structural reject).

### The derived statistics files

| File | One row per | Columns |
|---|---|---|
| `passk.csv` | (model, config, k) | `passk` is the combinatorial pass^k estimator computed per scenario then averaged, never pooled; `n_scenarios` is the denominator. k=3 covers only the 57 scenarios that ran three seeds |
| `passk_refined.csv` | (model, config, k) | Same estimator under refined scoring |
| `wilson_ci.csv` | (model, config, scoring) | `n` cells, `successes`, `rate`, and the Wilson 95% interval `ci_low`/`ci_high`. Cell-level (ignores seed-within-scenario correlation) |
| `bootstrap_deltas.csv` | (model, baseline, scoring) | `delta` is the C5-minus-baseline success-rate difference under a paired scenario bootstrap (resamples scenarios, not runs); percentile 95% CI and two-sided p-value; `n_resamples` = 10,000, fixed seed |
| `mcnemar.csv` | (model, baseline, scoring) | Exact McNemar over the per-scenario all-seeds-pass collapse: `b` = scenarios where C5 passes and the baseline fails, `c` = the reverse, `p_value` exact binomial |
| `scoring_delta.csv` | (model, config) | Strict vs refined pass counts and rates; `n_rescued` = cells that pass refined but not strict |

`results/turnbudget-01/aggregate/` holds the same eight aggregate files for the
single-model (gemini-3.1-flash-lite) continuation-budget sensitivity re-run
quoted in the paper's Results (47.8% to 48.1%); its raw transcripts are not
shipped, only the aggregates behind those two numbers.

## Citing this work

- Paper and reviews: https://openreview.net/forum?id=ckKOi3zWcY
- Interactive results explorer, pre-generated from the frozen run: https://prism-realm.pages.dev

```bibtex
@inproceedings{singh2026prism,
  title     = {{PRISM}: Bounded Autonomy for {LLM} Agents via Workflow Projection
               and Deterministic Consequence Boundaries},
  author    = {Singh, Prasannjeet},
  booktitle = {Proceedings of the {REALM} Workshop at {EMNLP} 2026},
  year      = {2026},
  address   = {Budapest, Hungary},
  url       = {https://openreview.net/forum?id=ckKOi3zWcY}
}
```

## License

MIT (see `LICENSE`).
