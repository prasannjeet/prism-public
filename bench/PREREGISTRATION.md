# PRISM benchmark — pre-registration

**Status:** committed and git-tagged **before** any full/billed experiment run and **before** the prompt/workflow/scenario freeze. This document fixes, in advance, the single primary hypothesis and the rule by which we will judge it, so that our headline confirmatory result cannot be a comparison selected after seeing the data. Secondary and exploratory analyses are listed here too, clearly separated from the primary. Any deviation from this document is disclosed in the paper with its rationale (§8).

**Date written:** 2026-06-16 (Phase 2.3, before the pilot and the `v0.2-frozen` freeze).
**Governs:** the full experiment run only. The pilot (a tiny pre-freeze slice read by hand to fix prompt/representation bugs) is *not* governed by this document.
**Form:** an in-repo, version-controlled, timestamped pre-registration (this file + the git tag at freeze). For a four-page workshop short paper this is the practical instrument; the timestamped commit is the binding record.

---

## 1. Study design (frozen at `v0.2-frozen`)

We compare five configurations of one agent over a single, unchanged deterministic engine (the fairness invariant — they differ only in projection tiers, exposed tools, and one enforcement boolean; see `AGENTS.md` §4):

- **C1 — prompt-only:** rules described in the system prompt; no live structured state.
- **C2 — next-step-only:** shown only the current field each turn.
- **C3 — full-context:** the entire state re-rendered every turn.
- **C4 — PRISM without enforcement:** full PRISM projection, but the consequence gate only logs (admits) violations.
- **C5 — PRISM:** global process awareness + local decision autonomy + enforced consequence boundaries.

| Dimension | Values |
|---|---|
| Workflows | `form_booking` (W1), `incident_investigation` (W2), `support_diagnosis` (W3) |
| Scenarios | 43 (W1 15, W2 14, W3 14), authored against the YAML, not observed behavior |
| Models — full grid (C1–C5) | `claude-haiku-4-5`, `gpt-5.4-mini`, `gemini-3.1-flash-lite` (the three "cheap" models) |
| Models — subset (C1 vs C5 only) | `claude-sonnet-4-6` (one mid-tier model) |
| Seeds (repeated runs) | 5 (8 if budget allows for C5 + C1); repeated runs, not a seed parameter, are the reliability axis |
| Injected date | fixed `2026-01-15` (relative-date resolution is deterministic) |
| Users | scripted (deterministic, reproducible), not an LLM user-simulator |
| Budget | $300 hard ceiling / $150 target (locked decision D6) |

Exact model IDs, per-model sampling settings, git tag, access dates, and the price table are recorded in the run manifest (`results/<run_id>/manifest.json`) and the paper.

---

## 2. Primary hypothesis (H1) — the single confirmatory claim

> **H1.** For each cheap model, full PRISM (**C5**) achieves higher **pass^5** than prompt-only (**C1**).

- **Primary metric — pass^5:** the fraction of scenarios for which the agent succeeds in all of k=5 repeated runs, via the combinatorial (unbiased) estimator, computed **per scenario then averaged over scenarios** (never pooled). Definition and rationale: internal planning notes (not part of this artifact). ("Task success" = a conversation reaches an admissible terminal state with all required evidence collected first and no forbidden landed-violation; scored deterministically by `bench/evaluator.py`.)
- **Admissibility is outcome-based (amended 2026-06-17, see note below).** For a *committed* terminal, the committed answers are admissible iff: (a) **structured fields** (types SELECT, NUMBER, EMAIL, DATE, DATETIME) exact-string-match the reference value; (b) **free-text fields** (types TEXT, CONCLUSION) are merely **present and non-empty** — their content is **not** auto-checked (we deliberately use **no LLM judge** in the core metric); and (c) the set of committed field names equals the reference branch's field set exactly (no missing, no extra). For an *escalated* terminal, admissibility is unchanged (no answer comparison). Conclusion *correctness* (whether a free-text root cause is actually right) is a qualitative/secondary read, not part of the confirmatory metric.

> **Amended 2026-06-17 (pilot finding: exact-matching free text is unworkable).** The original rule required the entire committed answer map to exact-match the reference, including free-text conclusion/summary fields. A pre-freeze pilot showed strong configurations being scored as failures despite producing correct outcomes, purely because no model reproduces a free-text root cause / resolution / complaint summary word-for-word. We therefore switched to the outcome-based admissibility above (structured exact, free-text presence-only, no content auto-check), consistent with tau-bench-style outcome scoring and our deliberate exclusion of an LLM judge from the core metric. Rationale: internal planning notes. This amendment predates the `v0.2-frozen` freeze.
- **Comparison:** C5 vs C1, computed **separately per cheap model** (the three full-grid models). We never pool across models (different tokenizers and cost bases make pooled numbers meaningless).
- **Test:** a paired bootstrap that resamples **scenarios** (the unit of independence — repeated runs within a scenario are correlated), yielding a two-sided **95% confidence interval** for the pass^5 gap (C5 − C1). McNemar's test on scenario-level pass/fail is reported as a cross-check. Effect sizes (the gap magnitude) are always reported, not only significance.
- **Decision rule:** H1 is **supported** if the 95% CI for the pass^5 gap lies entirely above zero (C5 higher) for **at least 2 of the 3 cheap models**. The full per-model pattern (all three) is reported regardless of the verdict.

**Why C1 is the primary baseline (the deliberate "safe anchor").** Beating prompt-only is the most robust and most widely relatable comparison (most deployed agents are prompt-only). We stake the *confirmatory* claim on it precisely because it is solid; the harder, more novel comparisons (especially C5 vs C2, below) are fully analyzed and featured in the discussion as **secondary** results, so no scientific content is lost. Rationale captured for the paper in internal planning notes.

H1 is a **single** comparison, so no multiple-comparison correction applies to the primary.

---

## 3. Secondary hypotheses (pre-registered, not primary)

Reported with the same per-model, no-pooling discipline. These are pre-registered so they are confirmatory-grade, but they are not the headline; where several are tested together we report CIs and treat them as a family (Holm correction within the family).

- **S1 — single-run reliability:** C5 > C1 on single-run task-success rate (this is pass^1), with Wilson 95% CIs.
- **S2 — the awareness claim (featured in the discussion):** C5 > C2 on pass^5. Tests whether *global process awareness* beats a compliant-but-myopic next-step-only view. C2 is a strong baseline, so this is the substantive intellectual contrast.
- **S3 — enforcement is load-bearing (the T2 contrast):** C5 admits **zero** admissible violations whereas C4 admits a positive number. Measured on the attempted-vs-admitted violation counts read from the event logs (the signature T2 table), **not** on pass^5 (an admitted violation does not always fail the task, so the reliability metric is the wrong instrument for this claim).
- **S4 — token economics:** C5 is comparably reliable to C3 at a fraction of the token cost. Measured on tokens and dollars per conversation (the "C5 cheap vs C3 expensive" asymmetry), **not** as a "C5 more reliable than C3" test — C3 has all information re-rendered each turn and may match or exceed C5 on reliability; the claim against C3 is cost, not reliability.

---

## 4. Exploratory analyses (explicitly NOT confirmatory)

Hypothesis-generating; reported honestly as exploration, never framed as predictions:

- **Capability gradient:** does PRISM help weaker models more? (the three cheap models vs the mid-tier `claude-sonnet-4-6` on the C1-vs-C5 subset).
- **Per-category flexibility probes:** success rate by scenario category (where next-step-only is expected to fail and prompt-only to violate).
- **Efficiency:** tool-call counts / unnecessary calls / tool errors; turns to completion.
- **Recovery rate:** of scenarios with injected corrections or invalid values, the fraction eventually succeeding.
- **(W2/W3) Evidence & escalation:** evidence coverage before conclusion; escalation correctness.
- Anything surfaced post hoc.

---

## 5. Analysis rules (fixed in advance)

- **Per model, never pooled across models.**
- **pass^k:** combinatorial estimator, per scenario then averaged, never pooled.
- **Rates:** Wilson 95% confidence intervals.
- **Pairwise differences:** paired bootstrap resampling **scenarios** (not runs); McNemar on scenario-level pass/fail as a cross-check.
- **Re-run policy:** a cell is re-run **only** on an infrastructure failure (HTTP 5xx / timeout / rate-limit exhaustion), and every re-run is logged with its reason. Model refusals or odd-but-valid outputs are **data**, never re-run.
- **Partial / missing data:** if a budget stop leaves a scenario with fewer than k completed seeds, pass^k is reported only up to that scenario's completed seed count; any scenario with an incomplete seed set is logged. No silent truncation.
- **The statistics layer itself (Wilson CIs, the bootstrap, McNemar) is implemented in Phase 3**, over the frozen master table produced by `bench/metrics.py`. This document commits to *which* analyses we will run; it does not run them.

---

## 6. What we report regardless of outcome

The paper claims a **map of the trade-off**, not "PRISM wins everything." Negative or mixed results are published. If a comparison comes out against us, the honest pivots already written into the internal evaluation plan apply: e.g. if the awareness contrast (S2) is flat, the paper leans on enforcement (S3) and token economics (S4); if enforcement is rarely exercised, the adversarial subset carries that claim.

---

## 7. Hypotheses written down so we cannot fool ourselves

The full directional expectations for every comparison (and the "if it comes out otherwise…" response for each) are in the internal evaluation plan (not part of this artifact). They are recorded there before the run as a companion to this pre-registration.

---

## 8. Deviations policy

Any departure from this pre-registration (a changed metric, an added/removed comparison, a different decision rule, a re-run not covered by §5) is **disclosed in the paper** with its rationale. The binding record is this file at the `v0.2-frozen` git tag.
