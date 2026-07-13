# Example workflows

Workflows, one engine — the abstraction demonstration. Each is a generic-schema YAML the
same `StateEngine` + projection + gates drive with no per-domain code.

| File | Kind | Profile | Status |
|---|---|---|---|
| `form_booking.yaml` | W1 — branching form | verified | built |
| `incident_investigation.yaml` | W2 — investigation (evidence + conclusion) | express | built |
| `support_diagnosis.yaml` | W3 — hybrid form/investigation | verified | built |
| `travel_insurance_claim.yaml` | T4 — deep multi-branch claim adjudication | verified | built |
| `incident_response.yaml` | T5 — deep multi-branch incident response (compound conditions) | verified | built |
| `clinical_trial_eligibility.yaml` | T6 — Phase-II oncology eligibility screen (the showpiece) | verified | built |

## W1 — `form_booking` (home cleaning-service booking)

The **form special case**: every node is an answer-field, branching by `service_type`. Verified
profile — the agent must `confirm` the collected data with the user before `commit`.

```
service_type ─┬─ move_out   → num_rooms, deposit_protection ─→ landlord_email   (2-level nest)
              ├─ deep_clean  → home_size_sqm, appliances_inside, [pet_on_site]
              └─ windows     → num_windows, floors_above_ground
shared (always): customer_name, customer_email, preferred_date, preferred_time, [access_notes]
```

`[field]` = optional. W1 demonstrates **value-conditioned activation** (the mandatory set differs
per branch), a **2-level cascade** (switching `service_type` prunes the whole `move_out` subtree
including the nested `landlord_email` in one transition), and validator coverage: email (×2),
`future_within_days` date, range (×4).

## W2 — `incident_investigation` (production incident triage)

The **investigation type**: alongside answer-fields it uses `evidence` nodes (satisfied by tool
calls, not user answers) and `conclusion` nodes (gated by `requires_evidence`), branching by
`symptom`. Express profile — commit as soon as the engine reports complete, with no confirm step.

```
affected_service (shared) · symptom ─┬─ server_down  → outage_time, deploy_check(ev), log_check(ev)
                                     │                 → server_root_cause [needs: deploy_check, log_check]
                                     ├─ slow_response → latency_window, metrics_check(ev)
                                     │                 → latency_root_cause [needs: metrics_check]
                                     └─ error_spike   → error_sample   (no conclusion → escalate path)
```

`(ev)` = evidence node, collected by a tool (`check_deployments` / `check_logs` / `check_metrics`),
never answered directly. W2 demonstrates **evidence-via-tools**, the **conclusion-requires-evidence
consequence boundary** (the C4-vs-C5 signature: under enforcement a premature conclusion is blocked;
without it the attempt is admitted and logged) shown in multi-evidence (`server_root_cause`, needs 2)
and single-evidence (`latency_root_cause`, needs 1) form, the **express commit profile** (vs W1's
verified), and **evidence cascade** (switching `symptom` prunes already-collected evidence). The
`error_spike` branch has no conclusion — the structural signal of a path that does not self-resolve.
`escalation:` is **declarative metadata only**.


## W3 — `support_diagnosis` (customer-support triage)

The **hybrid**: a shared, validated form front-section (W1 heritage) feeds a `category` branch
driver, each branch carrying diagnostic `evidence` nodes + an evidence-gated `conclusion`
(W2 heritage), ending in a `verified` confirm → ticket commit. One YAML, the unchanged engine —
the abstraction demonstration that the same machinery spans the space between W1 and W2.

```
shared (always, validated): account_id [pattern], customer_email [email],
                            plan_tier [select], issue_summary [length], [callback_number]
category ─┬─ billing      → invoice_number, billing_lookup(ev), payment_check(ev)
          │                 → billing_resolution      [needs: billing_lookup, payment_check]
          └─ connectivity → outage_started [not_future], service_status(ev), line_test(ev)
                            → connectivity_resolution [needs: service_status, line_test]
```

`[field]` = optional. `(ev)` = evidence node, collected by a tool (`check_billing_history` /
`check_payment_status` / `check_service_status` / `run_line_test`), never answered directly. W3 is
the workflow that exercises **every consequence boundary on one config** — inactive-field answer,
invalid value, evidence-faking (`wrong_kind`), conclusion-before-evidence, and premature/unconfirmed
commit (the `verified` boundary). Its form section completes the **validator registry**: W3 adds the
regex `pattern` and `length` validators that W1/W2 never used (W1∪W2∪W3 cover all five). And it shows
the **heterogeneous cascade** — switching `category` prunes the previous branch's answers *and*
already-collected evidence in one transition (W1 showed answer-cascade, W2 evidence-cascade; W3
unifies both). The `commit` always files a support ticket (no conditional-commit machinery; "optional"
refers to the optional `callback_number`).


## T4 — `travel_insurance_claim` (travel-insurance claim adjudication)

The **deep multi-branch** workflow (27 fields): a shared validated head feeds a `claim_type`
primary driver into three covered-reason sub-trees, each nesting **four levels deep** (baggage has
two depth-4 paths), ending in a per-branch evidence-gated adjudication and a `verified` payout
commit. The first workflow on the compound-condition engine: it uses an AND, an `in`-membership
inside an AND, and one numeric `>` threshold leaf, where the domain genuinely forks on two
conditions. One YAML, the unchanged-except-for-guards engine.

```
head (always, validated): policy_number [length], policy_purchase_date [not_future],
                          claimant_email [email], claimed_amount [range]
claim_type ─┬─ medical      → physician_statement(ev)
            │                 medical_subtype ─┬─ expense → medical_care_setting
            │                                  │            └─ inpatient → admit_discharge_dates [L4]
            │                                  └─ evacuation → evac_necessity_cert(ev)
            │                 → medical_adjudication        [needs: physician_statement]
            ├─ cancellation → proof_of_payment(ev)
            │                 cancellation_cause ─┬─ medical_reason → medical_outcome ─┬─ death → death_certificate(ev) [L4, all(cause=medical_reason, outcome=death)]
            │                                     │                                    └─ in[sickness,injury] → cancel_physician_statement(ev) [L4]
            │                                     └─ non_medical → non_medical_reason, named_reason_document(ev)
            │                 → cancellation_adjudication   [needs: proof_of_payment]
            └─ baggage      → baggage_proof_of_ownership(ev)
                              baggage_subtype ─┬─ stolen → police_report_filed ─ "yes" → police_report(ev) [L4]
                                               └─ damaged → damage_severity ─ repairable → repair_estimate(ev) [L4]
                              → baggage_adjudication        [needs: baggage_proof_of_ownership]
tail (numeric guard): claimed_amount > 10000 → extra_documentation(ev)
```

`[field]` = optional / `(ev)` = evidence node (tool-satisfied, never answered directly). T4
demonstrates **depth-4 nesting in all three branches at once** on the unchanged transition/prune
core, the **compound-condition guard language** (AND, `in`-membership, numeric `>`), and the full
consequence-boundary suite (inactive-field answer, invalid value, evidence-faking, conclusion-
before-evidence, unconfirmed commit) co-resident on a `verified` payout decision. The shared
`upload_physician_statement` tool satisfies two never-co-active evidence fields (legal tool sharing).


## T5 — `incident_response` (deep multi-branch security-incident response)

A deliberately **deeper, multi-driver sibling of W2**. One incident is driven from intake to
disposition through four type-specific deep subtrees, each a genuine 4-5 level stacked chain, on the
**compound-condition** engine extension (AND / OR / numeric-threshold `show_when`). Same node kinds
(answer / evidence / conclusion), same engine, only deeper. Verified profile: containment,
eradication, and notification are irreversible, so the agent must `confirm` before `commit`.

```
reporter_channel · alert_summary · triage_validation(ev)
  alert_disposition ─┬─ false_positive → (close, fields 5-42 inactive)
                     └─ confirmed_incident
                          functional_impact · information_impact · affected_host_count · severity · asset_class
                          mass_incident_bridge_note [opt, show_when affected_host_count >= 50]   ← NUMERIC GUARD
                          incident_type ─┬─ phishing_bec
                          (MASTER driver) │   is_malicious_email=yes → spoof_class, recipient_scope(ev), anyone_clicked
                                          │     anyone_clicked=yes → payload_exec(ev), entered_credentials
                                          │       entered_credentials=yes → bec_persistence(ev)
                                          │         → phish_takeover_confirmed [needs: bec_persistence, payload_exec]
                                          ├─ ransomware  (genuine 4-level stacked chain)
                                          │   variant_ioc(ev) · ransom_scope(ev) · encryption_scope
                                          │     encryption_scope in {partial, full} → recovery_path
                                          │       recovery_path=restore_from_backup → data_impact(ev), backup_freshness
                                          │         backup_freshness in {within_24h, within_7d} → backup_integrity(ev)
                                          │           → ransom_recovery_decision [needs: backup_integrity, data_impact, ransom_scope]
                                          │       recovery_path in {rebuild, negotiate} OR stale backup → escalate_external_dfir  ← ESCALATE-BY-DESIGN
                                          ├─ compromised_credentials
                                          │   compromise_confirm(ev) · principal_type · attacker_activity(ev) · persistence_established
                                          │     persistence_established=yes → backdoor_enum(ev)
                                          │       → cred_eradication_plan [needs: backdoor_enum, attacker_activity, compromise_confirm]
                                          └─ data_exfiltration  (also via information_impact=confirmed_loss, an OR-entry)
                                              exfil_confirmed=yes → exfil_egress(ev), exfil_data_class
                                                exfil_data_class=regulated → breach_notification_decision [needs: exfil_egress]
                          containment_action · disposition · incident_report · lessons_learned [opt]
```

`[field]` = optional. `(ev)` = evidence node (tool-satisfied, never answered). T5 demonstrates
**deep multi-branch navigation** (four independently deep subtrees, not one spine with shallow
siblings), the extension's **real numeric guard** (`affected_host_count >= 50` as an actual
visibility leaf), the **nested AND-of-OR** combinators (the ransomware fresh-backup gate and the
exfiltration OR-entry that converges two paths into one subtree), three **multi-evidence
conclusions** (2/3/3) plus one **single-evidence** by design, and the deliberate **ransomware
escalate-by-design** path (the recovery conclusion is unreachable on `rebuild` / `negotiate` /
stale-backup, so the only legal disposition is `escalate_external_dfir`). This is the C4-vs-C5
measurement on a workflow where an admitted wrong action (a premature recovery decision, an
unconfirmed irreversible commit) is genuinely consequential.


## T6 — `clinical_trial_eligibility` (Phase-II oncology eligibility screen) — the showpiece

The **deepest, largest** workflow: 91 fields, six branches nesting 4-5 levels (deepest chain 6
including the `cancer_type` driver), an `enroll` conclusion behind a 7-leaf conjunctive `all_of`
gated on five evidence packets, per-branch disease screen-fails, plus safety / rescreen / refer /
escalate dispositions (nine conclusions), `verified` profile. Built on the compound-condition
engine extension (`all_of` / `any_of` / numeric leaf guards) and the unchanged node kinds. Its job:
push global process awareness, local autonomy, and deterministic consequence boundaries to the
breaking point of the weak configurations while C5 holds.

```
consent_signed = yes  (hard gate)
  shared head: screen_date [not_future], planned_c1d1 [<=90d], patient_age, ecog_status, life_expectancy_ge_12wk
  all_of[consent_signed=yes, ecog_status<=2] → cancer_type  (DRIVER)              ← AND-of-(=)-and-NUMERIC
    A. disease (per-type spines)
       lymphoma  → subtype → mcl_cyclin_d1 → mcl_tp53 → mcl_high_risk_eligible    [depth 5]
       leukemia  → lineage → leuk_marrow_blasts → (>=20) leuk_cytogenetic_risk → leuk_adverse_cohort_open   [depth 5, NUMERIC]
       solid     → histology · stage · target_lesion_count → (stage=iv) solid_lesion_prior_rt → solid_new_lesion_confirmed   [depth 4]
       dx_confirmed_ev(ev) · measurable_disease_present → measurable_disease_ev(ev)
    B. prior therapy & washout  (DEEPEST)
       prior_therapy_any=yes → prior_class=radiation → cns_directed_prior=yes → cns_stable_no_disease=yes → cns_sponsor_approval   [depth 6 incl. driver]
       prior_records_ev(ev) · washout_elapsed_by_c1d1 · residual_tox_grade · (targeted/cellular sub-chains)
    C. marrow & organ  [depth 4]      marrow_adequate=no → cytopenia_cause → marrow_tumor_pct → (>=30) marrow_pi_discussion
       cbc_ev(ev) · chem_lft_ev(ev) · coag_ev(ev)[opt] · crcl/hepatic/bili/transaminase panel
    D. cardiac  [depth 3]             qtcf_ms → (>470) qtc_confirmed_2of3 → qtc_correctable;  any_of[cardiac_history in {arrhythmia, mi_acs}] → cardiac_cleared   ← OR-CONVERGING
       ecg_ev(ev) · echo_ev(ev) · lvef_pct · cardiac_history
    E. serology reflex  [HBV depth 4] hbsag=negative → anti_hbc=positive → hbv_dna_detectable=no → hbv_prophylaxis;  hiv/hcv reflexes
       serology_ev(ev)
    F. conmed / comorbid / reproductive  [depth 4]   prohibited_conmed=yes → conmed_switchable → conmed_switch_confirmed;  childbearing → pregnancy_ev(ev) → pregnancy_negative → contraception_method
       conmed_ev(ev) · steroid_mg_pred_eq · second_malignancy

conclusions:
  enroll                       [needs: dx_confirmed_ev, measurable_disease_ev, cbc_ev, chem_lft_ev, serology_ev]
    all_of[measurable=yes, marrow_adequate=yes, hiv=neg, hbsag=neg, hcv_ab=neg, cardiac=none, prohibited_conmed=no]   ← 7-LEAF GATE
  screen_fail_disease_lymphoma / _leukemia / _solid / _no_measurable   [needs: dx_confirmed_ev]
  screen_fail_safety           [needs: cbc_ev, serology_ev, ecg_ev, echo_ev]   ← QTc routes here, NOT to enroll
  rescreen[opt] · refer_waiver[opt] · escalate_cohort_slot[opt]
```

`[opt]` = optional (not in `remaining_mandatory`). `(ev)` = evidence node (tool-satisfied, never
answered directly). T6 exercises the full compound-condition language: the `cancer_type` AND-gate
(an `=` leaf AND a numeric leaf), numeric visibility leaves (`ecog_status<=2`, `qtcf_ms>470`,
`leuk_marrow_blasts>=20`, `marrow_tumor_pct>=30`, `target_lesion_count<1`), the OR-converging
cardiac-clearance path, and the recurring `any_of[cancer_type in {lymphoma, leukemia, solid_tumor}]`
("active for any chosen cancer type") on every cross-type field. QTc is modeled as a screen-fail,
not an enroll requirement, so confirmed prolonged QTc routes to `screen_fail_safety` while the
normal-QTc enroll happy path stays reachable (the prior happy-path bug, now pinned by a regression
test).
