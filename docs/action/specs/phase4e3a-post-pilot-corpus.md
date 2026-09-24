---
title: Phase 4E.3a post-pilot corpus completion
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4e3a-post-pilot-corpus.md
globalRef: qmd://saracura/docs/action/specs/phase4e3a-post-pilot-corpus.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
  - benchmarks/manifests/phase4e-protocol-pilot-recovery.v12.json
related:
  - benchmarks/saracura_universal_corpus.py
  - benchmarks/saracura_universal_training.py
  - benchmarks/phase4e_pipeline.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.3a post-pilot corpus completion

## Decision and task identity

`task_id=saracura-phase4e3a-post-pilot-corpus-v1`.

Use the model pair and observable acceptance contract that passed the complete
v12 protocol pilot to generate one immutable 1,600-task synthetic corpus. Keep
the existing corpus allocation and quality minimums, remove obsolete financial
authorization gates, report actual provider spend at every run-local USD 10
crossing, and seal a training-compatible packet only if all identity, privacy,
lineage, split and minimum gates pass.

This increment ends at a verified packet and corpus report. Embedding
extraction, training, holdout release, runtime registration and public quality
claims remain Phase 4E.3b and cannot begin unless this packet passes.

## Verified context

- The v12 report at `.artifacts/phase4e/pilot-report-v12/report.json` resolved
  all 140 tasks, accepted 134, accepted 64/70 English and 70/70 PT-BR rows,
  accepted 64/70 complete pairs, and returned `PASS` with zero operational or
  privacy failure. `docs/action/phase4e-execution.md` records its immutable
  hashes and USD 0.5680124 provider cost.
- The reviewed recovery manifest pins GPT-4.1 as author, GPT-4.1 Mini as
  reviewer, Azure-only routing, ZDR, data-collection denial, no fallback,
  strict parameter support and a 120-second timeout
  (`benchmarks/manifests/phase4e-protocol-pilot-recovery.v12.json`).
- The corpus policy is still historical v1: it binds Qwen/Llama, a USD 17 hard
  authorization and false generation/training flags
  (`benchmarks/saracura_universal_policy.py:13-209` and
  `benchmarks/manifests/phase4e-saracura-universal-policy.v1.json`).
- The current planner fixes 1,600 identities, 1,120/240/240 split allocation,
  300 cross-locale pairs and balanced locale/cardinality cells, but selects a
  scenario from the global code list instead of the pilot's domain-compatible
  map (`benchmarks/saracura_universal_corpus.py:353-510`).
- `acceptance_reason` already supports answer/quality acceptance with semantic
  mismatch disabled, but the default corpus resolver and packet validator
  still require exact reviewer attestation and pair agreement
  (`benchmarks/saracura_universal_corpus.py:1706-1905` and `2016-2217`).
- The v12 pilot has the proven relaxed reviewer observation schema,
  observable genericity mapping, conservative transport rejection and exact
  uncertain-lineage validation (`benchmarks/saracura_universal_pilot.py:101-118,
  650-704, 1379-1644, 1647-1880`). These mechanisms are currently pilot-owned.
- The corpus runner remains sequential and aborts on transport uncertainty or
  settled response-validation failure instead of closing only the affected
  identities (`benchmarks/phase4e_pipeline.py:690-874`).
- The existing training lane consumes a sealed packet, performs deterministic
  CPU training twice, freezes the checkpoint before one-time holdout release,
  and can emit `pre_holdout_failed`, `holdout_failed` or `passed`
  (`benchmarks/saracura_universal_training.py:2040-2230`). This increment must
  preserve that boundary and make the new packet schema consumable without
  opening holdout content.

## Scope

1. Add create-only `training-data-source-policies.v3.json` and
   `phase4e-saracura-universal-policy.v2.json`. V1 and v2 registries and policy
   v1 remain byte-identical historical inputs.
2. Bind the Phase 4E source-policy exception in registry v3 to
   `openai/gpt-4.1` author and `openai/gpt-4.1-mini` reviewer. Keep the exception
   synthetic-only: canonical training, calibration, publication, quality
   claims and automation remain false. The execution policy permits only
   synthetic generation in this increment; synthetic research training,
   real-checkpoint and runtime flags remain false until the packet passes and
   Phase 4E.3b reviews the training transition.
3. Preserve the 1,600-task seed, split counts, locale/cardinality allocations,
   300 planned pairs, minimums and Wilson stop. Change only the scenario choice
   to the exact manifest-owned domain-to-scenario map proven by the v12 pilot,
   using deterministic selection within each domain list.
4. Use the exact v12 request envelope for corpus calls: GPT-4.1 author,
   GPT-4.1 Mini reviewer, Azure-only `order`, no `quantizations`, ZDR,
   data-collection denial, no fallback, strict parameter support, 120-second
   transport timeout, author output capacity 1,024 tokens for both singleton
   and pair calls, reviewer output capacity 512, reviewer temperature zero and
   no reasoning parameter. The author uses temperature zero and omits reasoning;
   `author_request` and `reviewer_request` are both plan-bound. Use the pilot reviewer view, which excludes
   `family_id` and all answer/pair/planner fields. Author and reviewer remain
   distinct roles. Keep answer blindness, observable
   genericity, privacy, natural-language, exclusivity, capacity, leakage,
   duplicate and author-target gates.
   The author request also omits reasoning entirely and uses the exact Azure
   provider override. Its system prompt remains byte-identical to v12 with
   SHA-256 `439eb0b348c3c179a7f1c6618a466b05fc92fab24d3638652be98fb8dc473c6c`.
   The reviewer calls the byte-identical `_pilot_reviewer_system()` prompt with
   SHA-256 `f5e2f107e4b232099e364156c9dd8c596ab6bbeb4dd40bd27166806bc39627dc`,
   exposes `pilot_reviewer_schema(1)` on the wire and maps it through
   `bind_pilot_reviewer_record`. The old corpus `fictional` prompt/schema is not
   reachable from a v3 plan.
5. Reviewer scenario and criterion-role output become diagnostic-only for the
   corpus, exactly as measured in the pilot. A row is accepted by blind answer
   agreement plus the existing quality gates. A complete cross-locale pair
   requires both members to pass and their author targets to remain identical;
   reviewer diagnostic attestations need not match either each other or the
   author target. One accepted pair member survives when its sibling fails an
   independent reviewer or quality gate; v3 does not apply
   `paired_member_rejected`. Only author `semantic_target_mismatch` rejects both
   siblings.
6. Generalize the proven pilot reviewer/diagnostic types into shared corpus
   primitives. Historical strict corpus-v1 validation stays available for old
   artifacts; the new plan and packet schemas select the post-pilot behavior
   explicitly rather than changing old schema semantics in place.
7. Add a report-only corpus ledger revision. It records provider-reported cost
   and conservative debit but never blocks on either. The runner prints every
   run-local USD 10 crossing and final spend. No financial ceiling exists for
   this authorized run. Registry v3 replaces the old numeric authorization
   field with the closed `cost_mode=report_only` contract; policy v2 records
   exact GPT-4.1/GPT-4.1 Mini reference prices of USD 2/8 and USD 0.40/1.60 per
   million input/output tokens. The immutable plan reports first-attempt and
   maximum-attempt estimates separately; neither is an execution gate.
   `_aggregate_preflight` selects behavior by plan/ledger schema: a valid v3
   conservative bound above the historical USD 17 starts normally, while old
   corpus schemas retain their historical budget enforcement.
8. Use at most five precommitted attempts for settled but structurally invalid
   author/reviewer responses. A transport-uncertain request is never retried:
   an uncertain author call rejects its exact precommitted singleton/pair; an
   uncertain reviewer call rejects only that row; later identities continue.
   Every uncertain entry has zero response hash, request ID equal to reservation
   ID, a journal task set, and byte-exact rejection lineage. Any missing or
   altered binding makes the run operationally inconclusive and prevents packet
   sealing.
   The new ledger pins `automatic_retries=4`, so attempt indices produce unique
   reservation IDs for the five total attempts.
9. Persist per-call diagnostics as `diagnostics/call-*.json` and a final
   create-only corpus report containing
   accepted/rejected counts by split, locale, domain, option count and reason;
   semantic disagreement totals; transport/validation diagnostics; Wilson
   projections; provider cost; conservative debit; cumulative research totals;
   and plan, ledger, packet and inventory hashes. No raw provider response,
   reasoning, task text or secret enters the report.
   Each diagnostic file is closed to its exact task-ID set and counts for
   `reviewed`, `scenario_disagreement`, `criterion_role_disagreement`,
   `local_privacy`, `reviewer_privacy_flags`, `author_response_failures`,
   `reviewer_response_failures`, `retry_recoveries` and
   `transport_uncertain_calls`. It contains no response body, reasoning or
   generated text.
10. Seal `phase4e-accepted-packet.v3`, preserving physical train/dev versus
    holdout separation and identity-only pre-claim validation. Its validator
    applies the post-pilot pair rule and remains accepted by the existing
    extraction/training lane. Accepted reviews use the relaxed diagnostic
    observation model, so repeated inferred roles are valid metadata rather
    than a schema failure. Old packet-v2 validation and strict review types
    remain unchanged.
    The v3 accepted-row invariant permits an accepted blind reviewer to disagree
    with the author's scenario, disagree on roles, or repeat inferred roles.
    Fake end-to-end fixtures include all three cases and still seal and
    pre-holdout-validate the packet when answer and quality gates pass.
11. Live artifacts use create-only paths:
    `.artifacts/phase4e/corpus-plan-v3/`,
    `.artifacts/phase4e/corpus-work-v47/`, and
    `.artifacts/phase4e/corpus-report-v3/`. The plan schema is
    `phase4e-universal-plan.v3` and the workflow revision is
    `phase4e-saracura-universal-synthetic.v2`; neither reuses the existing v2
    identifiers or `.artifacts/phase4e/plan-v3.json`. The sealed packet is
    written beneath
    `~/Library/Application Support/saracura/phase4e/model-artifacts/` and the
    work evidence, including private synthetic task text in `resolved/*.json`,
    is copied create-only to the existing private durable research root. It is
    never published or added to Git.
12. Register the new corpus ledger schema in `ledger_policy_for_schema` and in
    the durable catalog scanner before any live call. A mixed-root test must
    rescan historical pilot ledger v1-v5, historical
    `phase4e-cost-ledger.v2` corpus ledgers including an open reservation, and
    the new corpus ledger. Stage validation is keyed by schema version: the new
    schema accepts only closed `corpus_author`/`corpus_reviewer` stages without
    changing v2 semantics. Before adding the new run, its fixture reproduces
    the current v12 index baseline of 3,340 entries across 58 directories;
    after create-only copy, report and fresh-rescan totals agree exactly.
13. Plan v3 binds `author_system_sha256`, `reviewer_system_sha256`,
    `provider_policy_sha256`, `domain_scenario_map_sha256`, exact
    `reviewer_request`, model IDs, policy-v2 SHA and registry-v3 SHA. Provider,
    map and reviewer digests equal their v12 counterparts
    (`72efe778...908f3`, `946715b6...7575`, and `f5e2f107...27dc`);
    policy and registry hashes bind their new versioned bytes and therefore
    differ from v12. Validation compares every field before ledger creation or
    transport.
14. A fully resolved work directory can re-enter the same command in seal-only
    mode: it validates ledger, resolution and diagnostic bindings and creates a
    missing report/packet without a provider request. Tests force a post-call
    seal failure, then prove recovery creates identical artifacts with a
    transport that would fail if invoked.

## Non-scope

- No Laya/Jev/TypeSafe output, private e-mail, AIOS trace, customer data, web
  crawl or human-original record enters generation, selection or training.
- No base-encoder fine-tuning, architecture search, calibration, threshold,
  action gate, server, checkpoint publication, wheel-bundled weight or runtime
  registration.
- No change to Phase 3B/4B registries, historical pilot evidence or old corpus
  work directories.
- No retry of an outcome-unknown request and no model/provider fallback.

## Implementation phases

### A. Versioned policy and plan

Add the registry/policy revisions, versioned validators, domain-compatible
planner, immutable plan cost estimate and manifest routing. Prove pilot task IDs
remain disjoint and all 1,600 corpus identities and allocations are stable.
The policy keeps `base_encoder`, `ranker`, `training`, `capacity` and planning
allocations byte-equivalent to v1 apart from the explicit workflow/plan seed
binding and domain scenario map.
The plan also binds a deterministic interleaved author-batch order across
split, locale and option-count cells. This avoids making late dev/holdout cells
absorb every duplicate found against earlier train rows while preserving task
IDs, split assignments and answer-blind acceptance.

### B. Post-pilot acceptance and transport completion

Refactor only the proven v12 primitives needed by both lanes. Add explicit
post-pilot acceptance/pair functions, diagnostics persistence, conservative
uncertain closure, resume validation and report-only cost telemetry. Preserve
strict legacy functions and schemas for old artifacts.

### C. Packet v3 and training compatibility

Seal/validate the new packet without weakening holdout isolation. Update
training packet binding only enough to consume the explicit v3 schema. Fake
transport and fake embedding tests must complete the full lifecycle without
network or real model bytes.

### D. QA, read-only review and live run

Run full QA and GPT/Terra read-only review before provider transport. Then
generate the plan, prove credential-absent preflight creates no work/report,
run the 1,600-task corpus with the 1Password-injected OpenRouter key, verify the
durable copy and sealed packet, and append the outcome to the execution record.
The run stays attached to a foreground PTY and is observed with bounded,
content-free snapshots of resolved counts, rejection reasons, ledger states and
cost at least once per minute. If interrupted, the same command resumes only
after validating every settled and uncertain binding; it never replays a
resolved task and is not detached into an unattended background job.
The expected duration is hours rather than minutes: the fixed plan has 1,300
author calls and up to 1,600 reviewer calls, plus global duplicate scans. The
pre-live fake resume exercise is mandatory.

## Acceptance criteria

1. Historical v1/v2 policy and registry bytes validate unchanged; v3 alone
   authorizes the new model pair and research execution.
2. The post-pilot plan is deterministic, domain-compatible, disjoint from pilot
   IDs and contains exactly 1,600 tasks with the existing split/cell/pair allocation.
   Its actual schema identifier is `phase4e-universal-plan.v3`.
3. Reviewer answer/quality failure rejects a row; semantic metadata disagreement
   only increments closed diagnostics. Paired author-target mismatch still
   rejects both members, while an independently rejected sibling does not reject
   its otherwise accepted pair member.
4. Tests prove settled validation retries are bounded, transport uncertainty is
   never retried, exact tasks are conservatively rejected, later work continues,
   and tampered uncertainty lineage prevents packet sealing.
5. Resume never resends a task bound to a settled or uncertain call. Duplicate,
   partial or orphaned coverage fails closed.
6. A mixed durable research root containing pilot ledger schemas v1-v5,
   historical corpus ledger v2 with an open reservation and the new corpus
   ledger rescans successfully after copy and produces the report's exact
   cumulative totals.
7. Corpus packet v3 resolves all planned identities, meets every existing
   minimum, contains no pilot identity/private pattern/raw provider payload and
   passes the pre-holdout validator. The full post-claim validator is exercised
   only in synthetic fixtures during this increment; live holdout content is not
   opened before the irreversible Phase 4E.3b claim.
8. Training packet binding can read train/dev plus holdout identities without
   opening holdout text before the existing irreversible claim.
9. Default installation and build artifacts remain free of data, weights,
   credentials, absolute local paths and ML runtime downloads.
10. Full lint, type check, tests, manifest validation, package build, artifact
   inspection and `git diff --check` pass; GPT/Terra review returns `PASS`.
11. Only a verified live packet result authorizes Phase 4E.3b. Insufficient
    minima, operational ambiguity, provider-policy drift or privacy failure
    remains fail-closed and does not trigger a weaker rerun.
12. A v3 plan whose conservative telemetry exceeds USD 17 reaches the fake
    transport; the historical corpus plan remains budget-enforced.
13. Packet-v3 fixtures include accepted scenario disagreement, role disagreement
    and repeated roles, and seal/pre-holdout validation succeeds without
    relaxing answer or quality gates.
14. A forced post-transport seal failure resumes in seal-only mode with zero
    transport calls and byte-identical packet/report output.

## Pilot-derived feasibility and operator agreement

The decision to retain 1,600 tasks is deliberate. The pilot accepted 95.7% of
members overall, 91.4% of English rows and 91.4% of complete pairs. Its
option-count cells accepted 18-20 of 20. The existing plan places 13-14 English
rows in each dev/holdout option-count cell against a minimum of 10; the worst
observed 18/20 cell yield projects 11-12 accepted rows, leaving only a one-line
margin in the smallest cells. Reducing the plan would make those cells fragile,
so the existing 1,120/240/240 allocation remains preferable to a cheaper run.

Scaling v12's observed provider cost gives a rough first-attempt expectation of
about USD 6.49 for 1,600 tasks; conservative debit may be around USD 23.59 and
maximum-attempt telemetry will be higher. These are planning estimates, not
ceilings. The operator explicitly removed the financial limit and authorized
continued execution, with notification at each actual USD 10 crossing. That
authorization supersedes the historical Phase 4E USD 17 and USD 5/USD 10 stage
gates for this run while preserving full cost reporting.

Task IDs are deterministic from the preserved corpus seed, so some v3 task IDs
also appear in historical diagnostic `corpus-work` and `corpus-work-v2..v46`
journals. Identity uniqueness is scoped to a plan/work directory; the preflight disjointness gate
is against pilot identities, not every historical failed attempt. New lineage
always binds the v3 plan, ledger schema and create-only work directory.

The first ten-author-batch projection reports duplicate and near-duplicate
rates by locale as well as minimum feasibility, before any possible USD 10
crossing. The plan's PT-BR allocation is exactly 60%; because the packet also
requires PT-BR to remain at least 60% of accepted rows, any lower PT-BR
acceptance rate than English can fail this ratio regardless of magnitude. The
pilot's 70/70 PT-BR versus
64/70 English result makes that risk low but does not remove the gate.

## Validation commands

```bash
uv run --isolated --locked ruff check .
uv run --isolated --locked ruff format --check .
uv run --isolated --locked mypy src tests benchmarks
uv run --isolated --locked pytest -q
uv run --isolated --locked python -m benchmarks.validate_manifests
uv run --isolated --locked python benchmarks/validate_default_environment.py
uv build
uv run --isolated --locked python benchmarks/inspect_wheel.py <wheel> <sdist>
git diff --check
```

## Rollout and rollback

The change is additive and create-only. Before the live call, rollback is
deleting only uncommitted v3/v2 code and artifacts. After any provider call,
the ledger/work directory is evidence and is never deleted or replayed; a
failed run remains sealed and requires a new reviewed seed suffix, plan schema
revision and spec; the preserved base seed applies only to the first live v3
plan. The runtime remains unsupported throughout this increment, so rollback
never changes user-facing inference behavior.

## Critique resolution

Round 1 returned `NO-GO` on three concrete omissions: the request envelope was
not fully pinned, the proposed plan name reused an existing v2 schema, and the
durable scanner did not explicitly support the new ledger. This revision pins
every v12 request axis and digest, moves the plan/workflow to v3/v2 identities,
and adds mixed-root scanner support and acceptance evidence. It also records
the cell-margin rationale, operator cost agreement, exact prices, stable policy
blocks, relaxed accepted-review type and private durable-copy boundary.

Round 2 kept one request-contract blocker and added one durable-scan fixture
blocker. This revision pins the exact author/reviewer prompts, wire schema,
reasoning omission and plan digests, and tests the real mixed root including
historical corpus-v2 open reservations. It also makes sibling survival, delayed
training authorization, diagnostic keys, historical task-ID reuse, early
duplicate reporting, locale-margin risk and foreground resume monitoring
explicit.

Round 3 returned `GO` with no blocker. Its highest-risk improvements are now
acceptance criteria: v3 fixtures seal rows with scenario/role disagreement and
repeated roles; a report-only plan above USD 17 reaches transport; retries bind
attempt indices; and a post-transport seal failure recovers without another
provider call. The live lane runs only the pre-holdout validator, binds author
temperature, interleaves cells deterministically and treats the exact 60%
locale ratio as a known gate.

## Risks

- The pilot sample may overestimate 1,600-task yield. Existing deterministic
  minimum impossibility and Wilson projections stop a doomed run without
  weakening thresholds.
- Thousands of sequential provider calls expose tail latency. Conservative
  task-local uncertainty closure prevents a single unknown outcome from
  invalidating unrelated work while keeping it in the denominator.
- Relaxed semantic diagnostics could be mistaken for relaxed answer quality.
  The implementation keeps answer, natural-language, genericity, exclusivity,
  privacy, capacity, leakage and duplicate gates unchanged and reports the two
  concepts separately.
- Versioning several historical contracts can create accidental fallback. Every
  plan, policy, ledger and packet schema selects one exact validator; unknown or
  crossed combinations fail before network or holdout access.
