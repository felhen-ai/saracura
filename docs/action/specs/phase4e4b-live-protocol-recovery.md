---
title: Phase 4E.4B live protocol recovery
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4e4b-live-protocol-recovery.md
globalRef: qmd://saracura/docs/action/specs/phase4e4b-live-protocol-recovery.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-26
sourceRefs:
  - docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md
  - docs/action/phase4e-execution.md
related:
  - benchmarks/saracura_universal_comparison.py
  - benchmarks/saracura_universal_corpus.py
  - benchmarks/saracura_universal_pilot.py
  - tests/test_phase4e_comparison.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.4B live protocol recovery

## Outcome

Correct the comparison generator so it actually reuses the already validated
Phase 4E V12 reviewer contract, then execute one new reviewed blind comparison
attempt. The first live attempt remains immutable evidence and is never
reclassified: it resolved 200 identities with 0 accepted, 200 rejected, 495
reviewer validation failures and 2 transport-uncertain calls at USD 1.0181964.
Its result is `inconclusive_transport`, not a checkpoint-quality conclusion.

## Verified context

- `benchmarks/saracura_universal_comparison.py:927-951` currently recreates an
  obsolete provider prompt with the `fictional` field instead of calling the
  V12 pilot reviewer messages.
- `benchmarks/saracura_universal_comparison.py:1845-1918` currently sends
  `corpus.reviewer_schema(1)`, binds `corpus.ReviewerRecord`, and resolves with
  `corpus.acceptance_reason`. This reintroduces the distinct-role validator and
  makes scenario or criterion-role disagreement a rejection.
- `benchmarks/saracura_universal_pilot.py:168-194`, `625-680`, and `683-696`
  own the successful V12 provider-facing `generic_or_invented` prompt/schema,
  the repeated-role-compatible reviewer model, and diagnostic-only semantic
  disagreement behavior.
- `benchmarks/saracura_universal_pilot.py:2580-2620` records scenario and
  criterion-role disagreements as diagnostics while retaining answer,
  privacy, language, exclusivity, genericity, duplicate, and near-duplicate
  gates.
- `benchmarks/saracura_universal_corpus.py:2322-2390` and `2577-2655` apply the
  same exact `SequenceMatcher.ratio() >= 0.92` near-duplicate rule repeatedly.
  Full-packet validation makes this exact scan quadratic and dominated the
  first live attempt's local preflight.
- The private attempt terminal report and final cumulative ledger show 100
  settled author calls, 602 settled reviewer calls, 2 uncertain reviewer calls,
  89 terminal reviewer-response failures, 81 scenario disagreements, 24
  criterion-role disagreements, 3 genericity rejections and 1 answer
  disagreement. These are aggregate diagnostics only; no task text or provider
  payload enters this repository.

## Scope

1. Make the comparison reviewer call the canonical V12 pilot message, schema,
   binder, review model, acceptance function and pair-resolution function.
   Promote the pilot's private reviewer system message,
   semantic-disagreement classifier and inline local privacy diagnostic to
   public pilot-owned helpers, and make both pilot and comparison call those
   helpers. The local-privacy diagnostic runs over
   `usable + rejected_author` before capacity and reviewer filters, matching the
   current pilot ordering. Do not copy or fork these semantics inside the
   comparison module.
2. Persist `reviewed`, `scenario_disagreement`,
   `criterion_role_disagreement`, `local_privacy`, and
   `reviewer_privacy_flags` alongside the existing response, recovery and
   transport counters. Increment retry recovery after a later reviewer attempt
   validates successfully.
3. Keep semantic disagreement diagnostic-only. Keep answer disagreement,
   explicit rejection, privacy, language, genericity, exclusivity, capacity,
   exact duplicate, semantic duplicate, normalized near-duplicate and training
   overlap as hard rejection gates.
4. Introduce one internal exact helper for the existing
   `SequenceMatcher.ratio() >= 0.92` predicate. It may short-circuit only through
   `real_quick_ratio()` and `quick_ratio()`, which are upper bounds; the exact
   `ratio()` remains authoritative whenever both bounds can meet the threshold.
   Use it in both corpus review resolution and accepted-packet validation.
5. Add direct regressions for the provider schema, repeated reviewer roles,
   semantic disagreement diagnostics, answer disagreement, recovered reviewer
   retries, and equivalence of the optimized near-duplicate predicate to the
   original exact predicate across adversarial and seeded samples.
6. Add `phase4e-comparison-policy.v2` and `phase4e-comparison-plan.v2` with a
   new fixed seed and task/family namespace. Preserve policy and plan V1 as
   immutable history. The V2 planner must prove its 200 task IDs and 100 family
   IDs are disjoint from both training and the V1 plan before sealing. Policy
   V2 also binds a `phase4e-v12-pilot-reviewer.v1` contract revision plus the
   canonical provider schema and reviewer system-message digests. The source
   commit continues to bind binder, review model, acceptance, pair-resolution
   and diagnostic-helper code.
7. Preserve the first private live attempt byte-for-byte under a separate
   private attempt directory. Before the rename, create a private receipt over
   the V1 plan, terminal report and final cumulative ledger digests; after the
   rename, verify the same digests. The V2 plan binds those three digests, the
   V1 `inconclusive_transport` outcome and the V1 accepted count of zero. The
   required semantic-fingerprint and normalized-content disjunction is
   therefore an explicitly verified empty-set comparison, not an assumption.
   Create a fresh canonical comparison root and a new immutable plan bound to
   the merged recovery commit before any provider request. Never resume or
   overwrite the old plan, ledger, resolutions, diagnostics, or terminal
   report.
8. Version comparison diagnostics as
   `phase4e-comparison-diagnostics.v2`; V1 diagnostics remain readable only as
   archived evidence through their own aggregate verifier and are never mixed
   into a V2 work root.
9. Preserve V1 validation through explicit schema dispatch: policy V1/plan V1
   remain independently verifiable, while live `plan` emits only V2. Policy V2
   may differ from V1 only in policy schema version, seed, task namespace,
   family namespace, `execution.schemas.plan`, the added
   `execution.schemas.diagnostics`, and the added reviewer-contract binding.
   Plan V2 changes its schema version and planner revision and adds the reviewed
   previous-attempt/reviewer-contract bindings; all other plan semantics remain
   unchanged.
10. Extend the live `plan` CLI with required `--previous-attempt-dir` and
    `--previous-attempt-receipt` arguments. The directory must be the exact
    archived V1 attempt root. The planner reads and re-hashes the archived
    `comparison-plan/plan.json`, `terminal-report.json`, and the receipt-named
    final cumulative ledger; it never trusts receipt values alone. The closed
    receipt schema is `phase4e-comparison-attempt-receipt.v1` with attempt ID,
    relative final-ledger filename, the three file digests, plan schema,
    terminal outcome, accepted/unresolved/provider-call counts and
    provider-reported cost. The planner validates the V1 plan through the V1
    dispatcher, derives V1 task/family IDs from its slots, validates the V1
    terminal report schema and requires `inconclusive_transport`, `accepted=0`,
    `unresolved=0`, matching `plan_sha256`, provider-call count and ledger cost.
    Receipt or archived-byte drift aborts before plan creation.
11. Add a shared clean-source guard used by live `plan` and live `generate`.
    It requires an empty `git status --porcelain`, obtains the 40-character
    HEAD, and in `generate` requires equality with `plan.bindings.source_commit`
    before the first provider reservation. `generate` accepts only plan V2 and
    recalculates the pilot provider-schema and reviewer-system-message digests
    against policy and plan bindings before transport.
12. Audit every consumer found by `rg` for `COMPARISON_POLICY_PATH`,
    `COMPARISON_PLAN_SCHEMA`, `policy_sha256`, and `phase4e-comparison-`.
    Evaluation derives `policy_sha256` from the validated packet plan rather
    than a fixed path; sealing and loading dispatch by plan schema; manifest
    routing validates both canonical policy files. The public result schema and
    filename remain `phase4e-comparison-v1` because they version the first
    public scored result, not private attempt or plan revisions; that result
    binds the actual V2 plan and policy digests.
13. Remove the comparison-local unused `_semantic_fingerprint` and `_normalized`
    helpers so the corpus remains the only active content-normalization owner.

## Non-scope

- No model, checkpoint, tokenizer, renderer, retry count, provider route,
  privacy setting, acceptance minimum, scoring rule, public claim, calibration
  status, automation authority, or runtime registration change.
- No reinterpretation of the first attempt and no reuse of its generated rows.
- No publication until the new generation attempt reaches `passed`, followed by
  fresh-process Saracura and Laya scoring and the existing sanitization gates.
- No approximate duplicate algorithm, threshold change, embedding shortcut, or
  false-negative allowance.
- No change to the identical predicate still used by
  `benchmarks/synthetic_research.py`; that independent path is a named
  follow-up, not part of this live recovery.

## Invariants

- Reviewer input remains answer-blind and omits author attestation, family,
  pair, gold position, sibling and split metadata.
- Provider responses remain private and ledgered. Credentials remain injected
  from the authorized secret runtime and never enter Git, argv, logs or reports.
- Every provider call is charged and reported; cost never stops the run. The
  operator receives progress after each cumulative USD 10 interval.
- An uncertain call is never replayed. Any minimum failure with even one
  transport-uncertain resolution remains `inconclusive_transport`.
- All 200 planned identities stay in the denominator. Only a new reviewed plan
  may start a new attempt.
- Policy V2 changes only the exact fields enumerated in Scope 9. Models,
  routing, transport, costs, topology and acceptance minimums remain byte-equal
  to policy V1.
- Comparison keeps `prior_rows=(*training_rows, *accepted)` when resolving V12
  reviews. Only reviewer transport/binding, acceptance model, pair resolution
  and diagnostics come from the pilot; training-content overlap remains a hard
  comparison-only gate.
- Pilot pair resolution is deliberate: the global 60-complete-pair minimum
  measures paired evidence, while one sibling's independent hard rejection
  does not erase the other valid row. Author target mutation remains rejected
  before reviewer traffic and is rechecked at packet sealing.
- V2 diagnostic `reviewed` counts successfully bound review rows, not provider
  calls. `reviewer_privacy_flags` counts successfully bound rows whose reviewer
  privacy flag is true.

## Acceptance criteria

1. Policy V2 has a new seed and namespace, while every protocol field other
   than the exact Scope 9 allowlist equals V1. Plan V2 proves task/family disjunction from the V1 plan and
   training, binds the verified V1 terminal/plan/ledger digests, and accepts the
   content-disjunction proof only because the bound V1 terminal report records
   exactly zero accepted rows. A changed receipt, plan, terminal report or
   receipt-named final ledger aborts before the V2 plan is sealed.
2. The captured comparison reviewer request body and messages are
   byte-equivalent to `pilot_reviewer_schema(1)` and
   `pilot_reviewer_messages(row)`. The provider schema contains
   `generic_or_invented`, contains no provider-facing `fictional` field, and is
   tested through a fake transport rather than by comparing a helper to itself.
3. A structurally valid review with repeated semantic roles can pass the hard
   answer and content gates; its semantic mismatch is counted but does not
   reject the row.
4. A scenario mismatch and a criterion-role mismatch each remain visible in
   persisted diagnostics, while answer disagreement and every existing hard
   quality/privacy gate still reject.
5. A reviewer response that validates on a later bounded attempt increments
   `retry_recoveries`; exhausted malformed responses and uncertain transport
   keep their prior fail-closed resolution. A settled reviewer exception with
   lineage but no valid review is also resolved once as
   `reviewer_response_failure`; a fake settled-error transport proves all
   affected identities remain in the 200-row denominator and none are left
   unresolved.
6. Pair resolution matches V12 deliberately: one sibling's independent hard
   rejection does not convert the other sibling into `paired_member_rejected`,
   and semantic-attestation disagreement remains diagnostic. The minimum of 60
   complete accepted pairs is still enforced globally. A direct test mutates an
   author's semantic attestation away from its precommitted slot and proves the
   row is rejected before reviewer traffic.
7. The comparison imports public pilot-owned helpers for semantic-disagreement
   classification and local-privacy diagnostics; the pilot itself uses the same
   helpers, and neither path duplicates or imports private classifier
   implementations.
8. Comparison diagnostics V2 uses a closed schema containing all declared
   counters and rejects a V1 file in a V2 work root.
9. The optimized near-duplicate helper, with the fixed argument order
   `(candidate, existing)`, returns exactly the same boolean as
   direct `SequenceMatcher(...).ratio() >= threshold` for fixed adversarial
   cases and at least 1,000 seeded string pairs, including strings longer than
   200 characters where `autojunk` applies, including cases where only the
   second argument exceeds 200 characters and one case whose exact ratio equals
   `0.92`. Existing packet tamper and duplicate tests remain green.
10. Default import stays offline and does not import Torch, Transformers,
   Tokenizers or Safetensors. The comparison CLI remains checkout-only.
11. Ruff, formatting, mypy, the Phase 4E comparison suite, the training suite and
   the full repository suite pass before merge.
12. Independent read-only review returns `PASS`; the PR is merged and the main
   checkout is updated before a new live plan is created.
13. The old private attempt remains independently hashable before and after its
   same-volume rename, and the new plan binds the new source commit. No provider
   request occurs before both proofs. A V1 generation binding cannot be loaded
   into the V2 root. Fake transports prove dirty source, divergent HEAD, changed
   reviewer-contract digest and plan V1 all fail before any provider call.
14. If the new generator reaches `passed`, scoring, evaluation and sanitized
    publication continue under the existing Phase 4E.4 spec. Any other outcome
    is reported unchanged and cannot be promoted into a model-quality claim.
15. Manifest routing accepts the canonical comparison policy V1 and V2 files,
    evaluation and publication carry the V2 digest from the validated plan, and
    the public result retains the reviewed `phase4e-comparison-v1` result schema
    and filename.

## Validation commands

```bash
uv run ruff check benchmarks/saracura_universal_comparison.py benchmarks/saracura_universal_corpus.py benchmarks/saracura_universal_pilot.py tests/test_phase4e_comparison.py tests/test_phase4e_pilot.py
uv run ruff format --check benchmarks/saracura_universal_comparison.py benchmarks/saracura_universal_corpus.py benchmarks/saracura_universal_pilot.py tests/test_phase4e_comparison.py tests/test_phase4e_pilot.py
uv run mypy src
uv run pytest -q tests/test_phase4e_comparison.py tests/test_phase4e_pilot.py tests/test_phase4e_training.py
uv run python -m benchmarks.validate_manifests
uv run pytest -q
uv build
```

## Rollout and rollback

Merge through the normal protected-branch PR flow. After main is updated,
write the private V1 receipt create-if-absent through an atomic sibling temporary
file at `phase4e/comparison-attempts/attempt-v1-receipt.json`, mode `0400` and
outside both attempt roots; archive the first attempt by an atomic same-volume
rename, verify its plan, terminal-report and final-ledger digests at the
destination, then create the new canonical root and plan. Rollback before
provider traffic is removal of the empty new root and restoration of the
archived directory name. After any provider
reservation, evidence is immutable: rollback means stop, retain the attempt,
and do not publish or retry under the same plan.

Provider progress and terminal cost inside attempt V2 remain ledger-local. The
operator-facing running total also includes the V1 amount read from the bound
terminal report (currently USD 1.0181964) and is reported
when the combined comparison-attempt spend crosses each USD 10 interval; neither
number is a stop condition.

## Critique record

Round 1 found that merely rebinding the same plan to a new commit would reuse
the V1 seed, namespaces and all identities, contrary to the Phase 4E.4 retry
gate. The spec now requires policy/plan V2, a new seed and namespaces, direct
task/family disjunction and an explicit empty accepted-content proof.

Round 2 found ambiguity between V1/V2 policy fields and private pilot helpers.
The spec now enumerates every permitted policy difference, binds the reviewer
contract explicitly, preserves V1 schema dispatch and promotes the pilot-owned
system message and diagnostic helpers to public single sources of truth.

Round 3 identified missing operational proofs around archived V1 bytes, clean
source/HEAD enforcement and downstream policy consumers. Although these paths
were inspectable in earlier rounds, the findings were verified and incorporated
because they are consequential: the planner re-hashes the archive instead of
trusting a receipt, generation revalidates source and reviewer-contract digests
before transport, and every policy/plan/result/manifest consumer has an explicit
V2 disposition.

## Risks

- Reusing the V12 reviewer contract should raise acceptance substantially, but
  does not guarantee the 150-row minimum or any Saracura advantage.
- Exact quick-bound filtering can still reach the slow exact matcher for highly
  similar strings; it improves avoidable work without changing the predicate or
  promising a fixed runtime.
- Two transport-uncertain calls occurred in the first attempt. The existing
  no-replay rule can still make a later minimum failure inconclusive even when
  the content protocol is correct.
