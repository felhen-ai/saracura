---
title: Phase 4E.2d observable review gate
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2d-observable-review-gate.md
globalRef: qmd://saracura/docs/action/specs/phase4e2d-observable-review-gate.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
  - docs/action/specs/phase4e2c-execution-first-pilot-recovery.md
related:
  - benchmarks/saracura_universal_pilot.py
  - benchmarks/saracura_universal_corpus.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2d observable review gate

## Decision

Replace the blind reviewer's unprovable fictionality instruction with an observable-content rule. The existing `fictional` response field remains wire-compatible, but its operational meaning becomes: the supplied row contains only generic or invented entities and contains no identifiable real person, organization, account, URL, credential, private record or claim tied to a real record.

The reviewer must set `fictional=true` when the visible content meets that rule. It must not require external provenance, explicit declarations of synthetic origin or proof about the author's intent. Privacy detection remains an independent, fail-closed gate.

## Verified context

- `benchmarks/saracura_universal_pilot.py:107-124` tells the reviewer not to treat synthetic intent as proof and then requires a positive fictionality conclusion. The v2 pilot showed that this wording is systematically non-actionable for a blind reviewer.
- `benchmarks/saracura_universal_pilot.py:500-523` constructs the answer-blind reviewer request; it excludes pair identity, gold position, answer, split and semantic target.
- `benchmarks/saracura_universal_corpus.py:1655-1698` maps `fictional=false` to `review_quality_fictional` independently of semantic disagreement.
- `benchmarks/saracura_universal_pilot.py:1040-1074` keeps semantic disagreement diagnostic-only and applies the unchanged minimum-acceptance thresholds.
- The sealed v2 outcome resolved all 140 tasks with 127 `review_quality_fictional` rejections, zero privacy violations and zero privacy flags. That concentration identifies the instruction itself as the next hypothesis to test.

## Scope

1. Add a v3 recovery manifest and v3 plan/report identities. Preserve the v1 and v2 manifests and evidence unchanged.
2. Bind the v3 plan to the exact v3 recovery-manifest digest and a digest of the reviewer system instruction.
3. Define `fictional=true` entirely from visible content: generic or clearly invented entities pass; identifiable real-world or sensitive entities fail. Absence of prohibited identifiers is sufficient; external proof is neither available nor required.
4. Keep reviewer blindness, selected-criterion validation, natural-language and exclusivity gates, privacy checks, report-only cost accounting, retry behavior and acceptance thresholds unchanged.
5. Run the complete 140-task protocol in new `pilot-plan-v3`, `pilot-work-v3` and `pilot-report-v3` paths. Report each USD 10 milestone and final actual provider cost.

## Non-scope

- Relaxing privacy, selected-answer, natural-language or exclusivity checks.
- Accepting a row merely because the author says it is synthetic.
- Sending hidden planner, answer, pair or lineage metadata to the reviewer.
- Changing task allocation, answer positions, locales, option-count coverage or minimum acceptance thresholds.
- Starting corpus generation, training, calibration, runtime registration or publication before a complete pilot reaches `PASS`.
- Adding a financial ceiling.

## Invariants

- V2 evidence is never overwritten or reinterpreted as a successful quality run.
- The v3 plan cryptographically identifies both the recovery manifest and reviewer instruction.
- Reviewer payload fields remain exactly `task_id`, `locale`, `domain`, `instruction`, `state` and `criteria`.
- Raw provider content and reasoning are not persisted.
- Every settled provider call has a journal binding and every resolved task has final lineage.
- A provider or local validation failure remains stage-specific and cannot stop unrelated pairs.

## Acceptance criteria

1. Tests prove the v3 system instruction defines the observable rule and contains no requirement for external proof of fictionality.
2. The v3 plan includes and validates an exact reviewer-system SHA-256.
3. A generic/invented fixture with no sensitive identifiers can be accepted when the reviewer returns `fictional=true`; a privacy or identifiable-real-world fixture remains rejected.
4. V2 manifest and sealed evidence paths remain unchanged.
5. Offline quality gates and the full test suite pass.
6. The real v3 pilot resolves all 140 identities or returns `INCONCLUSIVE` with exact operational diagnostics.
7. A real `PASS` is required before Phase 4E.3 corpus generation can start.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src benchmarks
uv run pytest -q
uv build
git diff --check
```

## Rollout and rollback

The rollout is additive and uses create-only v3 paths. If offline validation fails, do not call the provider. If a live call becomes transport-uncertain, preserve the v3 journal and report `INCONCLUSIVE`; do not replay the uncertain call. Rollback is selecting the committed v2 implementation while retaining all v3 evidence for audit.

## Risks

- Wording that is too permissive could turn the fictionality gate into a duplicate privacy check. Keep the rule explicit about identifiable real-world entities and real-record claims.
- Wording that is still epistemic rather than observable could repeat the v2 systematic rejection. Bind and test the exact instruction before the live run.
- A prompt-only change may improve fictionality decisions while leaving semantic disagreement high. That disagreement remains measured and must be addressed separately before corpus promotion if the existing acceptance thresholds do not expose it.

## Review resolution

The cross-model critic was operationally unavailable and returned no review. The GPT/Terra read-only reviewer confirmed that the v3 instruction implements the observable rule, preserves the six-field blind payload and binds the exact instruction digest into the plan.

It repeated the pre-existing Phase 4E workflow-contract concern in `src/saracura/runtime/workflows.py`. That path was introduced in commit `2a07d6e`, is unchanged by this increment, has no installed backend or CLI capability, and was already recorded as out of scope in the reviewed Phase 4E.2c recovery spec. It therefore does not meet the SDD blocking criterion of being caused or exposed by this change. No new blocker remains in the v3 diff.
