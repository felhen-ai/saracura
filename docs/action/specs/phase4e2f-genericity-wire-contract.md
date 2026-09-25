---
title: Phase 4E.2f genericity wire contract
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2f-genericity-wire-contract.md
globalRef: qmd://saracura/docs/action/specs/phase4e2f-genericity-wire-contract.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
related:
  - benchmarks/saracura_universal_pilot.py
  - benchmarks/saracura_universal_corpus.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2f genericity wire contract

## Decision

Expose `generic_or_invented` instead of `fictional` in the pilot reviewer schema and prompt. Map the provider field locally to the existing internal `fictional` field before typed validation and acceptance. This preserves downstream compatibility while asking the provider only for the observable judgment the gate actually needs.

## Verified context

- V2 rejected 127 rows as `review_quality_fictional` under the original epistemic instruction.
- V3 made the rule observable but still rejected 22 of 24 resolved rows on the same field.
- V4 excluded routing metadata and defined real-world anchors but rejected all 10 reviewed rows on the same field.
- `corpus.reviewer_schema()` derives the provider property directly from `ReviewerGeneratedRecord.fictional`.
- `corpus.bind_reviewer_record()` validates the provider record and materializes the internal `ReviewerRecord`.

## Scope and invariants

- Add a pilot-only schema adapter that replaces provider property `fictional` with `generic_or_invented` and preserves every other field and constraint.
- Add a pilot-only binder that accepts exactly `generic_or_invented`, maps it to `fictional`, removes the wire-only field and then delegates to the existing typed binder.
- Bind the exact v5 prompt and recovery manifest into a v5 plan.
- Preserve reviewer blindness, privacy, acceptance thresholds, downstream storage shape and v1-v4 evidence.
- Use exact create-only v5 paths and never resume v3 or v4 ledgers.

## Acceptance criteria

1. The pilot provider schema has `generic_or_invented` and no `fictional` property.
2. Missing, duplicate or wrong-type genericity fields fail local validation and enter existing bounded retry behavior.
3. The local binder produces the unchanged internal `fictional` boolean.
4. Preflight estimation and real execution use the same pilot-only schema.
5. Lint, formatting, typing, full tests, build and read-only review pass before network execution.
6. The real v5 pilot resolves all 140 tasks or reports exact operational incompleteness; only a complete `PASS` advances Phase 4E.3.

## Rollout, rollback and risks

The rollout is additive and create-only. Transport uncertainty is never replayed. V4 remains the rollback behavior. The main risk is accidental schema drift between preflight and execution; both must call the same pilot schema function and the plan binds the exact system instruction digest.

## Review resolution

Cross-model critique remains declared degraded after repeated operational unavailability in the immediately preceding increments. A GPT/Terra read-only review and complete local QA remain mandatory before the v5 network run.
