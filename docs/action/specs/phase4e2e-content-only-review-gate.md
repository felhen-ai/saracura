---
title: Phase 4E.2e content-only review gate
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2e-content-only-review-gate.md
globalRef: qmd://saracura/docs/action/specs/phase4e2e-content-only-review-gate.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
  - docs/action/specs/phase4e2d-observable-review-gate.md
related:
  - benchmarks/saracura_universal_pilot.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2e content-only review gate

## Decision

Version the pilot again and make the review boundary explicit: quality and privacy flags are evaluated only from `instruction`, `state.summary` and criterion descriptions. Routing metadata (`task_id`, criterion IDs, locale, domain and JSON field names) is excluded from those judgments.

A name without an observable real-world anchor is treated as generic or invented. The reviewer must not search, speculate about collisions or demand provenance. `fictional=false` requires visible evidence such as a URL, account handle, credential, exact private address, official registration, known public-entity claim or content represented as a real private record.

## Verified context

- V3 resolved 24 rows: 22 failed `review_quality_fictional` and 2 exhausted review validation.
- `pilot_reviewer_view` includes `task_id`, locale, domain and criterion IDs so the reviewer can return a task-bound selected criterion.
- The v3 instruction referred to every visible identifier without excluding those required routing fields.
- The author prompt already prohibits real organizations, URLs, credentials, private data and long digit sequences in generated content.

## Scope and invariants

- Add v4 manifest, plan and report identities; preserve v1-v3 evidence unchanged.
- Bind the exact v4 reviewer-system digest into the v4 plan.
- Change only reviewer interpretation of fictionality/privacy scope; keep blindness, schemas, task allocation, retries, cost telemetry and acceptance thresholds unchanged.
- Use create-only `pilot-plan-v4`, `pilot-work-v4` and `pilot-report-v4` paths.
- Never resume or replay the interrupted v3 open reservation.

## Acceptance criteria

1. Tests assert that routing metadata is explicitly excluded and that unanchored names are treated as generic/invented.
2. The v4 plan validates the exact reviewer instruction digest.
3. Offline lint, typing, full tests and build pass.
4. The real v4 run resolves 140 tasks or reports exact operational incompleteness.
5. Phase 4E.3 starts only after a complete pilot `PASS`.

## Rollout, rollback and risks

The rollout is additive and create-only. Transport uncertainty is never replayed. A systematic early rejection pattern may stop an exploratory run for diagnosis, but never because of cost. The main risk is making the gate too permissive; explicit real-world anchors, independent privacy detection and the existing local sensitive-pattern scan remain fail-closed controls.

## Review resolution

The cross-model critic route was operationally unavailable twice in the immediately preceding recovery increments, so this narrowly evidenced follow-up proceeds with a degraded critique declaration. It will still receive read-only GPT/Terra review after implementation and full local QA before network execution.
