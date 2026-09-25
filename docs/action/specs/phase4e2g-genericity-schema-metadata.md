---
title: Phase 4E.2g genericity schema metadata
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2g-genericity-schema-metadata.md
globalRef: qmd://saracura/docs/action/specs/phase4e2g-genericity-schema-metadata.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
related:
  - docs/action/specs/phase4e2f-genericity-wire-contract.md
  - benchmarks/saracura_universal_pilot.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2g genericity schema metadata

## Decision

Replace the inherited JSON Schema metadata of `generic_or_invented`. Its provider-facing property must use title `Generic Or Invented` and an observable description, and the complete provider schema must contain no `Fictional` title or `fictional` property.

## Verified context

- V5 renamed the property key but reused the Pydantic schema object generated for internal field `fictional`.
- The resulting provider schema was `{"title":"Fictional","type":"boolean"}` under key `generic_or_invented`.
- V5 resolved 10 rows: one accepted, four failed the mapped fictionality gate, four failed exclusivity and one was a local semantic duplicate.

## Scope, invariants and acceptance

- Add v6 manifest, plan and report identities and exact v6 paths.
- Construct the provider genericity property explicitly; do not reuse internal schema metadata.
- Keep the v5 key mapping, duplicate-key rejection, blindness, privacy, retries, thresholds and internal storage contract unchanged.
- Tests assert the exact title/description and absence of `Fictional`/`fictional` anywhere in the provider schema.
- Full QA and read-only review pass before a create-only v6 network run.
- Only a complete pilot `PASS` advances Phase 4E.3.

## Rollout and review

V1-v5 evidence remains immutable and v5 is never resumed. Cross-model critique remains degraded after repeated operational unavailability; GPT/Terra read-only review and full local QA remain required before execution.
