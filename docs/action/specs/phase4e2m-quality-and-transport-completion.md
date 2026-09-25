---
title: Phase 4E.2m quality and transport completion
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2m-quality-and-transport-completion.md
globalRef: qmd://saracura/docs/action/specs/phase4e2m-quality-and-transport-completion.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
related:
  - docs/action/specs/phase4e2l-nonblocking-semantic-diagnostics.md
  - benchmarks/saracura_universal_pilot.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2m quality and transport completion

## Decision

Promote GPT-4.1 to author and use GPT-4.1 Mini as the distinct answer-blind reviewer. Preserve all v11 content, privacy and acceptance gates. Treat a transport-uncertain call as a conservative rejection of only its precommitted task or pair, never replay it, and continue the remaining create-only v12 plan.

## Verified context

- V11 eliminated reviewer-schema failures but accepted only 39 English rows; even accepting all unresolved English rows would reach 42, below the minimum of 45.
- Its English rejections were dominated by 16 semantic duplicates and 11 answer disagreements. PT-BR accepted 57 rows.
- A 20-task GPT-4.1 author probe produced 20 locally valid tasks, zero exact duplicates and one near-duplicate, with 3.38-9.30 second latency per paired call.
- A 20-row GPT-4.1 Mini reviewer probe accepted all 20 previously clean rows with no transport or schema failure and 1.984-second median latency.
- Azure produced one transport-uncertain call in each of v10 and v11 despite raising timeout from 120 to 240 seconds. Repeated timeout expansion is not a completion strategy.

## Scope and invariants

- Add create-only v12 manifest, plan, work and report paths. V11 remains immutable and non-resumable.
- Swap the v11 checkpoints: author `openai/gpt-4.1` at USD 2/M input and USD 8/M output; reviewer `openai/gpt-4.1-mini` at USD 0.40/M input and USD 1.60/M output.
- Keep roles distinct, Azure-only ZDR/no-collection routing, fallback disabled, temperature zero, no reasoning parameter and the exact v11 prompts and schemas.
- Return the fixed timeout to 120 seconds. A timed-out call is never retried because its provider outcome is unknown.
- Persist every affected task as rejected with `author_transport_uncertain` or `reviewer_transport_uncertain`, zero response hash and the immutable reservation/request identity. Count the call separately in response diagnostics and conservative debit.
- Continue later precommitted pairs after a handled uncertain call. An unpersisted, unbound or otherwise unhandled uncertain entry remains operational and makes the report `INCONCLUSIVE`.
- Keep all 140 tasks in the denominator. Conservative transport rejections cannot increase acceptance, relax per-locale or per-cardinality minima, authorize training data or enter a packet.
- Keep the strict corpus reviewer model and production/training paths unchanged. Pilot rows remain evaluation-only.

## Acceptance

- Tests prove one uncertain author pair and one uncertain reviewer task are persisted as rejections, never replayed, and do not stop later pairs.
- Tests prove an unhandled uncertain entry, open reservation or orphaned settlement remains `INCONCLUSIVE`.
- Tests prove the swapped prices/models, 120-second timeout and exact routing are sealed into manifest, plan, body and report.
- V2-v12 manifests validate; pre-v12 paths and tampering fail closed before ledger or transport access.
- Full QA and GPT/Terra read-only review pass before live execution.
- Only a complete v12 `PASS` authorizes Phase 4E.3. `FAIL` or `INCONCLUSIVE` remains fail-closed.

## Validation commands

```bash
uv run --isolated --locked ruff check .
uv run --isolated --locked ruff format --check .
uv run --isolated --locked mypy src benchmarks
uv run --isolated --locked pytest -q
uv run --isolated --locked python -m benchmarks.validate_manifests
uv build
git diff --check
```

## Review resolution

Cross-model critique remains explicitly degraded after repeated operational unavailability in preceding recovery increments. GPT/Terra read-only review and full local QA are required before v12 execution.

The first GPT/Terra review found two fail-closed gaps: uncertain HTTP/parse outcomes could retain a nonzero response hash, and resume validation did not prove that each persisted transport rejection matched its exact uncertain ledger entry and journal. V12 now keeps every uncertain response hash at zero with the reservation as request identity, and validates task set, stage, reason and all three lineage fields byte-for-byte before treating an uncertain call as handled. Missing, altered or unbound evidence remains `INCONCLUSIVE`.

Pre-execution QA passed on 2026-09-24: Ruff lint and format checks, mypy, all 883 tests with 29 expected skips, all 26 routed manifests, package build and `git diff --check`. The second GPT/Terra review returned `PASS` with no blockers.
