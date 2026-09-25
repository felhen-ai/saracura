---
title: Phase 4E.2h model pair upgrade
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2h-model-pair-upgrade.md
globalRef: qmd://saracura/docs/action/specs/phase4e2h-model-pair-upgrade.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
  - https://openrouter.ai/openai/gpt-4.1-mini
  - https://openrouter.ai/google/gemini-3.8-flash
related:
  - benchmarks/saracura_universal_pilot.py
  - benchmarks/saracura_universal_corpus.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2h model pair upgrade

## Decision

Keep the validated v6 wire protocol and acceptance thresholds, but replace the model pair for the create-only v7 pilot: `openai/gpt-4.1-mini` authors and `google/gemini-3.8-flash` reviews. The independent families reduce correlated judgment while both models support strict JSON Schema through OpenRouter.

## Verified context

- V6 accepted 12 of 72 resolved rows; even perfect remaining output could not reach the required 98 accepts.
- V6's provider-facing schema contains no `fictional` key, title or reason code, so the remaining rejection distribution is model-pair quality evidence.
- OpenRouter currently lists GPT-4.1 Mini with structured outputs at USD 0.40/M input and USD 1.60/M output.
- OpenRouter currently lists Gemini 3.8 Flash with structured outputs at USD 0.75/M input and USD 3.75/M output.
- Existing pilot stages pin DeepInfra/CoreWeave for the historical Qwen/Llama pair; those pins cannot route the new models.

## Scope and invariants

- Add v7 manifest, plan, report and exact artifact paths.
- Add an explicit pilot model pair and plan bindings for both IDs.
- Use provider routing with `require_parameters=true`, `data_collection=deny`, `zdr=true`, and provider fallback enabled; do not pin a vendor endpoint.
- Omit `reasoning_effort` for the non-reasoning GPT author model.
- Version the report-only ledger price identity to v3 while keeping v1/v2 ledgers readable.
- Use the same provider body in preflight and execution, including routing preferences and reasoning omission.
- Preserve v6 schema, blind reviewer payload, privacy, retries, thresholds, duplicate-key rejection and downstream review shape.

## Acceptance criteria

1. V1, v2 and v3 pilot ledgers are readable; v7 writes only v3.
2. Plan contains exact author/reviewer IDs and a digest of the routing policy.
3. Preflight and transport bodies use the same models, provider policy and no author `reasoning_effort`.
4. Historical corpus calls retain their existing models, pins and `reasoning_effort=none` behavior.
5. Full QA and read-only review pass before execution.
6. The real v7 run resolves all 140 tasks or stops with exact operational evidence; only `PASS` advances Phase 4E.3.

## Rollout and review

V1-v6 evidence remains immutable and is never resumed. The v7 run is report-only for cost and announces every USD 10 plus the final actual total. Cross-model critique remains degraded after repeated operational unavailability; GPT/Terra read-only review and full local QA remain required before execution.

## Review resolution

The read-only reviewer found the v7 model IDs, routing policy, omitted author reasoning parameter and ledger v3 coherent. It repeated the pre-existing Phase 4E runtime-workflow concern from `src/saracura/runtime/workflows.py`, which was introduced before this increment, is unchanged by this diff and has no installed backend or CLI capability. It therefore does not meet the SDD requirement that a blocker be caused or exposed by this model-pair change. No new v7 blocker remains.
