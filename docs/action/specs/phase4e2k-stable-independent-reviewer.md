---
title: Phase 4E.2k stable independent reviewer
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2k-stable-independent-reviewer.md
globalRef: qmd://saracura/docs/action/specs/phase4e2k-stable-independent-reviewer.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
related:
  - docs/action/specs/phase4e2j-reviewer-provider-compatibility.md
  - benchmarks/saracura_universal_pilot.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2k stable independent reviewer

## Decision

Keep GPT-4.1 Mini as author and replace only the operationally unreliable Gemini reviewer with the distinct GPT-4.1 model. Preserve the v9 prompt, schema, privacy policy, 120-second timeout, 512-token reviewer output capacity and all quality thresholds in a create-only v10 run.

## Verified context

- V9 proved the corrected Gemini request contract and resolved four pairs, but its fifth author batch ended on another reviewer timeout after 120 seconds.
- The v9 durable ledger contains 16 settled calls and one uncertain reviewer call and cannot be resumed.
- `benchmarks/saracura_universal_pilot.py:45-59` owns the pilot model and request constants; `:300-347` validates their exact manifest representation.
- `benchmarks/saracura_universal_pilot.py:670-710` binds the model IDs and reviewer request into the immutable plan.
- `benchmarks/saracura_universal_pilot.py:1380-1410` injects those exact values into the live client.
- Three consecutive exact-schema GPT-4.1 reviewer probes under ZDR/no-collection routing completed through Azure in 1.7-2.2 seconds with HTTP 200, valid structured output and USD 0.002662 each.

## Scope and invariants

- Add create-only v10 manifest, plan, work and report paths. V9 remains immutable and non-resumable.
- Change only the reviewer model from `google/gemini-3.8-flash` to `openai/gpt-4.1` and its report-only reference prices to USD 2/M input and USD 8/M output.
- Keep author model `openai/gpt-4.1-mini`; the author and reviewer remain distinct checkpoints and roles.
- Reviewer request remains 512 output tokens, `temperature: 0`, no reasoning parameter, strict structured output and the same answer-blind prompt/schema.
- Keep `require_parameters=true`, ZDR, no data collection, the 120-second timeout and all acceptance gates unchanged. Pin both models to the verified Azure endpoint and disable automatic provider fallback to preserve the repository's local-first no-fallback policy.
- Remove the planned Phase 4E workflow revision from the installed runtime registry until Phase 4E.3 seals a real checkpoint; its offline contracts remain available to the research lane.
- Do not retry or resume v9. Cost remains report-only with a user report at every run-local USD 10 milestone and final actual total.

## Acceptance

- Tests prove the exact v10 model pair, prices and reviewer request are sealed into manifest, plan, preflight and live body.
- V2-v10 manifests validate and tampering is rejected; pre-v10 artifact paths fail before ledger or transport access.
- Runtime tests prove `universal-choice@phase4e-saracura-ranker.v1` fails before backend access, and the live provider body proves Azure-only routing with fallback disabled.
- Full QA and GPT/Terra read-only review pass before network execution.
- Only a complete `PASS` authorizes Phase 4E.3; `FAIL` or `INCONCLUSIVE` remains fail-closed.

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

Cross-model critique remains explicitly degraded after repeated operational unavailability in the preceding Phase 4E recovery increments. GPT/Terra read-only review and full local QA are required before v10 execution.

The first GPT/Terra review found that the previously planned Phase 4E workflow was still accepted by the installed registry and that the v9 routing policy allowed remote provider fallback. Both findings were resolved in this increment: the installed runtime now rejects the planned workflow before backend access, and v10 pins author and reviewer to Azure with fallback disabled. Direct structured-output probes completed through Azure for both models, and the second read-only review returned `PASS` with no blockers.

Pre-execution QA passed on 2026-09-24: Ruff lint and format checks, mypy, all 879 tests with 29 expected skips, all 24 routed manifests, package build and `git diff --check`.
