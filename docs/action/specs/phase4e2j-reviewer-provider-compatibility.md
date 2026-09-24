---
title: Phase 4E.2j reviewer provider compatibility
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2j-reviewer-provider-compatibility.md
globalRef: qmd://saracura/docs/action/specs/phase4e2j-reviewer-provider-compatibility.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
related:
  - docs/action/specs/phase4e2i-reviewer-timeout-recovery.md
  - benchmarks/saracura_universal_corpus.py
  - benchmarks/saracura_universal_pilot.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2j reviewer provider compatibility

## Decision

Keep the v8 model pair, 120-second timeout, privacy routing, prompts and quality thresholds. Create a non-resumable v9 that omits `temperature` only from the Gemini reviewer request, requests minimal excluded reasoning, and raises reviewer output capacity from 192 to 512 tokens.

## Verified context

- V8 settled its GPT-4.1 Mini author call and received a provider error for its first Gemini reviewer call, leaving 140 tasks unresolved.
- The OpenRouter endpoint inventory showed that ZDR filtering leaves a Google Vertex endpoint which supports structured outputs, reasoning and `max_tokens`, but does not advertise `temperature`.
- `benchmarks/saracura_universal_corpus.py:2904-2937` currently emits `temperature: 0` for every provider request.
- `benchmarks/saracura_universal_pilot.py:440-474` constructs both author and reviewer preflight bodies through that shared builder.
- `benchmarks/saracura_universal_pilot.py:1602-1638` sends one blinded reviewer row per request with a 192-token output limit.
- An exact-schema probe using the v8 reviewer prompt, schema, model and privacy policy returned HTTP 200 with valid structured output after omitting `temperature`, setting `reasoning={effort:minimal, exclude:true}` and using 512 output tokens.

## Scope and invariants

- Add create-only v9 manifest, plan, work and report paths. V8 remains immutable and non-resumable.
- Parameterize provider request temperature and reasoning without changing historical defaults: author and corpus requests retain `temperature: 0`; historical reviewer behavior retains its existing default unless explicitly overridden.
- V9 Gemini reviewer requests omit `temperature`, bind minimal excluded reasoning, and use 512 output tokens. These exact parameters are sealed into the manifest and immutable plan.
- Keep `require_parameters=true`, ZDR, no data collection, model IDs, prompt/schema hashes, acceptance thresholds and the 120-second timeout unchanged.
- Do not retry any uncertain v8 call. Only the new v9 identities and create-only paths may execute.
- Cost remains report-only. Report each run-local USD 10 milestone and the final provider-reported total; never stop for finance.

## Acceptance

- Tests prove the default request retains `temperature: 0`, the v9 reviewer omits it, and the exact reasoning/output policy is bound into the plan and live request.
- The manifest router validates v2-v9 and rejects tampering.
- Preflight, full QA and GPT/Terra read-only review pass before network execution.
- Only a complete v9 `PASS` authorizes Phase 4E.3; `FAIL` or `INCONCLUSIVE` remains fail-closed.

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

Cross-model critique remains explicitly degraded after repeated operational unavailability in the preceding Phase 4E recovery increments. GPT/Terra read-only review and full local QA are required before v9 execution.

## Validation evidence

- Live exact-schema probe: HTTP 200, valid structured output, zero reasoning tokens and USD 0.0013995 provider-reported cost with the v9 reviewer request.
- Focused QA: `121 passed`; all 23 routed manifests validate.
- Full QA: Ruff lint and format pass; mypy passes on 73 source files; `879 passed, 29 skipped`; source and wheel builds pass; `git diff --check` passes.
- GPT/Terra read-only review: `PASS`, with no blocking or non-blocking findings.
