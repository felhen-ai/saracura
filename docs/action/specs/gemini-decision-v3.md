---
title: Gemini Decision 1 benchmark V3 adapter
kind: spec
area: research
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/gemini-decision-v3.md
globalRef: qmd://saracura/docs/action/specs/gemini-decision-v3.md
reviewCadenceDays: 30
lastReviewedAt: 2026-11-12
sourceRefs:
  - https://aistudio.google.com/docs/decisions-api
related:
  - research/evaluation-guide.md
  - research/v3_spec.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Gemini Decision 1 benchmark V3 adapter

Evaluate the hosted model as an explicit remote comparator using unchanged frozen V3 inputs and scoring. Add only an opt-in standard-library research bridge, synthetic offline tests and usage documentation. The core runtime remains local-first.

## C1

Standalone stdlib research/gemini_bridge.py serves loopback POST /v1/systemone and health, model fixed gemini-decision-1, explicit GEMINI_API_KEY env only, fixed HTTPS Google endpoint /v1beta/decisions, no redirects, no fallback or downloads or ML packages.

## C2

Translate state verbatim to input and instructions verbatim. noul maps directly. choice uses zero-padded positional IDs whose alphabetical UTF-8 order equals incoming criteria order; descriptions retain original key and original description; reverse-map both winner and probabilities. No gold available. Support 32+ choices and full legal body within bounded generous request size; never truncate.

## C3

Validate upstream shape, winner membership, finite [0,1] probabilities/confidence, complete probability keys and approximately normalized sum with documented tolerance. Unknown choice, malformed/oversize body, redirects, timeout, 429/403, model mismatch fail closed. No upstream retries; error response to evaluator uses terminal 4xx to avoid automatic paid retries. No raw error text logged or exposed.

## C4

Mandatory append-only fresh telemetry journal with one sanitized record per outbound attempt, UTC timestamp, duration, status/error, original request digest, translated request digest, upstream response digest, usage when supplied, remapped answer probabilities/confidence and supplied model identity. No state/question text/credentials/headers/local paths/raw error. Serialize journal updates and requests, fail closed on journal errors. Model revision unpinned/unknown explicitly, never fake commit or digest. Health declares remote serving and no strict local weights evidence.

## C5

Meaningful offline synthetic tests cover transport with injected callable, order remapping incl 32 choices, string preserved, noul, invalid responses, no network on invalid request, no retry or redirects, full-length state, telemetry redaction and errors. Document setup, credential env injection, live preflight and V3 workflow without secrets, remote reproducibility limitation, no claim run succeeded.

## C6

Existing benchmark scoring/validation, runtime src/, frozen data and baselines untouched; research README or evaluation guide updated, ruff and focused tests pass. Orchestrator owns live API preflight, complete run, results and PR integration.

## Operational evaluation

The orchestrator locates an authorized corporate credential in 1Password and injects it only into the process. A synthetic live preflight must succeed before a small benchmark smoke and full historical source report plus the V3 Câmara and full legal replacement tasks, then combine/validate. Do not tune on the test answers. Preserve all attempts. Publish no training examples from hosted outputs. Report remote model identity and date honestly: a mutable API alias is not an immutable weights revision. Retain partial journals if access, quota, or transport prevents completion; never report a partial score as complete V3.

## Revision 2: critic resolutions

C3: Additionally: reject winner if it is not an argmax of complete probabilities (ties permitted). Never return 5xx, even on unexpected exception. Admit only one request with non-blocking lock (busy returns terminal 429, no queue); upstream timeout <=30s, pacing wait <=60s, and document evaluator timeout >=120s. Latch outbound access closed after any upstream/telemetry failure so the historical evaluator cannot keep spending after a failed item. Configurable minimum interval default 1s; bounds guarantee admitted call finishes before evaluator timeout under normal local execution. Never retry upstream or assemble results from separate partial attempts.

C4: Pin the first provider-returned model identity/version if available; fail closed on later drift. Never expose arbitrary upstream strings in journal: only safe bounded model identifier tokens, numeric counts, enum status/error and hash values. Usage missing is recorded unknown, never zero cost.

C5: Tests also cover busy rejection without outbound call, error latch preventing further calls, winner/argmax mismatch, provider identity drift, terminal handling of unexpected exceptions and pacing/timeout limits. Document hosted reports not eligible for historical immutable-checkpoint submission, V3 score consistency separate from submission eligibility. Raw per-item hosted telemetry remains outside public Git; only aggregate results may be committed after terms review.

C6: The main session runs the existing historical validator honestly and retains its expected immutable-revision failure; V3 combine/validate can still recompute score consistency without changing validators. Use same declared checkpoint id gemini-decision-1, revision unknown, base gemini-decision-1 across runs. Preflight must verify account limits before choosing pacing and exercise noul, 32 shuffled positions and full-size legal-equivalent synthetic text.
