---
title: Local LLM bridge for comparable PT BR decisions
kind: spec
area: research
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/local-llm-bridge.md
globalRef: qmd://saracura/docs/action/specs/local-llm-bridge.md
reviewCadenceDays: 30
lastReviewedAt: 2026-10-21
sourceRefs:
  - research/ptbr_bench.py
  - research/rounds/2026-10-21/manifest.json
related:
  - docs/action/comparable-round-2026-10-21.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Local LLM bridge for comparable PT BR decisions

Expose a pinned offline Transformers model through a loopback System One server for the seven general LLMs selected in the local round. Native decision models retain upstream runtimes. The operator downloads a pinned snapshot explicitly; the server never downloads weights on startup/request and never uses paid providers. Default development installation stays lightweight.

Preserve every state character supplied by the evaluator, instructions and received option order. Render one official user chat message with text, question and letter-coded options; greedily generate at most32tokens and accept exactly one uppercase letter. Map to the original key. Boolean questions use A Não/B Sim and return a discrete numeric decision, without claiming confidence. Any malformed output fails the item; do not repair or retry content. No tests use benchmark golds or model weights.

Reject malformed requests, unknown models and oversized input before inference. Tokenize without truncation and reject when prompt plus output exceeds explicit context. Pin model/tokenizer revision, dtype/device, package versions, template kwargs and codehash in execution evidence. Heavy imports occur only at explicit CLI startup. Use Transformers AutoModelForCausalLM or the native appropriate conditional-generation class for Qwen3.5 text-only; no trust_remote_code. Bind only loopback, bounded reads, no raw exception/prompt logging or telemetry.

Acceptance C1-C7 is fixed in the operational Phase Contract: pure order-preserving core; closed bounded protocol; pinned offline runtime and context check; compatible HTTP plus metadata; meaningful synthetic tests; focused checks and guide. Root owns live smoke/full evaluations of all14 candidates and final integration. No existing GPU/CPU service is stopped; CPU/Mac execution uses only task-owned processes. If runtime/allocator/context fails, record concrete failure and continue the remaining candidates rather than relaxing the protocol.


## Failure, time and evidence contracts

Return HTTP 200 with an empty answer object for each requested question, e.g. {"answers":{"q":{}}}, and bounded error code for invalid model output or a generation deadline, so the evaluator records no_answer once and never regenerates content through its 5xx retry. Malformed requests and model mismatch are 400; too-large bodies are413; context overflow422. Genuine internal execution failures use500 and remain bounded. Limit request bodies to1MiB and question count16; choices2..26. Decode with skip_special_tokens=True, strip surrounding whitespace, then require one exact uppercase A..Z within option count; do not accept punctuation/explanations.

Serve threaded HTTP for health and use one inference lock. A busy request returns429 without queueing another generation. Transformers generation uses a stopping criterion with monotonic deadline (--generation-timeout, default90s); client timeout must be larger (default120s, root may raise both based on smoke). Deadline can only be checked between decoding steps, not interrupt a single kernel/prefill; document this limitation. When client disconnects, do not enqueue retry work; operator root owns process timeout and stops only its own server if hung. Test a fake slow generator and lock contention without long sleeps.

Health is the execution manifest: JSON schema_version, bridge codeSHA256, pinned model/tokenizer revision, backend package versions, dtype/device/threads, context/output/deadline settings, exact fixed template kwargs and tokenizer.chat_template SHA256, plus prompt-rendering version/hash. Default date kwarg is frozen2026-10-21; rendered time-dependent templates must receive explicit date kwargs and cannot call wall-clock strftime_now. Reject such templates until the backend renders deterministically (a fixed helper or fixed date accepted after inspection). Write --evidence-output PATH at startup with sibling temporary plus exclusive hardlink creation; no overwrite. Report --server-version is sha256 of exact evidence bytes, capture /health and compare before/after full. Root writes a sidecar linking report SHA, evidence SHA and source commits, retaining actual CLI argv. Evidence contains no request texts, endpoints or local paths. Generator never fakes confidence.

Tests in tests/test_llm_bridge.py must cover all C1-C7, explicit3/24options, PT-BR, boolean order, permutation, full state preservation, malformed and overbudget outputs, token/context rejection, invalid/oversize/model mismatch HTTP, deadline/lock contention, exclusive evidence write, stable health/evidence digest and lazy-import/offline loading. Use self-authored fixtures and fake tokenizer/generator. A synthetic recording harness must send the same evaluator call sequence to native-fake and bridge-fake endpoints and assert identical incoming item/order/question bytes. Root real14smokes/full remain separate.

A monotonic deadline applies to the whole request across questions; deadline takes precedence over even a valid partial letter. Expose only aggregate failure counters (invalid_output, deadline, overbudget) in health, never response texts. The evaluator ignores response error reasons, so root captures before/after counters and links them in the sidecar. Root freezes per-model template kwargs during preflight, and health hashes the actual template used.

C7 explicitly covers failure HTTP shapes, whole-request deadlines, inference locking, exclusive evidence output and deterministic date/template handling. Health has an immutable evidence object plus separate mutable counters. Freeze strftime_now to2026-10-21 in the isolated template environment; preserve the original template and register the fixed helper in evidence. Root keeps the zero-failure publication gate: a full LLM with any invalid output is failed/unranked, and may be stopped early to avoid wasting CPU. Do not silently add constrained decoding or accept punctuation. Nonthinking kwargs are mandatory when supported, with model-specific equivalents inspected in preflight.
