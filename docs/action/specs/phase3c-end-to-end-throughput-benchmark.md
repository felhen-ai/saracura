---
title: Phase 3C end-to-end throughput benchmark
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
globalRef: qmd://saracura/docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-21
sourceRefs:
  - https://openrouter.ai/labs/jev/compile
  - https://openrouter.ai/typesafe/jev-1.13/api
related:
  - docs/action/specs/phase2b-encoder-candidate-gate.md
  - docs/action/specs/phase3b-synthetic-research-training.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 3C end-to-end throughput benchmark

## Context

Phase 3B-S produced a synthetic-only five-class linear head over the immutable
`multilingual-minilm-l12` encoder. Its holdout evaluation measured only the CPU
head, not tokenization, device transfer, encoder inference, pooling, or transfer
back to CPU. Comparing that head-only timing with a remote Jev request would be
misleading.

The operator has authorized the end-to-end benchmark. The existing OpenRouter
authorization remains capped at USD 0.25 cumulatively across the synthetic
research work. A TypeSafe account is now available, but this phase does not use,
configure, or mutate that account. Jev remains a pinned external control through
OpenRouter under the existing ZDR and no-data-collection controls.

## Goal

Produce reproducible, synthetic-only evidence for the actual local serving
boundary and its behavior as decision volume grows:

1. measure Saracura from PT-BR strings through tokenization, MPS transfer,
   frozen MiniLM forward pass, mean pooling, CPU transfer, and linear-head
   choice;
2. report cold local load separately from warm-path request latency;
3. measure fixed local batches of 1, 8, 20, 32, 64, and 128 records;
4. measure Jev 1.13 through OpenRouter at batches 1, 8, and 20 using the
   documented multi-record/question pattern;
5. report request latency, per-item latency, aggregate throughput, memory, cost,
   and the exact boundaries and limitations needed to interpret the comparison.

The benchmark is performance evidence only. It does not retrain, calibrate,
select thresholds, publish the synthetic corpus, authorize automation, or claim
general decision quality.

## Inputs and provenance

The command consumes:

- the sealed Phase 3B-S packet manifest;
- the sealed training manifest, embeddings, and checkpoint;
- the verified local encoder snapshot from the closed encoder registry;
- the exact final Phase 3B evaluation ledger named by `--prior-cost-ledger` when
  Jev is enabled; its SHA-256 is bound into the Phase 3C result.

Both manifests must pass their existing validators before model load. The
training manifest must bind the unchanged training code and checkpoint. The
benchmark result binds the packet, training manifest, benchmark source bytes,
encoder revision, checkpoint, environment, fixed protocol, and every raw timing
sample by SHA-256. Generated input text, weights, and approved raw response
envelopes remain below ignored `.artifacts/` directories and outside the result
JSON. Credentials, authorization headers, account metadata, and host/home paths
never enter any artifact, result, report, error, or log.

## Frozen local protocol

- device: Apple MPS only; no silent CPU encoder fallback;
- encoder and tokenizer: the exact verified `multilingual-minilm-l12` snapshot;
- encoder mode: frozen `eval()` under `torch.inference_mode()`;
- head: the sealed Phase 3B-S CPU `Linear(384, 5)` checkpoint in `eval()` mode;
- inputs: accepted synthetic rows ordered by `family_id`;
- maximum length: 128 tokens, padding and truncation enabled;
- pooling: attention-mask-weighted float32 mean without normalization;
- batch sizes: 1, 8, 20, 32, 64, 128, executed in the deterministic
  counterbalanced order 1, 128, 8, 64, 20, 32;
- warmups: 5 per batch size, excluded from measured samples;
- measured iterations: 30 per batch size;
- timing clock: `time.perf_counter_ns()`;
- measured event sequence for every warmup and sample: synchronize MPS; start
  total; tokenize on CPU; capture tokenization boundary; transfer inputs to MPS,
  run the forward pass and float32 mask-weighted pooling; synchronize MPS and
  capture the device-work boundary; copy the pooled tensor to CPU, synchronize
  MPS and capture the transfer-complete boundary; run the CPU head, argmax, and
  materialize the concrete choices; stop total;
- component arithmetic is exact and recomputable from those monotonic
  boundaries: tokenization, CPU-to-MPS plus encoder/pooling, MPS-to-CPU, CPU
  head, and total. The result rejects negative components or a component sum
  inconsistent with total beyond clock-resolution tolerance;
- local model/tokenizer/head load is measured once in a fresh process and kept
  outside warm-path request latency. It is named `fresh_process_load_ms` and is
  explicitly page-cache-sensitive, not a guaranteed cold start;
- RSS is captured before load, after load, and after the final workload.

Each workload uses the first N rows from the accepted rows sorted by
`family_id`, and repeats that same slice across warmups and measured iterations.
The result binds the ordered opaque input-ID digest, token-length sequence, and
token-length digest for each batch; it contains neither family IDs nor text. An
untimed precheck rejects any untruncated input longer than 128 tokens, so a valid
run always records `truncated=false`. Predictions must have an identical digest
in every iteration. Raw samples are retained so p50 and p95 can be recomputed
with nearest-rank semantics. Report both request latency and latency per item;
aggregate throughput is total items divided by total measured seconds, not the
reciprocal of a percentile. Process RSS is sampled before load, after load, and
after the last workload; it is not presented as total or peak unified-MPS
memory.

## Frozen Jev protocol

Jev is pinned to `typesafe/jev-1.13`; aliases and fallback models are forbidden.
Requests use `POST /api/alpha/decisions` with provider controls
`zdr=true`, `data_collection=deny`, and `enforce_distillable_text=true`.

Phase 3C uses a new transport path and does not reuse the Phase 3B single-record
Jev helper. For each batch, the canonical request has
`state={"records":[{"id":"r000","record":{"text":...}}, ...]}` and one
question per record under the flattened key `r000__routing`. IDs are sequential
within a request and disclose no family identity. Each `choice` instruction
references exactly one record ID and uses the same five frozen label
definitions. The exact answer-key set must equal the generated question-key set
with no missing or extra answers. Every answer must be a five-label `choice`;
the response must include non-empty ID, provider `TypeSafe`, exact pinned-model
revision prefix `typesafe/jev-1.13-`, finite non-negative usage cost, and
non-negative integer token counts.

The canonical provider object is validated byte-for-byte before dispatch:
`zdr=true`, `data_collection=deny`, `enforce_distillable_text=true`,
`allow_fallbacks=false`, `require_parameters=true`, and `max_price` capped at
the reviewed Jev catalog prices. These fields are nested under `provider`, as
specified by the current OpenRouter `DecisionsRequest` schema. No alias,
fallback, extra request key, session ID, user ID, or trace is allowed.

- batch sizes: 1, 8, 20;
- measured iterations: 10 per batch size, executed in order 1, 20, 8;
- no discarded remote warmup calls;
- requests are strictly sequential;
- the total boundary includes Mac-to-OpenRouter network time and provider time;
- each response is reduced to answer digests, timing, resolved revision, usage,
  and aggregate correctness; raw provider content is not placed in the result;
- the response-level `usage.cost` is authoritative for the local spending
  guard, while
  `/api/v1/key` remains an eventually consistent audit signal;
- every call, including a later failed call with known cost, is recorded in the
  cumulative ledger before another request can start.

Remote timing starts after canonical request-byte construction and durable
reservation, immediately before transport dispatch, and stops after receipt of
the complete response body. It includes DNS, TCP, TLS, WAN, routing, and
provider time; it excludes JSON parsing, validation, digesting, and ledger
persistence. Every measured request uses a new HTTPS connection, a 60-second
timeout, disabled environment proxies and redirects, and exactly one attempt.
The `/api/v1/key` audit uses a separate connection before measurement and is not
a discarded Jev warmup.

## Remote budget state machine

The USD 0.25 authorization is a durable local single-process spending guard,
not a provider-side or account-global cap. Phase 3C consumes the existing Jev
stage allocation of USD 0.04 and the exact final Phase 3B evaluation ledger is
the mandatory predecessor. Currency is represented as fixed-point decimal
strings and compared with `Decimal`; binary floats are forbidden for gates.

Before the output directory is created, the runner exclusively creates a
durable claim keyed by the predecessor-ledger SHA-256 in a sibling ignored claim
registry. A pre-existing claim fails closed, preventing concurrent or repeated
consumption from the same predecessor. The claim is retained after success and
its digest is bound by the final artifact manifest.

Before every dispatch, the runner appends and fsyncs a reservation event to the
cost journal. The reservation uses the canonical request-byte length as a
conservative one-token-per-byte prompt bound at the capped catalog prompt rate,
plus the capped completion and request charges. It must fit both the remaining
USD 0.04 Jev allocation and the remaining USD 0.25 cumulative authorization.
After a validated response, a settlement event with response-level
`usage.cost` is appended and fsynced before another call may start. HTTP failure,
timeout, malformed/missing cost, cost above the reservation, process restart
with an unsettled reservation, or any uncertain provider outcome retains the
reservation, marks the journal uncertain, and permanently prevents another
dispatch. There are no retries or replacement samples.

Jev batch support is capped at 20 here because that is the maximum demonstrated
by the reviewed OpenRouter recipe, not because the API schema proves a universal
hard maximum.

## Comparison rules

The report may compare local and Jev only at shared batch sizes 1, 8, and 20.
It must never present head-only timing as end-to-end timing. It must state:

- local warm-path excludes one-time model load;
- Jev includes WAN and provider routing, but provider cold/warm state is not
  controllable;
- local and remote measured-iteration counts differ;
- synthetic PT-BR routing is narrow and cannot establish general quality;
- Jev accuracy is contextual diagnostic evidence, not a training or selection
  signal;
- throughput on one M4 Pro does not predict every deployment target.

The benchmark may calculate speedup as Jev per-item p50 divided by Saracura
per-item p50 only for the same batch size. It must report the underlying values
next to every ratio. Remote p95 over ten samples is the maximum (nearest-rank
rank 10); p95 is descriptive only and remote/local tail-latency ratios or claims
are prohibited.

## Implementation scope

Allowed:

- a checkout-only `benchmarks/e2e_benchmark.py` runner and closed `phase3c.v1`
  result models;
- offline unit tests with fake clocks, fake local components, and fake OpenRouter
  transport;
- README instructions and this spec;
- ignored local benchmark artifacts.

Prohibited:

- modifications to runtime inference APIs, installed package behavior, model
  weights, training rows, checkpoint selection, or calibration;
- direct use of the new TypeSafe account;
- browser automation or TypeSafe account configuration;
- telemetry, background uploads, remote fallback, aliases, arbitrary model
  paths, or request-triggered downloads;
- committing inputs, raw responses, weights, credentials, account metadata, or
  benchmark result artifacts;
- publishing a release or making production-readiness claims.

## Fail-closed behavior

The runner refuses:

- a modified, incomplete, non-final, or incompatible packet/training manifest;
- missing MPS, unverified encoder files, symlinks, unexpected artifact files, or
  a checkpoint/embedding digest mismatch;
- texts exceeding 128 untruncated tokens, non-finite tensors or timings,
  prediction drift across iterations, or malformed result summaries;
- network use without `--allow-network`, a budget other than the already
  approved USD 0.25, a missing prior cumulative ledger, an unpinned Jev response,
  answer-key mismatch, invalid choice, missing response cost, a reused
  predecessor ledger, an unsettled reservation, or a reservation that could
  cross a stage or total ceiling.

If Jev becomes unavailable after local measurement, the immutable result records
`external_control_unavailable` plus completed-call count and digests. Local
evidence remains valid, but no partial remote workload is compared as complete.
No aggregate remote summary or comparison is emitted for an incomplete series.

## Result artifact contract

The new output directory is created exclusively with mode `0700`. It contains
only regular, non-symlink files from this exact allowlist, each mode `0600`:
`result.json`, `report.md`, `cost-journal.jsonl`, and
`artifact-manifest.json`. The final immutable artifact manifest records size and
SHA-256 for the first three files, the exact source-path allowlist
(`benchmarks/e2e_benchmark.py`, `benchmarks/synthetic_research.py`, the encoder
registry, and this spec), their digests, both input-manifest digests, the
predecessor-ledger digest, and its own canonical self-digest. Validation rejects
unknown fields, extra files, links, permission drift, digest drift, non-finite
values, unsafe strings, or summary/raw-sample contradictions.

## Acceptance criteria

1. Default install remains offline/lightweight and does not import Torch or
   Transformers.
2. Offline tests prove fixed batch sizes/counts and orders, nearest-rank
   summaries and ranks, throughput math, exact MPS sync/timer order, timing
   arithmetic, deterministic prediction digests, exact Jev flattening and
   answer validation, canonical provider controls, disabled redirect/proxy and
   retry behavior, durable reservation/crash/restart/concurrency handling,
   unknown/billed-failure cost behavior, no dispatch after uncertainty, budget
   carry-forward, source/artifact allowlists, privacy scanning, summary
   recomputation, and lazy optional imports in the default environment.
3. The real M4 Pro run completes all six local batch sizes with 30 samples each.
4. The real Jev run completes batch sizes 1, 8, and 20 with 10 samples each, or
   records a bounded external-control failure without invalidating local data.
5. The report includes raw timing samples, p50/p95, per-item timings, throughput,
   load/RSS evidence, resolved versions, cost, and explicit limitations.
6. Existing manifests, 402+ tests, lint, typing, wheel/sdist inspection, and
   default-environment checks remain green.
7. No generated or sensitive artifact enters Git or the package.

## Validation commands

```bash
uv sync --locked --dev
uv run python benchmarks/validate_default_environment.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
git diff --check

UV_PROJECT_ENVIRONMENT=.venv-e2e uv sync --locked --dev --extra encoder-eval
UV_PROJECT_ENVIRONMENT=.venv-e2e uv run python -m benchmarks.e2e_benchmark \
  --training <training-manifest.json> \
  --packet <packet-manifest.json> \
  --prior-cost-ledger <cost-ledger.json> \
  --output-dir .artifacts/e2e-v1 \
  --allow-network --budget-usd 0.25
```

## Rollout and rollback

This is a checkout-only research addition. There is no runtime deploy, migration,
package publication, or TypeSafe-account action. Rollback is the Git revert of
the benchmark/spec/test commit. Ignored artifacts can remain for review until
the user explicitly ends the session; they are not a deployed state.

## Open risks

- MPS scheduling and thermal state can move short-run latency materially.
- WAN and OpenRouter/provider load dominate Jev timing and are time/location
  specific.
- Fixed repeated batches improve comparability but may not represent production
  message-length or language distributions.
- Jev's documented 20-row recipe is evidence of a supported pattern, not a
  published maximum-throughput contract.
- The synthetic taxonomy is unusually clean, so accuracy values are not the
  decision criterion for architecture or launch.
