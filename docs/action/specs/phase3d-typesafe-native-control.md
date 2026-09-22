---
title: Phase 3D TypeSafe native control benchmark
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase3d-typesafe-native-control.md
globalRef: qmd://saracura/docs/action/specs/phase3d-typesafe-native-control.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-21
sourceRefs:
  - https://docs.typesafe.ai/api
  - https://docs.typesafe.ai/models
  - https://docs.typesafe.ai/primitives/choice
  - https://docs.typesafe.ai/confidence
related:
  - docs/action/specs/phase3b-synthetic-research-training.md
  - docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 3D TypeSafe native control benchmark

## Context and current truth

Phase 3C measured the synthetic Saracura research checkpoint locally and Jev
1.13 through OpenRouter. That result includes gateway, network, routing, and
provider time. It can support a latency, selected-label, model-identity, and
cost comparison against the native TypeSafe endpoint. It cannot support a
probability or confidence comparison because Phase 3C did not retain those
OpenRouter values.

The current TypeSafe API accepts `POST https://api.typesafe.ai/v1/systemone`.
The versioned model ID is `jev-1.13.0`; `jev-latest` is a moving alias and is
therefore unsuitable for a sealed comparison. Choice answers expose the chosen
label, a full probability distribution, and confidence. The response exposes
input and output token counts but no authoritative monetary-cost field. The
published price is USD 0.042 per million input tokens and output tokens are
free.

The operator has authorized a direct TypeSafe benchmark and has USD 10 account
credit. This phase imposes a narrower local USD 0.25 ceiling. Credentials are
resolved at process launch from the operator's approved secret manager into
`TYPESAFE_API_KEY`. No secret value, private vault reference, account metadata,
or Felhen-specific configuration may be committed or written to an artifact.

## Goal

Produce sealed, reproducible, synthetic-only evidence that compares the same
fixed routing workload across:

1. the prior Phase 3C local Saracura measurements;
2. the prior Phase 3C Jev-through-OpenRouter measurements; and
3. Jev 1.13.0 through the native TypeSafe endpoint.

The native lane must retain enough typed evidence to inspect its own latency,
throughput, selected labels, probability distributions, confidence, token use,
and computed cost without retaining prompt text or raw response bodies. Only
the fields retained by both lanes may be compared with OpenRouter.

This phase is a control experiment only. It does not modify Saracura runtime
behavior, train or calibrate a model, select confidence thresholds, authorize
automation, publish a dataset, or establish TypeSafe/OpenRouter weight parity.

## Inputs and provenance

The checkout-only runner consumes:

- a complete, sealed Phase 3C artifact directory whose native/local and
  OpenRouter control result validates through the existing Phase 3C validator;
- the exact final synthetic packet whose manifest digest equals the Phase 3C
  `packet_manifest_sha256`;
- the Phase 3D source and this spec.

The Phase 3C result must have `jev.status=complete`, all 30 remote samples, and
the fixed batch protocol `1, 20, 8` with ten samples per batch. The input rows
are the first N accepted rows ordered by `family_id`, exactly as in Phase 3C.
Their opaque input-ID digest must equal the matching Phase 3C local and
OpenRouter workload digests before any network request.

The result binds the predecessor artifact-manifest SHA-256, packet-manifest
SHA-256, Phase 3C request-builder source SHA-256, Phase 3D benchmark source
SHA-256, spec SHA-256, exact model ID, protocol, per-batch canonical
payload-structure digests, and all reduced samples. Synthetic source text
remains in ignored predecessor artifacts and never enters the Phase 3D result.

## Frozen native TypeSafe protocol

- endpoint: literal `https://api.typesafe.ai/v1/systemone`;
- model request: exact versioned ID `jev-1.13.0`;
- request shape: produced by calling the Phase 3C `build_jev_request` function,
  verifying its closed shape, removing only the OpenRouter-only `provider`
  object, and changing only the model ID. This preserves the same
  `state.records`, flattened question keys, Portuguese instructions, label
  order, and taxonomy definitions;
- batch sizes: 1, 8, and 20;
- execution order: 1, 20, 8;
- measured iterations: ten per batch, thirty calls total;
- no discarded warm-up requests;
- requests are strictly sequential;
- timing begins immediately before native HTTP dispatch and ends after the
  complete response body is received;
- JSON parsing, validation, digesting, artifact writes, and journal settlement
  are outside the measured boundary;
- every request uses a new HTTPS connection, a 60-second timeout, disabled
  environment proxies, disabled redirects, and exactly one attempt;
- 429, 529, timeout, malformed response, or uncertain outcome is recorded as a
  partial unavailable result and is never replaced by a retry sample.

The native API response must contain exactly `model`, `answers`, and `usage`.
The resolved model must equal `jev-1.13.0`. Answer keys must equal question keys.
Every answer must be a Choice containing only `type`, `choice`,
`probabilities`, and `confidence`; the selected choice and probability keys
must equal the five frozen labels. Probabilities and confidence must be finite
and within `[0, 1]`; each distribution must sum to 1 within an absolute
tolerance of `1e-6`, and its selected choice must have a maximal probability.
Usage counts must be non-negative integers.

Each reduced sample retains:

- request latency;
- a closed answer map keyed by the opaque question IDs; each value contains the
  selected label, the five-label probability map in frozen label order, and
  confidence;
- selected-choice and probability-distribution digests recomputed from that
  closed answer map;
- accuracy against the synthetic candidate labels;
- mean, minimum, and maximum confidence;
- input and output token counts;
- computed request cost as
  `input_tokens * 0.042 / 1_000_000`, represented as a decimal string;
- exact resolved model ID.

No state text, questions, authorization header, raw body, account ID, request
ID, provider metadata, or host/home path is retained.

## Budget and single-use state machine

The USD 0.25 ceiling is a durable local computed-cost guard, not an
account-global provider limit or billing guarantee. Before creating the output
directory, the runner exclusively creates a claim keyed by the predecessor
Phase 3C artifact-manifest digest under
`<phase3c-artifact-parent>/.phase3d-claims/<digest>.json`. A pre-existing claim
fails closed for network dispatch so one sealed Phase 3C result cannot silently
fund repeated Phase 3D runs.

Before every dispatch, the runner appends and fsyncs a fixed reservation based
on the documented 64,000-token request context limit and a reviewed price
ceiling of USD 0.05 per million input tokens. Each reservation is therefore USD
0.0032 and all thirty reservations could total at most USD 0.096, below the USD
0.25 phase ceiling. A request is refused if its reservation could cross either
limit. After a validated response, the runner computes cost from the returned
input-token count at the published USD 0.042 rate and appends and fsyncs a
settlement before the next request.

Every settlement journal row includes the complete reduced sample needed for
recovery. If reported input use exceeds 64,000 tokens or computed cost exceeds
the reservation, the runner appends a `settled_over_limit` transition with the
known computed use, marks the run uncertain, emits no aggregate, and blocks
another dispatch. An unknown billing outcome, a process restart with an open
reservation, or journal corruption also permanently blocks another dispatch
from that claim.

The append-only journal has a closed schema and monotonically increasing
sequence. Legal event shapes are:

- `reserve`: request ID, sequence, batch size, iteration, predecessor digest,
  and fixed reservation cost;
- `settle`: the matching identity fields, canonical computed cost, and the
  complete reduced sample;
- `settled_over_limit`: the matching identity fields, canonical computed cost,
  complete reduced sample, and the fixed `reported_usage_over_limit` category;
- `failure`: the matching request identity, one category from
  `http_error`, `timeout`, `transport_error`, `response_invalid`, or
  `process_interrupted`, and an HTTP status only when known.

Exactly one `reserve` must precede at most one terminal `settle` or
`settled_over_limit` for the same request. A `failure` may follow an unmatched
reservation but never closes its unknown-cost reservation. Request IDs, batch
sizes, iterations, and sequence must match the frozen 1/20/8 protocol without
gaps or duplicates.

The native API does not expose authoritative billed cost in the documented
response, so every monetary value in this phase is explicitly named
`computed_cost_usd`, never provider-billed cost. Account credit is neither read
nor treated as evidence. Decimal strings use fixed non-exponent notation with
trailing zeroes removed and `0` as the sole zero representation.

## Crash recovery and finalization

The normal run exclusively creates and permission-checks the `0700` output
directory plus an empty, fsynced `0600` `cost-journal.jsonl` before consuming
the irrevocable claim. It then creates and fsyncs the claim before the first
dispatch. If claim creation reports an existing claim, the runner removes only
the empty journal and output directory it created in that invocation and exits
without network use. Result, report, and artifact manifest are written only
after the remote loop finishes or fails cleanly.

The runner exposes `finalize-partial` as a strictly no-network recovery command.
It accepts the same predecessor and packet inputs plus the existing output
directory. It validates the existing claim, predecessor, packet, journal, and
directory allowlist; reconstructs reduced samples from settled journal rows;
and classifies an unmatched reservation without a failure event as
`process_interrupted`. Exactly thirty valid settlements in the frozen protocol
with no open reservation reconstruct a complete result and full aggregates;
every other valid terminal state reconstructs the closed partial result.

Finalization is deterministic and retry-safe. `result.json` and `report.md` are
created independently with atomic create-if-absent; an existing one must match
the exact derived bytes. `artifact-manifest.json` is created last and commits
the final set. A later `finalize-partial` invocation validates an existing
manifest and exits successfully without rewriting it. A crash between files is
recovered by validating and completing the exact missing files. It never
accepts an API key and never resumes dispatch. The claim remains consumed after
either complete or partial finalization.

## Comparison contract

Comparison is limited to shared batch sizes 1, 8, and 20. For each batch the
report presents the underlying values for:

- request and per-item p50/p95 latency;
- aggregate items per second;
- diagnostic accuracy;
- OpenRouter provider-reported cost and native computed cost;
- native-only input/output token use, probability diagnostics, and confidence
  summary, visually separated from cross-lane comparisons;
- local-to-native and local-to-OpenRouter p50 per-item ratios;
- native-to-OpenRouter request-latency ratio.

The report must say that:

- the prior local warm path excludes one-time model load;
- both remote paths include WAN latency and uncontrollable provider state;
- TypeSafe native and OpenRouter use different public revision identifiers, so
  equal Jev-family naming does not prove identical weights or serving stacks;
- native computed cost is not a provider billing receipt;
- confidence describes distribution concentration, not workflow correctness;
- accuracy is an in-sample diagnostic on synthetic accepted rows;
- the synthetic PT-BR workload cannot establish general PT-BR quality;
- no result may be used to choose an automation threshold.

Raw remote p95 remains descriptive over ten samples. No statistical
significance, universal gateway-overhead, calibration, or production-readiness
claim is permitted. The report may describe native probability and confidence
values but must explicitly say that Phase 3C retained no corresponding
OpenRouter distributions, so cross-gateway probability comparison is
unavailable.

## Implementation scope

Allowed:

- `benchmarks/typesafe_native.py` with closed `phase3d.v1` models, transport,
  budget journal, result writer, validator, and report renderer;
- offline unit tests with fake transport, clocks, failures, journals, and
  predecessor artifacts;
- README and Portuguese README instructions using only the generic
  `TYPESAFE_API_KEY` environment contract;
- ignored local artifacts from the authorized live run.

Prohibited:

- changing installed runtime behavior, public inference contracts, weights,
  training data, calibration, or checkpoint selection;
- adding the TypeSafe SDK or any default runtime dependency;
- retry middleware, aliases, remote fallback, telemetry, background uploads,
  arbitrary hosts, or user-selected endpoints;
- committing a private `op://` reference, secret-manager wrapper name, API key,
  request/response text, raw response body, account metadata, or benchmark
  artifacts;
- presenting native TypeSafe as a Saracura runtime backend;
- release publication or production-quality claims.

## Fail-closed behavior

The runner refuses modified or incomplete predecessor artifacts; packet digest
or workload-ID mismatch; a missing `TYPESAFE_API_KEY`; network use without
`--allow-network`; a budget argument other than the reviewed `0.25`; output or
claim reuse; a non-literal host; redirects or proxies; response/model/answer
shape drift; missing or extra labels; invalid probability math; non-finite
confidence/timing; usage drift; reservation overflow; or any unknown file,
link, permission, digest, or summary contradiction in the finished artifact.

Provider failures are reported with a stable category and HTTP status when
known. Response bodies and exception strings that could contain request or
account content are never placed in reports or CLI output.

Provider and artifact JSON parsing rejects duplicate keys. Result, manifest,
and journal JSON use the repository's canonical NFC plus RFC 8785 serializer.

## Artifact contract

The runner creates a new output directory with mode `0700`. It contains only
regular, non-symlink files from this allowlist, each mode `0600`:

- `result.json`;
- `report.md`;
- `cost-journal.jsonl`;
- `artifact-manifest.json`.

`artifact-manifest.json` records the size and SHA-256 of the first three files,
the exact source allowlist and digests, predecessor and packet digests, claim
digest, and its canonical self-digest. Validation rejects extra files, links,
permission drift, digest drift, non-canonical JSON, invalid journal transitions,
or discrepancies between raw samples, summaries, comparisons, and report.

The protocol section records the 64,000-token reservation bound, USD 0.05 per
million reservation rate, USD 0.042 per million computed-cost rate, fixed model,
batch order, iterations, endpoint identity, and single-attempt timeout.

A complete result requires thirty settled samples and no open reservations. An
unavailable partial result may retain only completed-sample digests, resolved
models, computed settled cost, completed-call count, failure category, and
journal state; it emits no per-batch aggregate or three-way comparison. Both
normal error handling and `finalize-partial` must produce this same closed
partial-result shape.

## Acceptance criteria

1. Default install remains offline and lightweight; importing the module needs
   neither TypeSafe SDK nor ML dependencies.
2. Offline tests prove the exact payload transformation, pinned model, host,
   disabled proxy/redirect/retry behavior, response closure, answer-key and
   five-label validation, probability/confidence gates, decimal cost math and
   canonical decimal strings,
   request order/count, latency/throughput math, comparison math, redaction,
   durable reservation/settlement, settled-over-limit behavior,
   crash/restart/concurrency rejection, no-network partial finalization,
   immutable artifact creation, self-digest, permission checks, and tamper
   rejection.
3. `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src
   benchmarks tests`, `uv run pytest -q`, and `uv build` pass.
4. A live run launched with an externally injected `TYPESAFE_API_KEY` completes
   or produces a sealed partial result without exposing the key, private secret
   references, text, raw responses, or account metadata.
5. A complete live result validates independently, contains exact model
   `jev-1.13.0`, thirty calls, fixed batch order/counts, and computed cost not
   exceeding USD 0.25.
6. The public report compares native TypeSafe only against the sealed Phase 3C
   evidence and includes every mandatory limitation.
7. Git-tracked files contain neither credentials nor generated artifacts, and
   the wheel contains neither benchmark code, credentials, data, weights, nor
   artifacts.

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

TYPESAFE_API_KEY=<externally-injected-secret> \
uv run python -m benchmarks.typesafe_native run \
  --phase3c-artifact-dir <sealed-phase3c-artifact-dir> \
  --packet <packet-manifest-or-directory> \
  --output-dir .artifacts/typesafe-native-v1 \
  --allow-network --budget-usd 0.25

uv run python -m benchmarks.typesafe_native finalize-partial \
  --phase3c-artifact-dir <sealed-phase3c-artifact-dir> \
  --packet <packet-manifest-or-directory> \
  --output-dir .artifacts/typesafe-native-v1
```

The command documents only the generic environment contract. Operators inject
the value with their approved local secret manager outside this public repo.
`report.md` is safe for review but is not automatically authorized for public
publication.

## Rollout and rollback

This phase changes checkout-only research tooling and documentation. There is
no deployment, migration, service, or production rollback. Before merge, run
the offline suite and the authorized live benchmark from the isolated worktree.
Rollback is a Git revert of the Phase 3D commit; ignored evidence remains local
and can be archived or removed separately only through an explicit cleanup.

## Next gate

Phase 3D does not authorize a remote runtime integration. Once merged, Phase 4A
may design an opt-in local encoder backend that loads an explicitly verified
checkpoint and derives calibration compatibility from backend metadata. The
synthetic checkpoint remains research-only until the independent human PT-BR
packet, calibration split, and blind test gates are completed.
