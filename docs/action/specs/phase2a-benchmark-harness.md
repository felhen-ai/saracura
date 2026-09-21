---
title: Phase 2A benchmark harness
kind: plan
area: product
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase2a-benchmark-harness.md
globalRef: qmd://saracura/docs/action/specs/phase2a-benchmark-harness.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-21
sourceRefs: []
related:
  - README.md
  - AGENTS.md
  - benchmarks/manifests/ptbr-fixture-v1.json
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 2A benchmark harness

## Context

Saracura Phase 1 provides the closed `v1alpha1` contracts, deterministic self-authored PT-BR fixture backend, encode-once runtime, immutable calibration artifacts, CLI, the immutable `decision-scaling@2026-09-21` workflow, and correctness tests for Q=1, Q=10, and Q=50. It does not yet provide a reproducible benchmark runner, raw-result contract, sanitized environment capture, persisted scaling artifacts, or a report generator. This increment creates those research instruments without adding a learned model, downloading weights, ingesting external data, or making quality claims.

## Current truth

- The default install contains only Pydantic and RFC 8785 runtime dependencies.
- The fixture backend is deterministic test infrastructure with `quality_claims=false`.
- `DecisionEngine.decide` encodes state exactly once per request and can optionally return timing boundaries.
- `known_scaling_questions` and the immutable `decision-scaling@2026-09-21` workflow already define 50 self-authored PT-BR questions; this phase consumes them without changing their schemas.
- The only approved benchmark material is the self-authored PT-BR fixture manifest.
- The initial local reference environment is an Apple Silicon MacBook; the benchmark contract must remain cross-platform and must not require MPS, CUDA, a server, or a private service.

## Goal

Deliver an in-process, reproducible decision-scaling harness that runs the fixture backend at Q=1, Q=10, and Q=50, preserves raw samples and provenance in a closed versioned schema, captures only sanitized environment facts, produces a deterministic summary report, and independently proves that state encoding occurs once per request.

## Non-goals

- Model or tokenizer downloads.
- External datasets, synthetic teachers, training, fitting calibration, or a new calibration status.
- Dynamic labels, boolean or ordinal heads.
- HTTP serving, telemetry, remote fallback, GPU integration, or private adapters.
- Publishing rankings, competitor comparisons, performance claims, packages, releases, or benchmark artifacts.
- Treating fixture timings as evidence of future learned-model latency.

## Public boundary and invariants

- All committed inputs remain self-authored and compatible with the existing fixture manifest.
- Raw results and reports must label the backend as a fixture and explicitly prohibit quality or production interpretation.
- Environment capture must exclude username, hostname, home directory, absolute repository path, serial numbers, hardware UUIDs, IP addresses, tokens, and environment-variable values.
- The runner must use the public backend protocol and `DecisionEngine`; it must not special-case internal fixture scoring beyond fixture construction.
- Every measured request must record exactly one state-encoding call through a harness-owned decorator that implements the public `Backend` protocol and measures the `encode_state` delta around `DecisionEngine.decide`. Any other count fails the run; the harness must not depend on the fixture backend's internal counter.
- Q=1, Q=10, and Q=50 must reuse the same semantic state while varying only the number of questions.
- Question schemas must stay within the known immutable workflow. Calibration artifacts remain `fixture_only` and compatible with each generated question.
- Result files are UTF-8 JSON, canonicalized for identity, written atomically, and never overwritten.
- Timings use the runtime's monotonic high-resolution clock. This phase records neutral `measured` samples after the configured number of warmup iterations, which may be zero; it makes no cold-start or warm-path claim because the fixture backend has no model-load cold path. Raw samples are preserved; for sorted samples of size `n`, percentile `p` is `sorted_samples[ceil(p * n) - 1]`.
- Tests must not assert absolute latency thresholds or depend on machine speed.

## Deliverables

### 1. Benchmark contracts

Add closed Pydantic models for:

- suite and schema version;
- run identity, creation time, code revision, backend/model reference, and sanitized environment; run identity may be a generated UUID, while hardware/device UUIDs are forbidden;
- workload definition including question count, warmup iterations, measured iterations, state hash, and fixture manifest identity as `{id, revision, sha256}`, where SHA-256 covers the manifest's exact committed bytes;
- per-iteration `measured` phase, sample index, state-encoding count, timing boundaries, answer count, and deterministic answer digest;
- summary statistics for total, state encoding, and decision time;
- limitations and research-only status.

Reject non-finite, negative, contradictory, unknown, or privacy-unsafe fields. Benchmark models use `allow_inf_nan=False`. Workloads are exactly Q=1, Q=10, and Q=50 in ascending order. Each sample requires `answer_count == question_count` and `total_ms >= state_encoding_ms + decision_ms`; summary statistics are recomputed and validated from raw samples rather than accepted independently. The raw schema must remain serializable without custom object encoders.

`state_sha256` is SHA-256 over `serialize_state(request)`. `answer_sha256` is SHA-256 over the existing NFC + RFC 8785 canonical bytes of `response.answers` serialized in response order, excluding timing, usage, model metadata, and run metadata.

### 2. Self-authored decision-scaling cases

Create a versioned benchmark-case generator that consumes `known_scaling_questions` and the existing immutable `decision-scaling@2026-09-21` workflow for Q=1, Q=10, and Q=50. Do not create or alter a workflow, question schema, fixture score, or support-routing fixture. Question IDs and calibration IDs must remain stable and unique. All workloads must share byte-identical serialized state. Common question answers must remain identical between Q=1, Q=10, and Q=50.

### 3. Runner

Add `python -m benchmarks.run` with explicit flags for suite, backend, warmups, iterations, output directory, code revision, and optional public hardware label. Only `decision-scaling` and `fixture` are accepted in this phase. Defaults are `warmups=1` and `iterations=10`; accepted bounds are `0 <= warmups <= 100` and `1 <= iterations <= 1000`.

For each workload, create a fresh backend, counting decorator, engine, and compatible fixture calibrations. Execute the configured warmups without recording them, then record every sample with phase `measured` and a zero-based sample index. Do not emit or imply a cold or warm performance regime in this phase. Wrap the selected backend in a harness-owned counting decorator over the public `Backend` protocol and validate an `encode_state` delta of exactly one after every request. Synchronization hooks may exist in the generic runner contract but are no-ops for the fixture backend.

`--code-revision` and `--hardware-label` accept only public identifiers matching `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. They reject whitespace, path separators, colon, at sign, control characters, and all other characters. Report rendering escapes all externally supplied text before inserting it into Markdown.

`code_revision` is an executor-declared identifier recorded for provenance; the harness validates its grammar but does not claim to verify the current Git checkout. Release-grade evidence must supply a commit identifier established by the surrounding release process.

### 4. Sanitized environment capture

Capture an allowlisted subset only: operating-system family and release, machine architecture, Python implementation/version, Saracura version, backend/model revision, and optional validated public hardware label. Do not serialize a raw `platform.uname`, environment dump, filesystem path, hostname, or device identifier.

### 5. Raw results and report

Write one immutable raw JSON result per run and a Markdown report derived solely from that validated result. Both outputs use sibling temporary files, durable flush where supported, and atomic create-if-absent semantics; an existing artifact is never replaced and temporary files are removed after failure. The report has a fixed workload order and deterministic rendering, and must include workload table, p50/p95, sample counts, encode-once evidence, environment disclosure, limitations, and the statement that fixture timing is harness validation rather than learned-model performance. Generated reports are artifacts and do not require repository frontmatter.

### 6. Documentation and CI

Document the local command, result location, schema boundary, percentile method, and interpretation limits in the English and PT-BR READMEs. `python -m benchmarks.run` is explicitly a checkout/development tool and is not included in the current wheel, which packages only `src/saracura`. Add tests for contracts, privacy allowlist, workloads, runner invariants, immutable writes, report output, CLI error mapping, stable common-question answers across Q=1/Q=10/Q=50, and two same-config runs whose deterministic answer digests match. CI remains CPU-only, runs MyPy over `src tests benchmarks`, and must not install ML dependencies.

## Allowed scope

- `benchmarks/**`
- `tests/**` for benchmark coverage
- `README.md`
- `docs/README.pt-BR.md`
- `docs/action/specs/phase2a-benchmark-harness.md`
- `.gitignore`
- `.github/workflows/ci.yml`
- `pyproject.toml` only if needed for package/test discovery or a benchmark script entrypoint
- `uv.lock` only if `pyproject.toml` changes require regeneration
- minimal reusable helpers under `src/saracura/**` only when the benchmark cannot consume an existing public contract safely

## Prohibited scope

- Model, tokenizer, or dataset downloads.
- New runtime dependencies unless unavoidable and separately justified; standard library is preferred.
- Changes to request/response semantics, calibration compatibility axes, fixture scoring, workflow authority, or safety status.
- AIOS, server, VM, Cloudflare, paid compute, publication, package release, or automation integration.

## Acceptance criteria

1. `python -m benchmarks.run --suite decision-scaling --backend fixture --warmups 1 --iterations 3 --code-revision test-revision --output-dir <temp>` succeeds offline as a checkout/development command.
2. The validated raw result contains Q=1, Q=10, and Q=50, exactly three measured samples per workload, and one state encoding per sample.
3. All workloads have the same `sha256(serialize_state(request))`; answer digests follow the frozen canonical definition, and answers for questions shared by Q=1/Q=10/Q=50 are identical.
4. A second run with the same semantic inputs produces the same answer digests while allowing timing and run identity to differ.
5. The raw JSON and report contain no username, hostname, home path, absolute repository path, serial number, hardware/device UUID, IP address, token, or environment-variable value. A generated run UUID is allowed.
6. Existing request, runtime, cache, serialization, calibration, CLI, manifest, and documentation tests remain green.
7. Default `uv sync --locked --dev` downloads no model, tokenizer, dataset, or heavy ML runtime.
8. Ruff, formatting, MyPy, Pytest, manifest validation, and package build pass.

## Validation commands

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.run --suite decision-scaling --backend fixture --warmups 1 --iterations 3 --code-revision local-smoke --hardware-label macbook-m4-pro-24gb --output-dir .artifacts/benchmarks
uv build
```

The smoke output under `.artifacts/` must be ignored by Git and inspected for privacy before reporting success.

## Rollout and rollback

This phase has no service deploy. Integration is a normal pull request into `main`. Rollback is a Git revert of that PR. Generated local artifacts are disposable and remain untracked. No model, dataset, package, release, or remote state is created.

## Risks

- Fixture microbenchmarks can be dominated by timer noise and Python overhead; the report must prevent extrapolation to a learned backend.
- Environment capture can leak identity if implemented as a denylist; use an allowlist model instead.
- Generating 50 questions by accidental object reuse can invalidate isolation tests; use validated immutable models with unique IDs.
- Percentile implementations differ for small samples; freeze nearest-rank semantics and test exact examples.
- Benchmark code can bypass the production runtime for convenience; acceptance requires execution through `DecisionEngine` and the backend protocol.

## Follow-up gate

After this phase, a separate reviewed increment may add optional ML dependencies and an immutable encoder candidate revision. That increment must decide model license, tokenizer/code loading policy, download/cache behavior, Mac CPU/MPS compatibility, and the data protocol before training or quality evaluation begins.
