---
title: Phase 5D native PT-BR decision benchmark
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase5d-native-ptbr-benchmark.md
globalRef: qmd://saracura/docs/action/specs/phase5d-native-ptbr-benchmark.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-28
sourceRefs:
  - https://github.com/tardellirs/mteb-br
  - https://huggingface.co/datasets/MTEB-BR/faq-bacen
  - https://dadosabertos.bcb.gov.br/dataset/perguntas-e-respostas-banco-central-do-brasil
related:
  - docs/action/specs/phase5c-fast-public-preview.md
  - docs/decisions/0002-open-model-runtime-and-readiness.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 5D: native PT-BR decision benchmark

## Outcome

Ship one small, reproducible benchmark that measures a practical four-way
typed decision on native Brazilian Portuguese public data. The first public
result compares the pinned Julia-1 backend with the existing Saracura-owned
ranker when its verified local artifacts are available. It must not present the
Julia-backed Saracura runtime as a new model or reuse Julia's published pt-PT
score as PT-BR evidence.

## Scope

The benchmark uses the FAQ of Banco Central do Brasil already packaged as the
MTEB-BR `FaqBacenRetrieval` task. It contains 373 citizen questions and 1,673
regulatory answers in native PT-BR. The source portal publishes the FAQ under
ODbL and states that its open-data catalog excludes personal or restricted
records. The upstream `Itau-Unibanco/FAQ_BACEN` repository declares
Apache-2.0. Saracura downloads the pinned MTEB-BR derivative at revision
`076d89a68a8b8d2f14e3161631c416ffe29b8463`, does not redistribute its rows,
and preserves attribution to both upstream sources.

Only evaluation-time downloads are allowed. Dataset rows, caches, and generated
request payloads remain untracked.
The committed report contains aggregate metrics and immutable provenance only.

Before acquisition, a create-only `training-data-source-policies.v4` adds one
self-contained Phase 5D external-control exception. It does not add a ninth v1
source, mutate `IDENTITY`/`FIXED_METADATA`, or change the v1-v3 loaders or
manifests. The v4 exception carries its own source metadata and layered rights:
the BCB source is ODbL, `Itau-Unibanco/FAQ_BACEN` declares Apache-2.0, and the
pinned MTEB-BR derivative declares no license in its card. Privacy state is
`publisher_open_data_statement_no_record_review`, not a Saracura record-level
privacy review. The exception is restricted to this aggregate external control,
forbids training/calibration/redistribution, retains only a caller-controlled
cache, and requires aggregate-only output. `validate_routed_manifest` routes v4
explicitly, and compatibility tests prove v1-v3 still validate unchanged.

## Frozen protocol

- Corpus: 1,673 rows, SHA-256
  `378c43e9126a31419680d66c4e43a178a66c8da45a98f64b89af94afb6ecd3e0`.
- Queries: 373 rows, SHA-256
  `5467fc92f2387b436526b609463cdb7f2251fb684667ac9fb4f589aba2e528c0`.
- Qrels: 373 rows, one score-1 relation per query, SHA-256
  `17c173c566e5e021a4ead335717e97927f925df4f82994ccf4a987aeb8a6affc`.
- IDs must be non-empty ASCII without NUL. Queries sort lexicographically by
  their UTF-8 ID bytes (`q0`, `q1`, `q10`, ...). Each row asks the model to
  choose the answer that begins to answer the citizen question correctly. The
  gold answer is paired with three deterministic non-relevant answers.
  Distractors sort by SHA-256 of three UTF-8 segments —
  `saracura-phase5d-faq-bacen-v1`, query ID, corpus ID — each framed by an
  unsigned 64-bit big-endian byte length; the first three with non-empty,
  distinct snippets are selected. Answers are NFC normalized,
  leading/trailing whitespace is stripped, internal whitespace collapses to one
  space, and the first 120 Unicode code points of corpus column `text` form each
  choice description. A candidate snippet must be distinct from the gold
  snippet and all already selected distractors.
- Gold position is `row_index % 4`, producing position counts 94/93/93/93.
  Query text is normalized with the same NFC/whitespace rule but is not
  truncated; `query_sha256` hashes its UTF-8 bytes. The plan is an ordered JSON
  array whose objects contain exactly `query_id`, `query_sha256`,
  `ordered_choice_ids`, `ordered_choice_snippet_sha256`, and `gold_position`.
  RFC 8785 canonical JSON bytes are hashed with SHA-256. The resulting immutable
  plan digest is
  `089a83874ed0cc57da45eb143f4b6f6494125dbfabddb660f8e7ca5105175f6f`.
- Requests use locale `pt-BR`, domain `finance`, one `Choice` question, four
  options, and state key `pergunta`. Julia uses
  `universal-choice@phase5c-julia.v1`; Saracura-owned uses
  `universal-choice@phase4e-saracura-ranker.v1`. Instruction and choice text are
  identical across backends and fit the smaller Saracura criterion envelope.
  The exact instruction is
  `Escolha a alternativa que começa a responder corretamente à pergunta.`
  and option IDs are `a`, `b`, `c`, `d` in request order.
- Input rows are normalized with the existing Saracura request contract. A row
  that exceeds a backend's declared envelope is rejected and counted, never
  truncated silently.
- Primary metrics are `planned_top1_accuracy = correct / 373` and
  `coverage = valid_responses / 373`. Prediction is the highest finite choice
  weight; exact ties choose the earliest option in request order. A valid
  research response counts even though it is correctly marked abstained and
  uncalibrated. The report also records total/correct/incorrect/rejected/error
  counts, accuracy by gold position, cold latency,
  warm p50/p95 latency, throughput, dependency versions, dataset revisions,
  protocol digest, model revision, device, and platform.
- Rejects and errors therefore count as incorrect in the primary score.
  `valid_top1_accuracy` may be reported as a secondary metric with its explicit
  denominator.
- Model outputs remain uncalibrated. Confidence calibration and automation
  claims are forbidden.
- The report always includes chance accuracy `0.25` and a dependency-free
  lexical baseline. Its tokenizer applies NFC plus Unicode `casefold`, extracts
  Unicode alphanumeric word runs excluding underscore, keeps tokens with a
  minimum of four code points (therefore discarding three or fewer), and scores
  each option by the count of unique query
  tokens shared with its 120-code-point snippet. Highest score wins; exact ties
  choose the earliest request option. The frozen plan produces 265/373 =
  `0.710455764075067` for this baseline. Julia and Saracura results must be
  interpreted relative to both chance and this strong lexical baseline.

## Interfaces

Add a checkout-only module runnable as:

```bash
uv run --extra ptbr-benchmark python -m benchmarks.ptbr_native run \
  --backend julia \
  --model-snapshot /absolute/path/to/Julia-1 \
  --device cpu \
  --data-root /absolute/path/to/pinned-files \
  --output /absolute/path/to/report.json
```

The same command accepts `--backend saracura-universal` with the already
required explicit encoder snapshot and training capsule. Paths must be
absolute, exist locally, and never appear in the report.

Reproduction documentation must show Julia's editable installation followed by
`uv run --no-sync`, and must use both `ptbr-benchmark` and `universal-local`
extras for the Saracura-owned backend. The protocol manifest pins lexical
`minimum_token_codepoints` to `4` and expects five query-state capacity rejects
before tokenization for the Saracura-owned request envelope.

The committed manifest is closed JSON and pins source repository, revision,
licenses, split/config, file hashes, selection algorithm, request template, and
the exact cardinalities and plan digest above. The runner
rejects unknown fields, duplicate keys, manifest drift, unexpected columns,
relations, row counts, or plan digest.

The result schema is `phase5d-ptbr-faq-bacen-report.v1`, committed only under
`benchmarks/results/phase5d-ptbr-faq-bacen-<backend>-cpu.json`. Its offline
verifier checks that outcome counts sum to 373, recomputes all metrics and gold
position totals, checks protocol/model/code digests, verifies the fixed chance
and lexical baselines, and rejects raw-text sentinels or absolute paths. The
result records `code_commit`, a release-code digest, manifest digest, model
revision, and checkpoint digest. It aggregates rejection reasons into the
closed categories `request_capacity`, `backend_capacity`,
`backend_unavailable`, `invalid_response`, and `other_error`, and never stores
exception messages. It also records whether a backend-declared choice differs
from the benchmark argmax. Qrels may share a gold corpus ID; the current 373
relations reference 372 distinct answers.

## Non-goals

- Training, fine-tuning, calibration, threshold selection, or model promotion.
- Shipping dataset rows or ML dependencies in the default wheel.
- Reproducing the English Jev playground fixture or a general retrieval
  leaderboard.
- Claiming general superiority from one four-way FAQ task.
- Calling the Julia-backed result a distinct Saracura model.
- Network fallback or automatic model/dataset acquisition during normal
  Saracura inference.

## Acceptance criteria

1. A create-only, self-contained v4 source-policy exception and closed protocol
   manifest pin the
   native PT-BR source, layered licenses, revision, exact file hashes,
   cardinalities, plan digest, selection algorithm, and request templates.
2. The runner reads only the three pinned local files, validates dataset shape,
   relations and plan digest, executes the selected existing backend, and never
   commits or emits raw row content in its aggregate report.
3. Rejects and backend errors are counted against planned accuracy; the run
   cannot silently skip difficult or oversized examples.
4. Tests cover v1-v3 registry compatibility, duplicate/unknown manifest fields,
   file/cardinality/relation/plan drift, deterministic distractor selection and
   balanced positions, report sanitization, error accounting, tie handling, and
   deterministic metric calculation without network or heavyweight ML imports.
   Pure row/plan/report logic is tested with in-memory records; Parquet I/O is
   isolated behind the `ptbr-benchmark` extra containing `pyarrow`.
5. The optional extra is isolated from the default install, and wheel/sdist
   inspection confirms that datasets, weights, caches, and reports with raw
   rows are absent.
6. A real CPU run against the pinned Julia-1 snapshot produces a committed
   aggregate report. If the Saracura-owned artifacts remain locally available,
   a second result is produced under the same protocol and explicitly described
   as also measuring its smaller state/context capacity; otherwise that absence
   is reported explicitly and does not block the Julia result.
7. English and Portuguese documentation explain exactly what is measured,
   provide reproduction commands, layered-license attribution, chance and
   lexical baselines, and state all limitations without a superiority or
   production-readiness claim. They disclose unknown Julia training overlap and
   the unchanged `synthetic_only_research` disposition of the Saracura-owned
   ranker.

## Validation

```bash
uv run ruff check benchmarks/ptbr_native benchmarks/data_policy_registry.py \
  benchmarks/validate_manifests.py tests/test_ptbr_native.py tests/test_data_policy_gate.py
uv run ruff format --check benchmarks/ptbr_native benchmarks/data_policy_registry.py \
  benchmarks/validate_manifests.py tests/test_ptbr_native.py tests/test_data_policy_gate.py
uv run mypy benchmarks/ptbr_native benchmarks/data_policy_registry.py \
  benchmarks/validate_manifests.py tests/test_ptbr_native.py tests/test_data_policy_gate.py
uv run pytest -q tests/test_ptbr_native.py tests/test_data_policy_gate.py
uv run pytest -q
uv build
```

The live run is a release evidence step, not a unit test. Its output must pass a
separate offline verifier before it is committed.

## Rollout and rollback

Land through one PR. This phase changes no production service, default runtime,
or model registry. Rollback is a revert of the benchmark module, manifest,
aggregate results, optional dependency group, and documentation.
