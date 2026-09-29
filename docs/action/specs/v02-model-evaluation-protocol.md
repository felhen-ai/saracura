---
title: v0.2 Model Preview evaluation protocol
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/v02-model-evaluation-protocol.md
globalRef: qmd://saracura/docs/action/specs/v02-model-evaluation-protocol.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-29
sourceRefs:
  - github:#44
related:
  - docs/decisions/0004-model-first-product-direction.md
  - docs/action/specs/phase5d-native-ptbr-benchmark.md
  - benchmarks/manifests/phase5d-ptbr-native.v1.json
  - benchmarks/manifests/v02-model-evaluation-protocol.v1.json
supersedes: []
supersededBy: []
sensitivity: public
---
# v0.2 Model Preview evaluation protocol

## Outcome

Freeze the smallest reproducible protocol that can select the first
Saracura-owned universal checkpoint and support one bounded public claim. The
protocol reuses existing benchmark machinery and adds only the missing
train/development/held-out boundary and aggregate-result contract.

This protocol closes GitHub issue #44. It does not train a checkpoint, create a
held-out corpus, publish a comparative result, calibrate scores, or authorize
automation.

## Decision contract

The v0.2 evaluation has three disjoint evidence lanes:

1. `training`: one immutable training capsule selected by #45. It may be read
   only by candidate training and development selection. Its record identities
   and normalized-content fingerprints must be disjoint from both evaluation
   lanes.
2. `public_dev`: the existing pinned native PT-BR FAQ benchmark. It contains
   373 planned four-way Choice decisions and is development evidence only. It
   may select architecture, hyperparameters, prompt/readout strategy, and the
   release candidate.
3. `sealed_ptbr_test`: an external, immutable capsule of at least 100
   independently adjudicated PT-BR Choice decisions. Before any candidate is
   selected, its descriptor commits the payload digest, identity-set digest,
   state-question-fingerprint-set digest, combined-content-fingerprint-set
   digest, descriptive option-multiset-fingerprint-set digest, slice counts,
   option-count distribution,
   and gold-position distribution. Training and development commands may read
   only that descriptor, never the payload. The payload is opened once for the
   selected candidate and frozen controls after the selection receipt exists.

The test lane may contain public-source derivatives or original contributed
cases, but its descriptor must state the source class and contamination risk.
It cannot be described as hidden from a foundation model merely because it was
hidden from Saracura's training process.

Record identities are opaque UTF-8 strings. All content fingerprints use
SHA-256 over RFC 8785 canonical bytes after recursively applying Unicode NFC,
`casefold`, leading/trailing strip, and internal-whitespace collapse to every
string value. `state_question_fingerprint` covers the identity-free state and
question/instruction. `option_multiset_fingerprint` covers normalized option
descriptions without option IDs, sorted by their canonical bytes. A
`combined_content_fingerprint` covers the normalized state/question together
with that order-invariant option multiset, so option reordering and ID changes
cannot hide reuse. Before selection, a sealing tool that can read all three
lanes emits a closed disjointness receipt binding each lane's identity,
state-question, combined-content, and option-multiset set digests. The receipt
contains the three explicit pairwise counts (`training_public_dev`,
`training_sealed_test`, and `public_dev_sealed_test`) for each fingerprint
kind; every identity, state-question, and combined-content count is zero.
Option-multiset intersections are descriptive only because generic sets such
as `sim`/`não` may legitimately recur across unrelated decisions. This detects
exact normalized reuse but does not claim semantic near-duplicate detection.
Later validators verify the receipt, never the raw sets.

The held-out payload is permanently ineligible for training, retrieval,
prompt/readout selection, calibration fitting, threshold selection, or the
post-v0.2 work in #50. Calibration requires its own disjoint fit and evaluation
lanes.

## Typed task contract

- API: `v1alpha1`, research mode, local execution, `Choice` only.
- Locale: `pt-BR` for the v0.2 claim.
- Options: 2 through 8, with every supported cardinality represented in the
  sealed test capsule.
- Prediction: the highest finite ranking weight; exact ties choose the earliest
  option in request order and are counted in option-order diagnostics.
- Capacity failures, invalid outputs, and execution errors are never silently
  skipped. They reduce planned accuracy and coverage.
- Outputs remain uncalibrated and `automation_allowed=false`.

## Frozen development lane

`public_dev` is exactly the protocol in
`benchmarks/manifests/phase5d-ptbr-native.v1.json`, whose canonical SHA-256 is
`af983ce181018a2f6c5abec351ce1b78e875764616875aed3f59aa416a499ff6`
over the raw file bytes.
Its 373 records, request framing, lexical baseline, gold-position balance,
normalization, rejection accounting, and aggregate-only output remain
unchanged. It is not renamed or presented as held-out evidence.

## Frozen controls

The same request content, candidate-independent preprocessing, denominator,
host hardware class, and quality-scoring procedure apply to the release
candidate and model controls. Runtime, device, precision, batch, and
concurrency are declared per result row because the pinned controls do not
share one executable device path.

- `laya-multilingual`: pinned local option-marker encoder at model revision
  `052592a15d198d9ad47da779604259b10b47b7aa`.
- `julia-1-cyclic-mean`: pinned Julia-1 checkpoint revision
  `a85b127321d580d65176c89ced8273f305745d85`, using the already frozen
  position-invariant cyclic-mean strategy.
- `kev-4b`: pinned Kev adapter revision
  `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` over
  `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`.
- `lexical-baseline`: the dependency-free lexical baseline already frozen by
  the development protocol. It is a public-development sanity check only and
  is not adapted post hoc to the generic held-out lane.

Jev, remote APIs, models above 5B parameters, and a general leaderboard are not
same-hardware v0.2 controls. They may remain external context but cannot define
the release claim.

`kev-4b` is the pre-registered primary quality and CUDA systems reference for
the first Saracura candidate because it shares the Qwen 3.5 4B deployment
class. Laya and Julia are descriptive quality controls. Systems metrics are
comparable only inside an explicit `systems_comparability_group` whose members
share host class, runtime boundary, device class, precision, batch, concurrency,
and timing procedure. No latency, throughput, or memory claim may cross groups.
Every control row carries a required training-overlap/contamination disclosure;
`unknown` is valid evidence and can never be rewritten as `none`. Service or
HTTP transport overhead is part of the declared runtime boundary.

The comparable v0.2 lane has a frozen maximum of 512 rendered input tokens
under the pinned Kev tokenizer and adapter rendering function, measured with
truncation disabled. The protocol manifest binds the adapter revision and the
rendering-function digest
`eca2a60af37c539c984e89cf920c53e8d1c93cff6e980dea1a24dd520f86e169`.
The 384/1024/2048 encoder limits and the 512-token ceiling are the joint
admission contract. They do not describe Kev's full standalone serving context.
The sealed descriptor proves every held-out record fits that envelope. The
benchmark client repeats the preflight before every Kev call. Every model row
records `input_truncation_detected`; the valid protocol requires that counter to
remain zero for every model. A capacity rejection remains in the planned
denominator, and a response from a request that did not pass the applicable
model preflight is never scored as valid even if an upstream service returns
HTTP 200 after truncation.

## Metrics and slices

The primary quality metric is `planned_top1_accuracy`: correct predictions
divided by every planned record. Required companion metrics are coverage,
valid-only top-1 accuracy, macro-F1 over option positions, invalid-output rate,
error rate, and accuracy by domain, option count, state-length bucket, and gold
position.

Required systems metrics are cold-load time, warm request p50/p95, decisions
per second, peak host RSS, peak device memory when applicable, and artifact
bytes. Every result identifies the hardware class, device, numeric precision,
batch size, concurrency, code revision, model revision, tokenizer revision,
and protocol digest.

Option-order sensitivity is measured on a deterministic paired permutation
subset of at least 50 records committed by the sealed descriptor, including
its explicit UTF-8 selection seed. Records are ordered by SHA-256 of
length-framed seed and opaque identity, and the
lowest hashes are selected. The paired permutation is one cyclic left rotation
of the declared option order. A prediction is stable only when
the selected semantic option remains unchanged after remapping positions.
Deterministic repeat stability uses two identical executions of that subset.

The sealed lane contains at least 14 records for each option count from 2
through 8. Gold positions are balanced within each option count with maximum
count difference one. Every record is independently labeled by two annotators;
disagreements require a third adjudicator. Slice results remain descriptive
unless their denominator is at least 30.

The public development lane exercises only four-option questions. The aggregate
report must preserve that limitation; development evidence cannot be claimed
for the other held-out cardinalities.

Calibration metrics, risk-coverage curves, confidence thresholds, and
automation gates belong to #50 and are forbidden from the v0.2 release claim.

## Candidate selection and one-time test

1. #44 publishes this protocol and its closed manifest.
2. #45 seals and commits the closed test descriptor plus disjointness receipt
   before comparing training candidates.
3. #45 trains a bounded candidate set and selects exactly one candidate from
   training plus `public_dev` evidence.
4. The selection receipt binds the candidate, training capsule, code,
   hyperparameters, seed, development report, and sealed-test descriptor.
5. Only then may the test runner open the committed payload and run the selected
   candidate plus the frozen controls.
6. A failed held-out result is terminal for that selected checkpoint. A new
   candidate requires a new protocol revision and a new disjoint sealed test
   capsule; the existing test cannot become development data for v0.2.

The descriptor, disjointness receipt, and selection receipt have closed v1
schemas frozen by the protocol manifest, including closed nested descriptor,
receipt, selection, available-row, and unavailable-row forms. The selection receipt records both
the descriptor commit and its own selection commit; offline validation requires
the descriptor commit to be an ancestor of the selection commit in the full
`felhen-ai/saracura` Git history and verifies with Git object inspection that
the descriptor digest is present in that commit's tree at the receipt's
repository-relative descriptor path. A shallow checkout, missing commit,
absolute path, path traversal, or blob mismatch fails closed. The selection
receipt also records the candidate deployment class; a
candidate outside the Qwen 3.5 4B pointer-head class may still be evaluated but
cannot support the pre-registered Kev comparative claim. The aggregate
report binds all three receipt digests and must identify the exact selected
checkpoint, tokenizer, and code revisions from the selection receipt.

## Aggregate result contract

The checkout-only validator accepts only the closed
`saracura-v02-evaluation-report.v1` schema. The aggregate report contains
digests, counts, paired comparison counters, slice aggregates, systems measurements, limitations,
and an optional bounded claim. It rejects raw record text, record identifiers,
absolute paths, exception messages, calibration fields, and automation
authorization.

The validator recomputes ratios from integer counts, requires results for the
selected Saracura candidate and all available frozen controls, verifies that
all rows share the same planned denominator and quality-scoring settings, and checks
that the report binds the canonical protocol bytes and sealed descriptor.
Unavailable controls remain explicit with a closed reason and cannot silently
disappear from the comparison. If Kev-4B is unavailable, the report cannot make
a comparative quality or systems claim. Laya or Julia unavailability prevents
wording about the complete frozen control set.

The held-out report has model rows only for Saracura, Kev, Laya, and Julia.
The lexical baseline is recorded separately as `public_dev_only` and is never
represented as an unavailable held-out control.

The public claim is optional and its reference is pre-registered as Kev-4B.
When present, it must name the PT-BR task, both model revisions, hardware,
metric, denominator, and exact winning margin. A quality claim additionally
requires a one-sided exact McNemar test over paired held-out predictions with
`p < 0.05`, candidate accuracy above Kev-4B, and aggregate discordant counters
`candidate_correct_control_wrong` and `candidate_wrong_control_correct`.
The validator requires their difference to equal the difference between
candidate and Kev correct counts and their sum to be no greater than the
planned denominator.
A systems claim requires both rows to share one systems comparability group.
No generic “best sub-5B” claim is permitted. If these gates do not pass, the
report contains no comparative claim.

## Interfaces

The canonical offline commands are:

```bash
uv run python -m benchmarks.v02_evaluation validate-protocol
uv run python -m benchmarks.v02_evaluation validate-report \
  --report /absolute/path/to/aggregate-report.json \
  --sealed-descriptor /absolute/path/to/sealed-test-descriptor.json \
  --disjointness-receipt /absolute/path/to/disjointness-receipt.json \
  --selection-receipt /absolute/path/to/selection-receipt.json
```

Both commands are network-free and import no heavyweight ML dependency. The
second command validates aggregates only and never opens the sealed payload.

## Non-goals

- Building another benchmark framework or general leaderboard.
- Acquiring data, calling a teacher, training a model, or opening held-out rows.
- Selecting thresholds, calibrating confidence, or authorizing automation.
- Claiming multilingual quality, production readiness, or general superiority.
- Making any third-party model the Saracura product identity.

## Acceptance criteria

1. A closed manifest freezes the three lanes, task subset, controls, metrics,
   execution equality rules, descriptor/receipt/report schemas, and prohibited
   claims.
2. Offline validation rejects unknown fields, duplicate JSON keys, manifest
   drift, nonzero overlap receipts, post-selection descriptor commits, raw
   content, inconsistent metrics,
   missing controls, unequal denominators, and calibration or automation data.
3. The development lane remains bound byte-for-byte to the existing native
   PT-BR manifest and is never described as held-out.
4. Unit tests exercise valid aggregate evidence and each fail-closed boundary
   without network, model weights, datasets, or optional ML dependencies.
5. Documentation states that cloud training hardware may differ from benchmark
   hardware; quality comparisons share the host class, while systems claims
   require an identical systems comparability group.

## Validation

```bash
uv run ruff check benchmarks/v02_evaluation.py tests/test_v02_evaluation.py
uv run ruff format --check benchmarks/v02_evaluation.py tests/test_v02_evaluation.py
uv run mypy benchmarks/v02_evaluation.py tests/test_v02_evaluation.py
uv run pytest -q tests/test_v02_evaluation.py tests/test_manifests.py
uv run python -m benchmarks.v02_evaluation validate-protocol
uv run python -m benchmarks.validate_manifests
uv run pytest -q
uv build
```

## Rollout and rollback

Land through one pull request linked to #44. This phase changes no runtime,
model, dataset, provider, production service, or public quality claim. Rollback
is a Git revert of the spec, manifest, validator, tests, and manifest routing.
