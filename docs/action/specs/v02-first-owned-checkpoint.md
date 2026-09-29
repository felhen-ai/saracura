---
title: v0.2 first Saracura-owned checkpoint
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/v02-first-owned-checkpoint.md
globalRef: qmd://saracura/docs/action/specs/v02-first-owned-checkpoint.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-29
sourceRefs:
  - github:#45
  - github:#53
related:
  - docs/decisions/0004-model-first-product-direction.md
  - docs/action/specs/v02-model-evaluation-protocol.md
  - benchmarks/manifests/v02-model-evaluation-protocol.v1.json
  - benchmarks/manifests/phase5-open-model-candidates.v3.json
  - benchmarks/manifests/phase4e-saracura-universal-training-authorization.v1.json
supersedes: []
supersededBy: []
sensitivity: public
---
# v0.2 first Saracura-owned checkpoint

## Outcome

Produce exactly one immutable, Saracura-owned adapter-and-head checkpoint for
the pinned Qwen 3.5 4B base, selected without reading the sealed PT-BR test and
evaluated once under the protocol frozen by #44. The checkpoint is the input to
#46 packaging and #47 publication of comparative evidence. This increment does
not publish a model release, change the runtime default, or make a public
superiority claim.

The execution is intentionally narrow. It is an adaptation run, not an
architecture search, dataset program, new benchmark, provider integration, or
calibration project.

## Frozen decisions

### Model and ownership boundary

- Base model: `Qwen/Qwen3.5-4B-Base` at immutable revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Architecture class: `qwen35_4b_pointer_head`, matching the deployment class
  pre-registered by #44 and the reviewed Kev control.
- Saracura-owned artifact: LoRA adapter tensors, pointer/readout-head tensors,
  closed configuration, tokenizer/base references, licenses, provenance, and
  digests. Do not copy or republish base-model tensors.
- Training precision: QLoRA/NF4 with BF16 compute on one NVIDIA CUDA device
  with at least 24 GiB device memory. Training may use any recorded compatible
  cloud NVIDIA device because training hardware is not a systems claim.
- Held-out quality and systems host class: one dedicated x86_64 Linux host with
  exactly one NVIDIA RTX 4090 24 GiB, candidate and Kev both using CUDA BF16,
  unmerged LoRA scale 1, batch 1, concurrency 1, the same pinned base bytes,
  the same tokenizer bytes, and the same timing procedure. Their
  `systems_comparability_group` is
  `linux-x86_64-rtx4090-cuda-bf16-b1-c1-qwen35-4b-v1`. Laya and Julia remain
  descriptive quality controls outside this systems group. #47 publishes the
  already validated #45 aggregate and does not rerun or reopen the payload.
- Candidate and Kev share runtime boundary
  `loopback_http_python_worker_v1`: one fresh Python worker per model on the
  same host, client requests over `http://127.0.0.1`, no proxy, batch 1 and
  concurrency 1. Timing procedure
  `saracura-v02-loopback-single-request-v1` measures client-side monotonic time
  around HTTP serialization, request, response and schema validation; records
  cold process-start-to-first-valid-response; excludes 10 warmups; then runs
  100 sequential measured requests, nearest-rank p50/p95, total decisions per
  measured wall time, peak worker RSS and peak device memory. Service overhead
  is deliberately included for both. Inside the one-time held-out session,
  both systems use the same descriptor-defined deterministic 50-record
  permutation subset: warmups and measured requests cycle through that ordered
  subset identically, without another payload opening. Both workers set the
  same determinism flags, including `CUBLAS_WORKSPACE_CONFIG`, before startup.
- Input envelope: at most 512 candidate-rendered tokens under the pinned Qwen
  tokenizer, with truncation disabled. Candidate rendering imports the
  Apache-2.0 `user_tokens`, `encode`, `training_context` behavior and delimiter
  constants from `kev/model.py` at source revision
  `9c41005b2180347c3c646dfc9e50c4428483ec6b`, with `max_state=384`,
  `max_branch=1024`, `max_packed=2048`, `strict=true`,
  `option_isolation=false`, `special_embeddings=false`, `lora_targets=all`,
  and `head_dim=256`. The public training manifest stores the raw SHA-256 of
  that source file and a `candidate_rendering_digest` computed as SHA-256 over
  RFC 8785 canonical bytes of the source revision/path/hash, callable names,
  constants and every parameter above. It does not reinterpret the unrelated
  #44 Kev rendering-contract digest as a source-file digest. The private sealer
  applies both the #44 Kev preflight and this candidate preflight to every
  sealed record and records the candidate digest plus a true all-records-pass
  boolean in the #45 readiness descriptor. No Kev checkpoint tensor or
  trainable special embedding is copied. Records outside the envelope fail
  readiness before candidate scoring.
- API task: local `v1alpha1` research-mode `Choice`, 2 through 8 options,
  `pt-BR` claim lane. Outputs remain uncalibrated and
  `automation_allowed=false`.

Changing the base family, base revision, architecture class, context envelope,
task type, or ownership boundary is a design change and requires a successor
spec. It is not an executor choice.

### Data boundary

The private Phase 4E packet is useful prior evidence but is not automatically
authorized for this release. Its current grant explicitly sets
`publication_authorized=false`, `canonical_training_authorized=false`, and
`quality_claims_allowed=false`. No executor may reinterpret that grant.

Before data upload or GPU rental, the prerequisite data-readiness issue must
create and validate one closed `v02-training-capsule-authorization.v1`
successor that binds:

- the exact Phase 4E accepted-packet manifest digest
  `3e5dccc8bb551cf4840046b20712a52c9409c9424f20333185f3072e8dd6c239`;
- the exact accepted training and development identity/fingerprint digests;
- the author and reviewer model identities and provider route already recorded
  by the packet;
- a dated provider-output rights review and the Apache-2.0 Saracura artifact
  license;
- explicit authorization for processing those rows on the named cloud GPU
  provider, including retention/deletion terms and region, or an explicit
  requirement for a zero-retention encrypted worker;
- authorization to train and publish adapter/head weights only;
- explicit prohibition on publishing raw rows, prompts, provider payloads,
  private paths, the historical synthetic holdout, or base-model weights;
- explicit continued prohibition on calibration and automation claims.

The successor authorization may clarify the permitted use of the already
sealed packet; it may not mutate historical manifests, invent missing lineage,
or silently choose a different corpus. If the rights and provenance evidence
cannot support adapter/head publication, the Phase A evidence receipt returns
`BLOCKED_DATA_RIGHTS` and the entire issue stops before GPU spend. Selecting or
generating another training corpus is outside #45.

The historical Phase 4E synthetic holdout is consumed research evidence and is
not the v0.2 sealed test. It must not be opened, copied, relabeled, or used for
candidate selection in this increment.

### Evaluation boundary

- Development lane: exactly `benchmarks/manifests/phase5d-ptbr-native.v1.json`
  at canonical raw-file SHA-256
  `af983ce181018a2f6c5abec351ce1b78e875764616875aed3f59aa416a499ff6`.
- Evaluation protocol: exactly
  `benchmarks/manifests/v02-model-evaluation-protocol.v1.json` at the bytes in
  the descriptor commit.
- Sealed test: an external private capsule satisfying #44, with at least 100
  independently adjudicated PT-BR Choice records, 2 through 8 options, frozen
  descriptor, option-order subset, and disjointness receipt.
- The sealed descriptor and disjointness receipt must be merged to `main` with
  preserved commit ancestry before any candidate development score is
  computed. The prerequisite data-readiness issue owns creation/adjudication of
  the capsule; #45 only validates and consumes its descriptor and receipt.
- Candidate training and selection processes receive the descriptor and
  receipt only. They must not receive the sealed payload path, bytes, mount, or
  credentials.
- After candidate selection is committed, the held-out runner may open the
  payload exactly once for the selected candidate and the controls required by
  #44. A failed held-out result is terminal for this protocol revision.

Creation of genuinely new sealed-test content is not delegated to the model
trainer. If no compliant sealed capsule already exists, the Phase A evidence
receipt returns `BLOCKED_SEALED_TEST` before GPU spend. The executor must not
generate, self-label, or weaken the test lane merely to complete the issue.

The prerequisite data-readiness issue owns new sealed-payload creation,
independent annotation/adjudication and its private provenance receipt. #45's
Phase C owns the only three-lane sealing step because it alone can read the
authorized training lane. The sealer executes under a separate data-operator
process; the later trainer receives descriptors and the training payload, but
never the sealed payload path, mount or credential.

For #44 disjointness, the `training` lane contains every Phase 4E row used by
the optimizer **or** internal early stopping, including its historical train
and dev splits. Before any candidate score, the public sealer implemented in
Phase B runs in a private data environment with read access to the training
lane, public development lane, and sealed lane. It applies exactly #44's NFC,
casefold, whitespace-collapse, RFC 8785 and length-framing rules; emits the
closed training descriptor and zero-overlap disjointness receipt; and discards
all raw fingerprint sets after their set digests and pairwise counts are
verified. Raw rows and fingerprint sets never enter Git. The #45 manifest
defines a closed `v02-training-descriptor.v1` containing record count, lane
split counts, normalized identity/state-question/combined-content/
option-multiset set digests, accepted-packet digest, successor-grant digest,
candidate-rendering digest, and source-policy identifiers. It also defines a
closed `v02-readiness-descriptor.v1` binding that training descriptor, the
#44 sealed descriptor, the #44 disjointness receipt, both renderer digests,
and true all-records preflight booleans. These companion schemas do not alter
the frozen #44 schemas.

## Candidate grid

Train exactly three candidates. All use the same rendered examples, optimizer
family, effective batch size, maximum sequence length, seed, evaluation code,
and stopping rule. Only the declared LoRA capacity and learning rate vary:

| ID | LoRA rank | LoRA alpha | Dropout | Peak learning rate |
| --- | ---: | ---: | ---: | ---: |
| `c1-r8` | 8 | 16 | 0.05 | `2e-4` |
| `c2-r16` | 16 | 32 | 0.05 | `1e-4` |
| `c3-r32` | 32 | 64 | 0.05 | `5e-5` |

Shared settings:

- seed `20260929` for Python, NumPy, and Torch;
- AdamW, weight decay `0.01`, linear warmup over 5% of optimizer steps and
  cosine decay;
- effective batch size 32 through fixed microbatch plus gradient accumulation
  chosen only from device capacity;
- at most 5 epochs; after each epoch evaluate only the Phase 4E internal dev
  rows, using `stratified_macro_accuracy.v1`; early stopping patience 2;
- export the epoch with highest internal-dev stratified macro accuracy, breaking
  ties in favor of the earliest epoch; `public_dev` is evaluated only after the
  exported epoch is frozen and never drives early stopping;
- gradient clipping at 1.0;
- base weights frozen; train only declared LoRA modules and pointer/readout
  head;
- no prompt variants, no alternate renderer, no post-result hyperparameter
  additions, no extra seed, and no fourth candidate.

Target-module names and the pointer-head tensor contract must be closed in the
training manifest during Phase B, reviewed before the Phase C provenance PR is
merged, and must match the pinned base revision. The manifest proves that no
Kev adapter, head, or other third-party checkpoint tensor is copied.
An incompatibility is an implementation failure, not permission to swap the
base or architecture.

## Selection rule

Candidate selection uses training evidence plus the frozen public development
lane only. Every `public_dev` score, repeat/order diagnostic and latency
tie-break runs on the same RTX 4090 host, BF16 unmerged-LoRA artifact,
loopback-HTTP worker boundary, batch, concurrency, renderer and timing procedure
frozen for held-out execution. QLoRA/NF4 is a training representation only; it
is never the evaluated candidate artifact. Selection is deterministic:

1. Disqualify a candidate with any invalid output, truncation, non-finite
   weight, identity mismatch, or deterministic-repeat mismatch.
2. Rank remaining candidates by `planned_top1_accuracy` on `public_dev`.
3. Break an exact tie by higher option-order semantic stability.
4. Then break a tie by lower warm p95 latency in the same declared development
   environment.
5. Then choose lexicographically smallest candidate ID.

The candidate is eligible for held-out evaluation only if it has:

- coverage `>= 0.98`;
- invalid-output rate `= 0`;
- deterministic repeat stability `= 1.0` on the frozen public-development
  diagnostic subset;
- option-order semantic stability `>= 0.95` on that subset; and
- public-development `planned_top1_accuracy >= 0.40` (chance is 0.25).

The public-development diagnostic subset contains exactly 50 valid records.
Order records by SHA-256 over unsigned 8-byte big-endian length-framed UTF-8
segments `("saracura-v02-public-dev-order-v1", opaque_record_id)` and take the
lowest hashes. Run two identical batch-1 inference passes plus one cyclic left
rotation of each record's options. Stability is semantic after mapping rotated
positions back to the original option identity. CUDA execution enables
`torch.use_deterministic_algorithms(True)`, disables cuDNN benchmarking, records
all determinism environment flags, and fails rather than silently selecting a
nondeterministic kernel.

The lexical baseline and frozen external controls remain reported context; #45
does not lower its eligibility gate or add candidates after seeing their
scores. If none is eligible, emit `NO_RELEASE_CANDIDATE`, preserve all reports,
do not open the sealed payload, and close #45 without a checkpoint selection.
Every development aggregate records that the training capsule contains both
PT-BR and English synthetic rows, while `public_dev` contains only four-option
PT-BR FAQ answer selection; no broader language or cardinality claim follows.

## Phases and post-conditions

### Phase A — prerequisite evidence readiness

No GPU, model download, provider call, dataset generation, or sealed payload
access.

Post-condition:

- exact base/source/license identities verified;
- publication-capable training authorization validated;
- a compliant sealed payload and its independent adjudication evidence exist
  outside #45, without exposing payload bytes to the trainer;
- private artifact destinations are outside Git and synchronized folders;
- disk, secret, and cloud credential preflights report booleans/digests only;
- a private, manually reviewed Phase A evidence receipt, distinct from the
  public Phase C readiness descriptor, returns `READY`,
  `BLOCKED_DATA_RIGHTS`, or
  `BLOCKED_SEALED_TEST`.

Any blocked result terminates #45 without continuing to implementation or GPU.

### Phase B — offline implementation

Implement only the smallest training, selection, export, and receipt machinery
needed by this spec. Required public additions are:

- `benchmarks/v02_training.py`;
- `benchmarks/manifests/v02-first-checkpoint.v1.json`;
- `tests/test_v02_training.py`;
- manifest routing/validation additions;
- this spec and bounded documentation links.

`pyproject.toml` and `uv.lock` may change only to add an isolated optional
training extra. `benchmarks/v02_evaluation.py` adds a bounded
`validate-selection` subcommand that reuses its existing descriptor,
disjointness-receipt and `_validate_selection` logic before held-out access.
It may not duplicate that proof in `v02_training`, change schemas, or alter
protocol semantics and thresholds.

Post-condition: unit tests prove closed schemas, deterministic selection,
create-only artifacts, no held-out access before selection, exact candidate
grid, source-history bindings, resume behavior, and all fail-closed paths using
fixtures only. Ordinary installation and CI download no model or dataset.

Phase B ends with independent review, a green implementation PR, and merge to
`main` before any private sealing or GPU work. All later commands run from a
fresh full clone of that `origin/main`. The exact clean main commit used by the
sealer and trainer initially becomes `code_revision`; later evidence-only
commits do not rewrite it.

After Phase B, a code correction is allowed only through an independently
reviewed PR merged to `main`. The new clean reachable main commit becomes the
single replacement `code_revision` for every later manifest, report, receipt,
and candidate artifact. Every correction invalidates the earlier evidence
boundary and reruns Phases C, D, and E from scratch under new create-only
descriptor, receipt, smoke, and candidate artifact identities; the old
evidence remains immutable and ineligible. No artifact from different
`code_revision` values may be combined. After `selection_commit`,
`NO_RELEASE_CANDIDATE`, or any held-out payload access, code correction is
forbidden: the run ends terminally and any retry requires a successor protocol.

### Phase C — seal and merge the evidence boundary

Run `benchmarks.v02_training seal-readiness` in the private data-operator
environment. It reads the complete
Phase 4E train **and internal-dev** lane, frozen `public_dev`, and externally
prepared sealed lane; emits the #45 training and readiness descriptors plus the
#44 sealed descriptor and disjointness receipt; verifies all required pairwise identity,
state-question, and combined-content overlap counts are zero; and discards raw
fingerprint sets.

Commit only aggregate descriptors and receipts. Merge them to `main` through a
dedicated provenance PR using a merge commit, never squash or rebase. Record
the resulting reachable `descriptor_commit`. Candidate scoring is forbidden
until a fresh full clone from `origin/main` validates descriptor bytes and Git
ancestry.

The data-operator environment may download only the tokenizer and configuration
at the pinned revision needed for deterministic rendering; it must not download
or mount base-model weights.

Then run `benchmarks.v02_training validate-readiness` from a fresh full clone
using only the training descriptor, sealed descriptor, readiness descriptor
and disjointness receipt. Post-condition: the exact evidence boundary is
reachable from public `main` before any candidate score or GPU smoke.

### Phase D — GPU smoke

Use 16 training records and 16 development records, one optimizer step per
candidate, with the real pinned base and production code path. The smoke may
test memory, serialization, reload, deterministic inference, and receipt
generation; it is not quality evidence and cannot select a candidate.

Post-condition: all three candidate configurations train, save, reload, and
produce valid typed Choice outputs. Failure stops before full training.

### Phase E — bounded candidate training and development selection

Train exactly the three frozen candidates once each. Do not retry a completed
candidate under changed settings. Operational interruption may resume only
from a checkpoint whose manifest, source revision, data digest, environment,
seed, and optimizer state match exactly.

After all three complete, run the frozen development protocol, apply the
selection rule, export create-only candidate reports, and commit an aggregate
development report. If one candidate is eligible, create and commit the closed
selection receipt binding the candidate ID, adapter/head digest, base and
tokenizer revisions, training capsule, code revision, configuration, seed,
development report, protocol, sealed descriptor, and descriptor commit.

The closed #44 selection schema maps these bindings as follows:

- `candidate.checkpoint_revision` is the 64-hex SHA-256 of the exported
  adapter/head artifact;
- `candidate.tokenizer_revision` is the pinned 40-hex base/tokenizer revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`;
- top-level `training_capsule` is the 64-hex SHA-256 of the authorized
  `v02-training-descriptor.v1` bytes and must equal the disjointness receipt's
  `training_descriptor_digest`;
- top-level `hyperparameters` is the 64-hex SHA-256 of the complete closed candidate
  configuration, base/tokenizer revisions, target modules, renderer digest,
  seed, dependency lock and license ledger;
- top-level `development_report_digest` is the 64-hex SHA-256 of the selected
  candidate report blob as stored at `selection_commit`;
- top-level `seed` is the decimal string `20260929`;
- top-level `code_revision` is the full 40-hex clean implementation commit used
  consistently by sealing, training and evaluation, including any reviewed
  replacement required by the post-Phase-B correction rule;
- `sealed_descriptor_digest`, `descriptor_commit`, `descriptor_path`,
  `selection_commit`, and `candidate_deployment_class` retain their exact #44
  meanings. The selection schema has no `protocol_digest`; the later aggregate
  report binds the protocol digest.

Commit the three terminal candidate manifests at
`benchmarks/results/v02-candidate-{c1-r8,c2-r16,c3-r32}-dev.v1.json` plus
development aggregate at
`benchmarks/results/v02-candidate-selection-dev.v1.json` first; that commit is
`selection_commit`. The aggregate contains the SHA-256 digest of each of the
three candidate report blobs. The standalone validator proves all four exact
blobs and their recorded digests exist in that tree, proves
`training_capsule == disjointness_receipt.training_descriptor_digest`, and
proves every artifact binds the same `code_revision`. These #45-specific
checks are an additive layer around the reused #44 validation: the command
reads the fixed descriptor and result paths from `descriptor_commit` and
`selection_commit` without changing `_validate_selection` or its schema.
Generate the receipt referring to that already-existing commit, add it in a
later commit, and merge the provenance PR
to `main` with a merge commit, never squash or rebase. Before held-out access,
validate the receipt from a fresh full clone of `origin/main` using
`benchmarks.v02_evaluation validate-selection`. In addition to #44's existing
ancestry/blob rules, this standalone command resolves the explicit
`refs/remotes/origin/main` ref and requires `selection_commit` to be its
ancestor; absence or ambiguity of that ref fails closed.

### Phase F — one-time held-out execution

This phase is forbidden until both provenance PRs are on `main`, the descriptor
commit is an ancestor of `selection_commit`, and the standalone
`validate-selection` gate passes against their Git blobs.

Run the selected Saracura checkpoint and every available frozen control on the
sealed PT-BR payload under #44. Candidate and Kev execute in the frozen RTX
4090 systems-comparability group declared above. Publish only the validated aggregate report;
never commit raw rows, identities, predictions, labels, local paths, or secrets.
Do not tune, retry for quality, change a threshold, or replace the candidate
after opening the payload.

Post-condition: the aggregate report validates offline and records either a
bounded claim permitted by #44 or no comparative claim. A poor result is valid
evidence and does not authorize another test run.

### Phase G — final integration

Run independent review, full repository validation, PR CI, merge, and
post-merge verification. Model publication, Hugging Face upload, README
quickstart replacement, version bump, GitHub release, and public announcement
remain #46–#48.

## Allowed and forbidden scope

Public aggregate evidence paths are fixed:

- `benchmarks/manifests/v02-first-checkpoint.v1.json`;
- `benchmarks/manifests/v02-training-descriptor.v1.json`;
- `benchmarks/manifests/v02-sealed-test-descriptor.v1.json`;
- `benchmarks/manifests/v02-readiness-descriptor.v1.json`;
- `benchmarks/manifests/v02-disjointness-receipt.v1.json`;
- `benchmarks/results/v02-candidate-c1-r8-dev.v1.json`;
- `benchmarks/results/v02-candidate-c2-r16-dev.v1.json`;
- `benchmarks/results/v02-candidate-c3-r32-dev.v1.json`;
- `benchmarks/results/v02-candidate-selection-dev.v1.json`;
- `benchmarks/results/v02-selection-receipt.v1.json`;
- `benchmarks/results/v02-heldout-aggregate.v1.json`.

`benchmarks.validate_manifests` routes every public descriptor/receipt manifest
above. Private adapter tensors, raw rows, fingerprints, predictions, optimizer
state and cloud receipts remain outside Git. Operational GPU cost has no hard
ceiling; the executor records provider cost and reports each cumulative USD 10
crossing without stopping the run.

Allowed:

- the files listed in Phase B;
- private create-only training/evaluation artifacts outside the repository;
- a cloud GPU provider as an operational substrate, provided the exact device,
  image, dependencies, cost, and artifact digests are recorded;
- bounded corrections necessary to make the frozen implementation work.

Forbidden:

- a different base model or architecture;
- additional candidates, seeds, prompts, datasets, or benchmark frameworks;
- changes to public API behavior, default runtime, Julia, Laya, Kev, Jev, email
  adapters, browser automation, AIOS, or calibration;
- use of private Felhen operational/customer data;
- generation of a replacement training or test corpus inside #45;
- weakening #44, opening held-out data early, or treating held-out data as
  development evidence;
- publishing weights, uploading to Hugging Face, tagging a release, changing
  README positioning, or making comparative claims;
- provisioning or modifying the VM151 clinical-first runtime.

`IMPLEMENTATION_BLOCKED` is distinct from `NO_RELEASE_CANDIDATE`. It covers an
unsupported deterministic CUDA kernel, bitsandbytes/base incompatibility, or
the inability to reproduce the closed architecture/renderer. It stops before
selection and held-out access and cannot be converted into a quality failure.

## Deterministic validation

Before any live data or GPU phase:

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src benchmarks tests
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.v02_evaluation validate-protocol
uv build
git diff --check
```

Live phases add exact commands defined by `benchmarks.v02_training`; they must
support separate `preflight`, `seal-readiness`, `validate-readiness`, `smoke`,
`train-candidate`, `select`, `evaluate` and `verify-artifact` subcommands. The
sealer alone accepts all three lane paths; `validate-readiness` accepts only
aggregate descriptors/receipts. A single command that trains and opens the
held-out payload is forbidden.

Before Phase F, the additional standalone gate is:

```bash
uv run python -m benchmarks.v02_evaluation validate-selection \
  --sealed-descriptor /absolute/path/to/sealed-test-descriptor.json \
  --disjointness-receipt /absolute/path/to/disjointness-receipt.json \
  --selection-receipt /absolute/path/to/selection-receipt.json
```

## Completion evidence

#45 is complete only with:

1. a `READY` Phase A evidence receipt created before GPU spend;
2. descriptor and selection provenance commits reachable from `origin/main`
   with preserved ancestry and standalone selection validation;
3. three immutable candidate manifests and terminal reports;
4. one deterministic development aggregate;
5. either one valid selection receipt or a terminal
   `NO_RELEASE_CANDIDATE` report with no held-out access;
6. when selected, one held-out aggregate passing the #44 offline validator;
7. an immutable selected adapter/head artifact with SHA-256 and exact base,
   tokenizer, data, code, configuration, dependency, seed, license, and
   hardware provenance;
8. independent review, green CI, merged PR, and clean `main` verification.

No qualitative statement such as “looks good” substitutes for these artifacts.

## Rollback and interruption

Public implementation rolls back by Git revert. Private artifacts are
create-only and are preserved as evidence; they are never overwritten or
silently deleted. Before selection, an operationally interrupted run may resume
only from exact matching state. After `selection_commit` or
`NO_RELEASE_CANDIDATE`, candidate identity is immutable and no correction is
allowed under this protocol revision. After sealed-test access, no retry or new
candidate is allowed either.

Cloud GPU shutdown is an operational cleanup step after artifacts are copied
and digest-verified. It must not delete the only copy of a candidate, receipt,
or report.
