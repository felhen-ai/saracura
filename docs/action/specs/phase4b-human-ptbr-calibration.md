---
title: Phase 4B human PT-BR packet, training, calibration, and blind evaluation
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4b-human-ptbr-calibration.md
globalRef: qmd://saracura/docs/action/specs/phase4b-human-ptbr-calibration.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs: []
related:
  - docs/action/specs/phase3a-first-party-packet.md
  - docs/action/specs/phase4a-opt-in-minilm-backend.md
  - docs/annotation-guides/support-routing.v1.md
  - SECURITY.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4B human PT-BR packet, training, calibration, and blind evaluation

## Context

Phase 4A made the frozen synthetic MiniLM head executable as an explicit local
backend. It deliberately returns `fixture_only`: the head was trained from a
synthetic packet, has no fitted temperature, and has never been evaluated on an
independently reviewed human PT-BR blind set.

Phase 3A already freezes the support-routing annotation rules and deterministic
family split, but it stops before contributor grants, independent record review,
the final packet, human-head training, calibration, and blind evaluation. Those
are the remaining prerequisites for research-grade confidence evidence.

This phase builds the entire offline pipeline and its fail-closed boundaries.
The repository will contain validators, schemas, simulated test fixtures, and
reproducible algorithms. It will not contain private contributor mappings,
attestations, human records, model weights, or generated evidence. A real run is
an external operator action after actual humans and grants exist.

Two language models, two accounts controlled by one person, or model-authored
text reviewed by a human do not become `human_original`. Such records remain in
the synthetic/model-assisted lane and cannot satisfy this phase.

## Goal

Add a checkout-only, offline `benchmarks.human_research` pipeline that can:

1. validate independently authored and independently reviewed PT-BR records;
2. materialize immutable train, dev, calibration, and blind-test views from the
   already frozen Phase 3A split plan;
3. train a reproducible five-class linear head using only train and dev;
4. fit one bounded temperature using only calibration records;
5. evaluate that frozen temperature once on the untouched blind-test records;
6. produce a runtime-compatible immutable calibration artifact and a sanitized
   research report without authorizing automation or broad quality claims.

The tooling is complete when all simulated positive and adversarial acceptance
tests pass. Phase 4B evidence is complete only when a separately authorized real
human packet passes the pipeline. These states must never be conflated.

## Non-goals

- Author, paraphrase, translate, label, or review real packet records with an AI.
- Bundle real records, contributor identities, attestations, grants, weights, or
  calibration artifacts in Git or the wheel.
- Treat opaque IDs or digests as proof that a person or grant exists.
- Tune the encoder, change the five-label taxonomy, or add dynamic labels.
- Choose thresholds, abstention policy, risk gates, production automation, or a
  service API.
- Select hyperparameters or retry a model after seeing blind-test metrics.
- Claim statistically representative Brazilian Portuguese from the minimum
  viable research packet.
- Make Phase 4C batching or high-volume changes in this increment.

## External evidence boundary

All real inputs and outputs live outside the repository. Training, fit, and
blind stages receive only operator-chosen capsule directories with mode `0700`,
an exact allowlisted file set, and no sibling split. The only narrow input
exceptions are: the installer-only four-view release-input capsule; the explicit
Phase 4A verified encoder snapshot directory, whose existing registry and
descriptor contract remains unchanged; and the one externally created signed
receipt file. Files are regular, non-hardlinked, non-symlinked, mode `0600`, and
are opened through a package-owned extension of the Phase 4A descriptor walker.
Every read is bounded, snapshots bytes once, and proves the same
`(device,inode,size,mode,uid,nlink,mtime_ns,ctime_ns)` before, during, and after
the read. No command performs network access, cache discovery, telemetry,
environment-variable path discovery, or parent-directory scanning.

The private mapping from opaque contributor/reviewer IDs to real people and the
actual consent/grant documents remains outside packet bytes. The public packet
contains only lowercase SHA-256 commitments. Maintainers must verify the
documents out of band before invoking the finalization command. Code validates
the declarations and commitments; it cannot establish their truth.

Real human inputs are never test fixtures. Tests use a distinct
`simulated-human-packet.v1` trust domain and ephemeral test public keys that are
not present in package resources. Production schemas reject that domain. Only a
maintainer release receipt signed by an active key in the bundled
`research-trust-keys.v1.json` resource can promote a packet or final calibration.
The initial resource contains no active key, so this implementation cannot emit
or accept `verified_for_research` until a later reviewed PR adds a public key;
the private key is never generated, stored, or used by this repository.

## Required input contracts

### Phase 3A base states and split plan

The pipeline consumes the exact canonical base-state JSONL and split plan
validated by `benchmarks.first_party_gate`. It recomputes both file digests,
revalidates the Phase 3A protocol manifest, records, duplicate/privacy scans,
and deterministic assignments, and refuses a changed seed, assignment,
protocol digest, family, text, or label.

The serving backend accepts at most 320 code points and 1,280 UTF-8 bytes.
Materialization therefore excludes a reviewed Phase 3A record exceeding either
limit with fixed reason `backend_input_limit` before any minimum is counted.
It does not truncate, replace, move, or rebalance the family.

The final packet requires at least 500 independent families:

- at least 200 train, 100 dev, 100 calibration, and 100 blind-test records;
- at least 10 independent families for every label in every split;
- exactly one record per family in this first human revision;
- at least two human authors overall, at least three human reviewers overall,
  and at least two accepted records from each participating author and reviewer.

The minimum is a research floor, not proof of representativeness or production
fitness. Counts are computed from the frozen plan; records are never moved or
rebalanced to satisfy them.

### Contributor registry

`contributors.json` is canonical RFC 8785 JSON plus one trailing newline. Its
root is a closed object:

- `schema_version`: `support-routing-contributors.v1`;
- `protocol_sha256`: digest of the Phase 3A protocol manifest;
- `contributors`: nonempty array sorted by `contributor_id`;
- `registry_sha256`: SHA-256 of the canonical root without this field.

Every contributor object has exactly:

- `contributor_id`: `human_` plus 16 lowercase hex characters;
- `identity_attestation_sha256`: 64 lowercase hex;
- `contribution_grant_sha256` and `review_grant_sha256`: 64 lowercase hex;
- `grant_scopes`: sorted nonempty subset of `human_original_authoring`,
  `independent_annotation`, and `adjudication`;
- `verified_by`: `maintainer_` plus 16 lowercase hex;
- `verified_at`: canonical UTC seconds;
- `revoked`: literal `false`.

The registry must cover every author, reviewer, and adjudicator reference.
Phase 3A v1 author states repeat the author ID and identity-attestation digest,
but their frozen closed schema cannot repeat a contribution-grant digest. For an
author, this registry binds that ID to a valid `human_original_authoring`
contribution grant, and the later packet receipt binds the complete registry
digest. Reviews and adjudications still repeat their matching identity and
review-grant digests from this registry. In unsigned intake, `verified_by` is an
opaque, format-checked maintainer ID; the later signed packet receipt
collectively authenticates it rather than resolving it individually during
materialization. A revoked, missing, conflicting, duplicate, or unreferenced
contributor fails closed.

### Blinded independent annotations and adjudication

The author candidate label is hidden from reviewers. `annotations.jsonl` is
canonical JSONL sorted by `(state_id, reviewer_id)`. Train records have one
independent annotation. Dev, calibration, and blind-test records have exactly
two annotations from two distinct reviewers. Reviewers are distinct from the
author and cannot read each other's submission before both are sealed. Each
closed object has:

- `schema_version`: `support-routing-independent-annotation.v1`;
- `state_id`, `family_id`, and `content_digest`, matching the base state;
- `reviewer_id`: `human_` plus 16 lowercase hex and different from `author_id`;
- `reviewer_attestation_sha256` and `review_grant_sha256`: lowercase SHA-256;
- `annotation_guide_sha256`: exact Phase 3A guide digest;
- `reviewer_label`: one protocol label, `ambiguous`, `out_of_scope`, or
  `rejected`; there is no candidate-label or agreement field;
- `privacy_review`: `approved_no_personal_data` or `rejected`;
- `rights_review`: `approved_first_party` or `rejected`;
- `reviewed_at`: canonical UTC seconds;
- `annotation_digest`: SHA-256 of the canonical object without this field.

After required annotations are sealed, reconciliation compares them with the
hidden author candidate. A record is accepted without adjudication only when
all reviewer labels equal the candidate and all privacy/rights decisions are
approved. Any non-label outcome or rejected review excludes the family.

Label disagreement produces one `adjudications.jsonl` entry from a third human
who is distinct from author and reviewers and has the adjudication grant. The
adjudicator may see the candidate and sealed reviewer labels. Its closed object
binds all annotation digests and selects one label or an exclusion outcome. It
accepts the record only when the selected label equals the original candidate;
Phase 4B never relabels a frozen state. Missing, duplicate, premature, or
unnecessary adjudication fails closed. Counts and author/reviewer minima are
evaluated after every exclusion, including `backend_input_limit`.

### Artifact control ledger and takedown lineage

`controls.json` is a closed `support-routing-artifact-controls.v1` object bound
to the policy-registry digest and packet input digests. It contains exactly the
13 controls required for `saracura-human-original`, ordered by control ID. Each
entry has `control_id`, `status=completed`, `scope=packet`, a versioned
`method_id`, `evidence_sha256`, `reviewer_id`, and `reviewed_at`. The
cross-locale entry uses method `ptbr-only-no-cross-locale-families.v1`; it is
not silently omitted. Contamination and shortcut entries bind their complete
sanitized reports, including known-source revisions, thresholds, counts, and
uncertainty. A detector result is evidence, never a proof of no contamination.

`takedown-ledger.jsonl` is append-only canonical JSONL. The initial packet has
one genesis record and zero removal events. Every later event binds the prior
event digest, source atom, affected state/family and artifact digests, reason,
timestamp, and disposition. Any removal invalidates packet release receipts,
training manifests, checkpoints, fits, and blind reports derived from the old
digest and requires a new packet revision and retraining. The packet cannot be
training-eligible until all 13 controls and the lineage ledger validate.

### Signed maintainer release boundary

The installed package resource `research-trust-keys.v1.json` is a closed,
versioned registry of Ed25519 public keys with key ID, 32-byte public key in
lowercase hex, activation/revocation timestamps, and allowed scopes. The
`local-minilm` extra adds `cryptography>=45,<46`, with the resolved version
locked in `uv.lock`. Runtime
verification uses only this resource; there is no operator-supplied public key,
environment override, remote key, or fallback.

An external signer creates canonical `maintainer-release-receipt.v1` JSON. Its
signed preimage excludes only `signature_ed25519` and binds `scope`, `key_id`,
`evidence_mode=real_human`, protocol/policy/state/plan/contributor/annotation/
adjudication/control/takedown/packet digests, timestamp, and expiry. Scope
`packet_training` is required before embedding extraction. Scope
`research_calibration` additionally binds training manifest, checkpoint, fit,
blind view, exposure run, blind report, and calibration candidate digest.
The signature is lowercase hex over the exact RFC 8785 preimage bytes. Unknown,
inactive, revoked, expired, wrong-scope, test, or invalid signatures fail before
model construction. Reviewer identity/grant digests are transitively bound via
the signed contributor, annotation, and adjudication digests.

## Closed nested schemas

Every JSON object below rejects unknown/missing keys, duplicate keys, non-NFC
strings, booleans where integers are required, non-finite numbers, invalid
calendar timestamps, invalid identifier/hash lengths, and noncanonical bytes.
Arrays have the stated order and exact or bounded length. Unless a field is
explicitly nullable, `null` is forbidden.

### Trust keys and receipts

The trust registry root has exactly `schema_version=research-trust-keys.v1`,
`keys`, and `registry_sha256`. `keys` is sorted by `key_id`, at most 16 entries,
and each entry has exactly `key_id`, `public_key_ed25519` (64 lowercase hex),
`active_from`, nullable `revoked_at`, and sorted `scopes` drawn only from
`packet_training` and `research_calibration`. `registry_sha256` hashes the
canonical root without itself.

A release receipt has exactly `schema_version=maintainer-release-receipt.v1`,
`scope`, `key_id`, `evidence_mode=real_human`, `issued_at`, `expires_at`,
`evidence`, and `signature_ed25519` (128 lowercase hex). Its signed preimage is
the canonical root without only `signature_ed25519`; its external receipt digest
is SHA-256 of the complete canonical file bytes. `expires_at` is after
`issued_at`, at most 31 days later, and verification uses a current UTC instant
provided through an internal clock dependency, never a CLI argument,
environment variable, artifact field, or file mtime. Production uses the
system UTC clock; tests inject a fixed clock.

For `packet_training`, `evidence` has exactly the SHA-256 fields `protocol`,
`policy_registry`, `states`, `split_plan`, `contributors`, `annotations`,
`adjudications`, `controls`, `takedown_ledger`, and `packet`. For
`research_calibration`, it has exactly those ten plus `training_manifest`,
`checkpoint`, `fit`, `blind_view`, `exposure_run`, `blind_report`, and
`calibration_candidate`. Cross-scope fields fail closed.

### Capsule descriptors

Every `capsule-descriptor.json` has exactly `schema_version=human-capsule.v1`,
`capsule_kind`, `packet_sha256`, nullable `packet_release_receipt_sha256`,
`files`, and `descriptor_sha256`. `capsule_kind` is one of `train-dev`,
`calibration`, `blind`, `embedding`, `training`, `fit`, or `blind-run`. `files` is a 1..16
array sorted by relative POSIX basename; each object has exactly `path` (one
basename, no slash/dot), positive `bytes`, and lowercase `sha256`. It lists every
sibling file except the descriptor itself and no other directory entry may
exist. `descriptor_sha256` hashes the canonical root without itself. The packet
and receipt fields must match the byte-identical listed files when that capsule
kind requires them. Intake and unreleased governance use their separately named
closed schemas and are never accepted by stage commands.

### Packet and embedding manifests

`packet-manifest.json` has exactly `schema_version`, `protocol_sha256`,
`policy_registry_sha256`, `states_sha256`, `split_plan_sha256`,
`contributors_sha256`, `annotations_sha256`, `adjudications_sha256`,
`controls_sha256`, `takedown_ledger_sha256`, `files`, `accepted_counts`,
`excluded_counts`, `labels`, `split_algorithm`, `source_type`, `locale`,
`synthetic_only`, `authorizations`, and `packet_sha256`.

`files` has exactly the four keys `train`, `dev`, `calibration`, and
`blind_test`; each value has exactly positive `bytes` and lowercase `sha256`.
`accepted_counts` has exactly positive `total`, `by_split`, and
`by_split_label`; `by_split` has exactly the four split names and positive
integers; `by_split_label` has those four split names, each mapping exactly the
five ordered labels to positive integers. Totals, views, labels, unique
families, and the minimum rules must reconcile. `excluded_counts` has exactly
nonnegative `total` and `by_reason`; `by_reason` has exactly the fixed reasons
`backend_input_limit`, `review_rejected`, `review_disagreement_excluded`, and
`adjudication_excluded`, including zeros. `labels` is the exact ordered five-
label taxonomy; `split_algorithm` is the frozen Phase 3A identifier;
`source_type=human_original`, `locale=pt-BR`, and `synthetic_only=false`.
`authorizations` has exactly five literal-false keys: `training`, `calibration`,
`blind_test`, `automation`, and `broad_quality_claims`. `packet_sha256` hashes
the canonical root without itself.

`embedding-manifest.json` has exactly
`schema_version=human-embedding-manifest.v1`, `packet_manifest_sha256`,
`packet_release_receipt_sha256`, `train_dev_capsule_sha256`, `protocol_sha256`,
`encoder`, `labels`, `device=mps`, `batch_size=32`, `max_length=128`, `records`,
`files`, `environment`, and `manifest_sha256`. `encoder` is the same closed
encoder object used by the human training manifest. `records` is sorted by
`(split,state_id)` and each entry has exactly `split` (`train` or `dev`),
`state_id`, `family_id`, `label_index` 0..4, and `row_index` equal to its
zero-based tensor row; no text or token IDs appear. `files` has exactly
`embeddings.safetensors={bytes,sha256}`. `environment` has the exact
version/platform keys of the human training manifest. The tensors are exactly
float32 `embeddings [N,384]`, int64 `labels [N]`, and uint8 `splits [N]`, where
train is 0 and dev is 1. Sizes, rows, ordering, labels, and digests must
reconcile. `manifest_sha256` hashes the canonical root without itself.

### Human training manifest

The human manifest top-level field set is the one listed in the training
section. Nested contracts are exact:

- `encoder`: `candidate=multilingual-minilm-l12`, immutable reviewed `revision`,
  package `registry_sha256`, `device=mps`, `frozen=true`;
- `head`: `width=384`, `classes=5`, `optimizer=AdamW`, finite `lr=0.01`,
  `weight_decay=0.01`, `batch_size=32`, `maximum_epochs=80`,
  `early_stopping_patience=10`, `early_stopping_min_delta=0.001`;
- `files`: exactly lowercase SHA-256 values for `embeddings.safetensors` and
  `checkpoint.safetensors`, with the latter equal to supplied checkpoint bytes;
- `environment`: nonempty strings `python`, `torch`, `transformers`,
  `tokenizers`, `safetensors`, and `platform`;
- `authorizations`: exactly `calibration=true`, `automation=false`, and
  `quality_claims=false`;
- `metrics`: exactly `synthetic_only=false`, integer `epochs` 1..80,
  `selected_epoch` 1..epochs, `epoch_ledger`, `train`, and `dev`;
- every epoch-ledger object has exactly sequential positive `epoch`, finite
  nonnegative `train_loss`, and finite `dev_macro_f1` in `[0,1]`; selection and
  early stopping recompute by the frozen Phase 3B rule;
- each train/dev metric object has exactly positive `count`, finite `accuracy`
  and `macro_f1` in `[0,1]`, a nonnegative integer 5x5 `confusion_matrix`, and
  `per_label` with exactly the five ordered labels, each containing nonnegative
  integer `support` plus finite `[0,1]` `precision`, `recall`, and `f1`; counts,
  supports, precision/recall/F1 (zero denominator is `0.0`), macro-F1,
  accuracy, and matrix totals must all recompute and reconcile;
- `serving_conformance`: exactly `schema=human-minilm-conformance.v1`, package
  `token_payload_sha256`, ordered five `labels`, arrays of exactly five finite
  float CPU and MPS logits, their lowercase SHA-256 digests, and
  `comparison={rtol:0.00001,atol:0.000001}`.

Top-level digest fields are lowercase SHA-256; seed is integer `20260921`; labels
are the exact ordered taxonomy. `manifest_sha256` hashes the canonical root
without itself. Synthetic-only fields or values are rejected by the human
branch and human-only fields are rejected by the synthetic branch.

### Fit, metrics, candidate, and final artifact

Every occurrence of “all `CalibrationContext` fields” in this phase means
exactly `model_id`, `model_revision`, `checkpoint_sha256`,
`architecture_config_sha256`, `serializer_version`, `tokenizer_revision`,
`truncation_policy_id`, `precision`, `quantization`, `output_transform`,
`workflow_id`, `workflow_revision`, `question_id`, `dataset_id`,
`dataset_revision`, `split_manifest_sha256`, `locale`, `domain`, `head`,
`cardinality_bucket`, and literal-null `risk_policy`; no context alias or
additional field is accepted.

`temperature-fit.json` has exactly `schema_version=temperature-fit.v1`, all
closed `CalibrationContext` fields, packet/receipt/calibration-view/model/
checkpoint/golden-resource SHA-256 fields, `algorithm_revision`, temperature,
fit count, independent count, exact five-label count object, `metrics_before`,
`metrics_after`, `created_at`, and `fit_sha256`. Its metric objects have exactly
finite nonnegative `nll`, `brier`, and `ece_15`. `fit_sha256` hashes the canonical
root without itself.

A blind metric object has exactly `count`, `per_label`, `accuracy`, `macro_f1`,
`confusion_matrix`, `nll`, `brier`, `ece_15`, `mean_confidence`, and
`max_confidence`. Its classification subobjects and reconciliation rules match
the training metric contract; the four confidence/calibration values are finite
and in their mathematical ranges (`nll,brier>=0`; others `[0,1]`). A blind report
has exactly schema `blind-report.v1`, model/packet/fit/blind/exposure/golden
digests, `metrics_before`, `metrics_after`, `created_at`, and `report_sha256`,
which hashes the canonical root without itself.

`ResearchCalibrationCandidate` has exactly
`schema_version=research-calibration-candidate.v2`, every `CalibrationContext`
field, `status=pending_maintainer_release`, `method=temperature-scaling`,
`parameters={temperature,bounds:[0.5,5.0]}`, the positive fit/evaluation sample
and independent-state counts, `minimum_independent_state_count=100`, complete
blind `metrics_before` and `metrics_after`, `fit_sha256`, `blind_view_sha256`,
`exposure_run_sha256`, `blind_report_sha256`, `created_at`, and
`candidate_sha256`. Its digest hashes the canonical root without itself. It has
no calibration ID, receipt, threshold, abstention, risk, or authorization field.

`ResearchCalibrationArtifact` has the same context, method, parameters, counts,
metrics, evidence digests, and timestamp, but exactly `schema_version=2`,
`status=verified_for_research`, `candidate_sha256`, the full closed
`release_receipt`, `release_receipt_sha256`, and derived `calibration_id`; it has
no extra fields. Finalization copies candidate fields byte-for-byte except
schema/status, adds receipt/digests/ID, and never recomputes metrics. The research
receipt evidence binds `candidate_sha256`, so there is no signature/artifact
circularity. Receipt verification precedes compatibility checking.
`exposure_run_sha256` is exactly the lowercase `state_sha256` stored in
`02-running.json`; the blind report, candidate, `research_calibration` receipt
evidence, derived calibration ID, and finalization all use that value. It never
means the sealed-state or capsule-descriptor digest.

### Blind exposure states

Every exposure state has exactly `schema_version=blind-exposure-state.v1`,
`state`, `custody_key`, `packet_sha256`, `blind_capsule_sha256`,
`training_manifest_sha256`, `checkpoint_sha256`, `fit_sha256`, nullable
`previous_state_sha256`, `created_at`, and `state_sha256`. The precommit state is
`precommitted` with null predecessor; running is `running` and names the exact
precommit digest; sealed is `sealed`, names the exact running digest, and adds
exactly `blind_report_sha256` and `calibration_candidate_sha256`. Those two keys
are forbidden in earlier states. `state_sha256` hashes the canonical root
without itself. The fixed filenames `01-precommitted.json`, `02-running.json`,
and `03-sealed.json`, their predecessor links, immutable context, directory
allowlist, and creation order must agree. A partial file, skipped state,
additional entry, or different first precommit fails closed.

## Final packet materialization

Before materialization, an operator creates one intake capsule with exactly
`base-states.jsonl`, `split-plan.json`, `contributors.json`,
`annotations.jsonl`, `adjudications.jsonl`, `controls.json`,
`takedown-ledger.jsonl`, and `intake-manifest.json`. The closed intake manifest
records schema `support-routing-human-intake.v1`, the exact size/SHA-256 ledger
for the other seven files, protocol/policy digests, `evidence_mode=real_human`,
and its self-digest. Tests use schema `simulated-human-intake.v1`, which cannot
be promoted or accepted by production commands.

The materialization command is:

```text
python -m benchmarks.human_research materialize-packet
  --intake-capsule <intake-directory>
  --output-parent <existing-0700-directory>
  --packet-name <new-child-name>
```

The named child must not exist and is installed by atomic rename from a sibling
staging directory after every file and the parent are fsynced. Existing paths
always fail. It creates one unreleased release-input capsule containing exactly
`train.jsonl`, `dev.jsonl`, `calibration.jsonl`, `blind-test.jsonl`,
`packet-manifest.json`, `packet-release-request.json`, `controls.json`,
`takedown-ledger.jsonl`, and `release-input-manifest.json`. The last file is a
closed `packet-release-input.v1` object with exactly `schema_version`,
`packet_sha256`, `files`, and `manifest_sha256`; `files` is the sorted exact
size/SHA-256 ledger for the other eight files and `manifest_sha256` hashes the
canonical root without itself. This capsule is accepted only by
`install-packet-release`, never by a training, fit, or blind command.

After an external signer returns canonical `packet-release-receipt.json`,
`install-packet-release --release-input-capsule <directory> --receipt <file>
--output-parent <existing-0700-directory> --release-name <new-child-name>` opens
only those two explicit inputs, verifies every release-input byte and the receipt
against the bundled trust store, and creates new immutable released capsules
under a new child path. It never scans a parent or resolves a sibling. `train-dev/` contains
exactly `train.jsonl`, `dev.jsonl`, `packet-manifest.json`,
`packet-release-receipt.json`, and `capsule-descriptor.json`; `calibration/`
contains exactly `calibration.jsonl` plus those four governance files; `blind/`
contains exactly `blind-test.jsonl` plus those four governance files. Each
descriptor has schema/split, exact size/SHA-256 ledger for all other files,
packet and receipt digests, and a self-digest. Canonical packet-manifest and
receipt bytes are identical in all three capsules. Stage commands open only the
supplied capsule descriptors by directory FD and cannot resolve sibling names.
Views are sorted by `state_id` and contain only bounded non-private fields and
evidence digests; they contain no attestation paths or identity details.

`packet-manifest.json` is a closed `support-routing-human-packet.v1` object that
binds:

- protocol, states, split-plan, contributor, annotation, adjudication, control,
  and takedown-ledger digests;
- exact file byte sizes and SHA-256 digests for all four views;
- accepted/excluded totals and per-split/per-label independent-family counts;
- the ordered five-label taxonomy and frozen split algorithm;
- `source_type=human_original`, `locale=pt-BR`, and `synthetic_only=false`;
- nested `authorizations={training:false,calibration:false,blind_test:false,
  automation:false,broad_quality_claims:false}`;
- `packet_sha256`, computed over the canonical manifest without that field.

No command accepts individual view files. Every trusted stage capsule carries
the canonical packet manifest and full canonical receipt bytes, not only their
digests; every downstream stage re-verifies the signature. The unsigned packet
never authorizes training by itself.

## Reproducible human-head training

`extract-train-dev-embeddings` accepts only the exact train/dev capsule, signed
packet-training receipt, explicit Phase 4A verified encoder snapshot, device
`mps`, batch size 32, and a new output child. It reconstructs tokenizer/encoder
directly from verified bytes through the package loader; the existing
cache-discovering benchmark loader is forbidden. It reads records in state-ID
order, uses max length 128 and the exact Phase 4A pooling path, and writes one
sealed float32 safetensors embedding artifact plus its manifest. It makes one
MPS extraction pass. Reproducibility is claimed only downstream of those sealed
embedding bytes; Phase 4B does not claim bitwise reproducibility of independent
MPS encoder runs.

Its output embedding capsule contains exactly `embeddings.safetensors`,
`embedding-manifest.json`, `packet-manifest.json`,
`packet-release-receipt.json`, and `capsule-descriptor.json`; the latter four
bind the exact input capsule and encoder bytes. `train-head` output contains
exactly `checkpoint.safetensors`, `training-manifest.json`, `training-report.json`,
the same canonical packet manifest and receipt, and its descriptor. Every
downstream command receives the training capsule, not loose manifest/checkpoint
paths, and verifies the packet receipt before constructing the model.

`train-head` accepts that embedding capsule, public seed `20260921`, and one
explicit Phase 4A verified encoder snapshot solely to generate the
checkpoint-specific CPU/MPS serving conformance evidence. The snapshot identity,
revision, and package-registry digest must equal the embedding manifest's frozen
encoder binding. It never receives packet views, calibration, or blind paths;
it performs no cache discovery or snapshot-path re-open. It repeats the CPU head
training twice and requires byte-identical checkpoints and epoch ledgers.

The numerical contract remains Phase 3B's frozen encoder and linear
`Linear(384,5)` head. Training uses the exact Phase 3B optimizer, loss,
batch-size, maximum-epoch, early-stopping, deterministic ordering, label order,
and MPS encoder/CPU head execution path. Any change requires a new training
revision and new calibration. It emits:

- `checkpoint.safetensors` with only `weight [5,384]` and `bias [5]`, float32;
- `training-manifest.json`, schema `human-training-manifest.v1`;
- a sanitized training report with aggregate train/dev metrics only.

The closed human manifest has exactly `schema_version`, `workflow_revision`,
`synthetic_only`, `packet_manifest_sha256`, `packet_release_receipt_sha256`,
`embedding_manifest_sha256`, `training_code_sha256`, `protocol_sha256`,
`encoder`, `labels`, `seed`, `head`, `files`, `metrics`, `environment`,
`serving_conformance`, `authorizations`, and `manifest_sha256`.
`schema_version=human-training-manifest.v1`, workflow revision is
`phase4b-human-research-training.v1`, and `synthetic_only=false`. It binds the
packet and release, exact train/dev and embedding digests, encoder registry and
revision, code/environment, seed, epoch ledger, selected epoch, and checkpoint.
`serving_conformance` binds the package token vector/digest plus checkpoint-
specific CPU and MPS logits/digests for that frozen self-authored text. The
backend uses these manifest values for a human checkpoint; it never compares a
human head with the Phase 4A synthetic-head logits. It declares
`calibration_authorized=true`, `automation_authorized=false`, and
`broad_quality_claims_authorized=false`. The Phase 4A verified-byte loader is
generalized to accept this closed manifest variant without weakening the
existing synthetic variant.

Repeating CPU-head training from the same sealed embedding bytes on the same
bound runtime must produce the same selected epoch and checkpoint digest.
Existing output paths are never replaced.

## Calibration-only stage

`fit-temperature` accepts only an explicit Phase 4A verified encoder snapshot,
the training capsule, and the released calibration capsule. The verified human
head comes only from the training capsule. Both capsules must carry
byte-identical packet manifest and packet receipt bytes. Its exact directory
allowlists make a blind file invalid, and it re-verifies the canonical receipt
before loading model or text.
It reconstructs the human backend and emits immutable `temperature-fit.json`;
it does not emit a runtime calibration.

For each calibration record it records in memory only the ordered five logits,
gold label index, state/family ID, and input-token count. The output contains
only aggregate counts/digests and never text or per-record logits.

Float32 backend logits are copied to CPU and converted elementwise to float64
before any metric. For temperature `T`, stable log-sum-exp is
`m + log(sum(exp(z_k/T-m)))`, `m=max(z/T)`; softmax subtracts that same `m`;
NLL is the mean negative gold log-probability. Temperature minimizes calibration
NLL on `[0.5,5.0]`. Search uses `a=log(0.5)`, `b=log(5)`,
`r=(sqrt(5)-1)/2`, points `c=b-r*(b-a)`, `d=a+r*(b-a)`, and exactly 96 updates.
Compute `fc=objective(exp(c))` and `fd=objective(exp(d))` once. For each update,
first snapshot `(old_a,old_b,old_c,old_d,old_fc,old_fd)`. If
`(old_fc,exp(old_c)) <= (old_fd,exp(old_d))`, set `a=old_a`, `b=old_d`,
`d=old_c`, `fd=old_fc`, `c=old_d-r*(old_d-old_a)`, then compute only
`fc=objective(exp(c))`. Otherwise set `a=old_c`, `b=old_b`, `c=old_d`,
`fc=old_fd`, `d=old_c+r*(old_b-old_c)`, then compute only
`fd=objective(exp(d))`. A reused objective is never recomputed. This defines
equality, endpoints, and point reuse for all 96 iterations.
The final candidate set is `{0.5,1.0,5.0,exp(a),
exp(b),exp(c),exp(d)}`; choose its minimum ordering key. Non-finite intermediates
fail. Including `T=1` guarantees fitted calibration NLL cannot exceed the
uncalibrated calibration-set NLL. The artifact stores the finite IEEE-754
double serialized by RFC 8785.

For `C=5`, Brier is `mean(sum_k((p_k-1[k=y])**2))` without division by C.
Prediction and confidence are the lowest-index argmax and its probability.
Macro F1 is the arithmetic mean of all five per-label F1 values; any undefined
precision/recall/F1 is zero. ECE uses bin
`min(14,floor(confidence*15))`; each nonempty bin contributes
`n_bin/n * abs(mean(correct)-mean(confidence))`, empty bins contribute zero.
The package includes checkout-only `human-calibration-golden.v1.json` with
finite logits/labels, expected probabilities and all metrics, plus optimizer
objectives exercising constant/lower-bound, interior, upper-bound, and exact-
tie branches. Expected values are compared at `rtol=1e-12, atol=1e-12` on Python
3.11-3.13; the resource digest is bound into fit/report revisions.

The fit artifact binds the full `CalibrationContext`, packet/calibration view
digests, calibration independent-state count and per-label counts, algorithm
revision `bounded-log-temperature-golden-v1`, pre/post calibration NLL, ECE,
and Brier score, and a fit digest. It requires at least 100 independent states
and 10 per label. It is not accepted by `DecisionEngine`.

The fit output capsule contains exactly `temperature-fit.json`,
`packet-manifest.json`, `packet-release-receipt.json`, and its descriptor.

## One-time blind evaluation and runtime artifact

Blind execution is an append-only state machine inside an explicit
`--exposure-registry` directory. The custody key is SHA-256 of only the packet
manifest and blind-capsule descriptor digests. It is independent of model,
checkpoint, fit, code, or timestamp, so the same blind release can never be
used to compare a second candidate. `begin-blind` atomically creates the new
keyed child and `01-precommitted.json` before opening blind bytes. The
precommit records the first and only model/checkpoint/fit. Any existing custody
key blocks every later begin, including a different model or fit. It then creates
`02-running.json` inside `<custody-key>.active`, and evaluates the exact blind
capsule. It then creates `blind-report.json`, `calibration-candidate.json`,
`03-sealed.json`, and `capsule-descriptor.json` in that order in the active
directory, fsyncs every file and the directory, and atomically renames the
complete directory to `<custody-key>`. The descriptor is written last before
rename, so no visible sealed path can lack it. `resume-blind` accepts only the
`.active` run directory and the same explicit inputs; it may continue a
precommitted/running run after recomputing every digest. While still running it
reuses an already-created immutable report or candidate only if every byte and
expected digest recomputes exactly; otherwise it fails. It cannot alter or rerun
a sealed result. If `.active` already contains the exact complete sealed set,
`resume-blind` validates every byte, state link, descriptor, and fsync invariant,
then performs only the pending atomic rename; it never reevaluates the model.
`begin-blind` fails if either the active or sealed name exists;
creation and final rename use create-if-absent semantics. No state file is
edited or deleted. A sealed run is a
`blind-run` capsule containing exactly the three exposure states, report,
candidate, and descriptor; finalization accepts only this descriptor-backed
sealed form. The exposure registry and an unsealed run are the sole narrow
state-machine exception to the input-capsule rule: their exact state-dependent
allowlists are `{01}`, `{01,02}`, `{01,02,report}`,
`{01,02,report,candidate}`, and the complete active pre-rename set above, with
no other entry.

The blind command accepts only the released blind capsule, training capsule,
fit capsule, explicit verified encoder snapshot, and registry. Each trusted
capsule carries the same packet manifest and packet receipt, which are verified
again. It has no train/dev/calibration paths.
It first proves all context digests and split-family disjointness. The registry
proves at most one run within this custody boundary; operators remain responsible
for not copying or exposing blind bytes elsewhere.

The blind report contains aggregate, non-record-level metrics before and after
temperature scaling:

- count and per-label support;
- accuracy, macro F1, 5x5 confusion matrix;
- multiclass NLL and Brier score;
- 15-bin equal-width ECE with `[0,1/15) ... [14/15,1]` boundaries;
- mean confidence and maximum confidence;
- model, packet, fit, and blind-view digests.

Metrics use float64 CPU calculations, deterministic label order, explicit empty
bin behavior, and no rounded values in JSON. Markdown may round for display.
Blind evaluation requires at least 100 independent states and 10 per label.

Blind output first creates an unsigned `research-calibration.v2` candidate with
status `pending_maintainer_release`. A separate external signing step produces
the research-calibration release receipt. `finalize-calibration` verifies that
receipt against the bundled trust store. It accepts only the sealed blind run
directory, the canonical externally returned receipt file, an injected internal
clock, and a new output path; it reconstructs and verifies the candidate,
report, exposure chain, packet/training/fit digests, and receipt preimage before
atomically creating the final closed
`ResearchCalibrationArtifact` with `schema_version=2`, status
`verified_for_research`, method `temperature-scaling`, full context, fixed
bounds, fit/blind counts, fit digest, blind-view digest, exposure-run digest,
blind report digest, release-receipt object, metrics, and timestamp. It has no
threshold, abstention, risk policy, or automation authorization.

The dataset profile is:

- `dataset_id=support-routing-human-ptbr`;
- `dataset_revision=human-ptbr-v1.<first-32-hex(packet_sha256)>`;
- `split_manifest_sha256=packet_sha256`.

Its ID is `research-calibration-v2-<first-32-hex(sha256(canonical({context,
method,parameters,fit_sha256,blind_view_sha256,exposure_run_sha256,
blind_report_sha256,release_receipt_sha256})))>`, where those are the exact
serialized object keys and `context` is the exact closed context object. Pydantic recomputes it and
validates a real canonical UTC timestamp. Existing schema-v1 temperature
artifacts remain parseable only with `fixture_only`; schema v1
`verified_for_research` is rejected. Only schema v2 with a valid signature may
produce that response status.

## CLI and public package boundary

Human packet construction, training, fitting, and blind evaluation remain under
`benchmarks` and are excluded from wheels. The installed package changes
only enough to:

- accept and strictly validate the human training manifest variant;
- parse the exact discriminated human manifest and its checkpoint-specific
  serving conformance;
- validate schema-v2 artifacts, bundled trust keys, receipts, derived IDs, and
  timestamps through descriptor-safe reads;
- derive the human `CalibrationDatasetProfile` only from a signed schema-v2
  artifact after receipt verification;
- run the existing MiniLM backend with a compatible final artifact.

`saracura decide` first securely parses the calibration envelope. Omitted
`--dataset-profile` remains valid only for schema-v1 identity/fixture artifacts.
For schema v2, the CLI derives the profile from the signed artifact, constructs
the expected backend/request context, and then runs normal compatibility
validation. The Python API exposes the same verified `dataset_profile()` method;
it never trusts an arbitrary operator profile. There is no implicit discovery.
Default fixture behavior remains byte-compatible.

## Security and privacy failure behavior

All parsing is closed, duplicate-key rejecting, NFC/RFC-8785 canonical, bounded,
and fail-closed. Errors expose codes, paths, counts, and digest identities only;
they never echo record text, contributor mappings, logits, secrets, or external
absolute paths. Socket creation is forbidden in tests for every command.

Symlinks in any ancestor, hard-linked mutable inputs, FIFOs/devices/sockets,
wrong ownership modes, same-process mutation, noncanonical bytes, unknown
fields, duplicate records, family leakage, author/reviewer equality, missing
grants, changed split assignments, and digest mismatches fail before model
construction.

The verified I/O primitives use these hard maxima: base states 16 MiB; split
plan 8 MiB; contributors, controls, packet, intake, release-input, embedding,
training, capsule-descriptor, packet-release-request, training-report, blind-
report, calibration-candidate, and final-calibration JSON files 1 MiB each;
annotations 32 MiB; adjudications and each split view 16 MiB; takedown ledger
8 MiB; trust registry and each receipt/exposure state 64 KiB; embeddings 64 MiB;
checkpoint 16 MiB; temperature fit 1 MiB. Record count is
at most 10,000. Atomic writers set 0600 before content, fsync file and parent,
and use create-if-absent or sibling-directory rename; no existing byte is
replaced. Human runtime calibration uses secure verified reads; legacy public
fixture examples retain their current lightweight path behavior.

The sdist may contain benchmark source and the checkout-only golden JSON, but
must exclude `.artifacts`, every JSONL, weights, and packet/report basenames:
`contributors.json`, `annotations.jsonl`, `adjudications.jsonl`, `controls.json`,
`takedown-ledger.jsonl`, all four views, every capsule/packet/embedding/training/
fit/blind/exposure/calibration manifest or report name, and any release receipt.
Package inspection tests the concrete wheel and sdist denylist.

## Acceptance criteria

1. Phase 3A manifests, golden Unicode vectors, and existing fixture/MiniLM paths
   remain unchanged and pass the full suite.
2. Simulated tests prove contributor grants, reviewer blinding, one-versus-two
   annotation cardinality, disagreement truth table, adjudicator independence,
   backend-limit exclusion, post-exclusion minima, all 13 controls, takedown
   chaining, packet capsules, canonicalization, and no-network behavior.
3. Production trust-store tests prove no active key means no real promotion;
   ephemeral test keys work only through an internal test trust store, while
   unknown/revoked/expired/wrong-scope/test/altered receipts all fail before
   model construction. Schema-v1 forged research status is rejected.
4. Training tests prove the embedding stage opens only the exact train/dev
   capsule and explicit verified snapshot, never cache discovery or sibling
   splits; CPU training is twice byte-identical; the checkpoint-specific
   conformance and exact human manifest are enforced.
5. Calibration golden vectors prove stable softmax/log-sum-exp, float64
   conversion, NLL/Brier/ECE/macro-F1 semantics, every optimizer branch,
   `T=1` fallback, tie-break, minima, digest binding, and inability to open blind
   bytes during fit on Python 3.11-3.13.
6. Blind tests prove the append-only `precommitted -> running -> sealed` state
   machine, explicit same-input resume, no second begin, family disjointness,
   sanitized aggregate metrics, unsigned pending output, signed v2 finalization,
   derived IDs/profile, and no real status from simulated evidence.
7. The Phase 4A loader accepts both its original real synthetic manifest and a
   valid signed human manifest with its own conformance vector; cross-variant
   fields, synthetic logits, and authorization changes fail before construction.
8. Every new file family is covered by the exact byte/count limits and
   descriptor mutation/symlink/hardlink/mode tests. Default and `local-minilm`
   installs pass on Python 3.11-3.13; wheel/sdist
   inspection proves the wheel has no human pipeline and neither archive has a
   packet, report, checkpoint, record, receipt, or generated evidence. The sdist
   intentionally retains reviewable benchmark source and the golden resource.
9. `ruff`, formatting, `mypy`, full `pytest`, manifest validation, build,
   archive inspection, and `gitleaks` pass.
10. An independent code reviewer returns no blocking findings.

The acceptance suite does not fabricate a completed real Phase 4B evidence run.
That gate remains visibly pending until independently verified human inputs are
provided.

## Rollout and rollback

This is a research-tooling and local package increment with no service deploy,
external write, or paid call. Merge through the normal PR/CI path. Rollback is a
Git revert; external operator artifacts are immutable and are neither deleted
nor rewritten by rollback.

After merge, maintainers may recruit contributors and reviewers under a
separate human process. Real packet paths and identities must never be pasted
into issues, PRs, CI, logs, or the public repository.

## Next gate

Phase 4C may begin after this tooling is merged because its batching and
backpressure work does not require looking at blind labels. Public confidence
or PT-BR quality claims still wait for the real Phase 4B human evidence run.
Phase 4C must preserve numerical conformance or create a new model revision and
new calibration requirement.
