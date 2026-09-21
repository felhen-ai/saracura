---
title: Phase 3A annotation protocol and deterministic split gate
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase3a-first-party-packet.md
globalRef: qmd://saracura/docs/action/specs/phase3a-first-party-packet.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-21
sourceRefs:
  - github:felhen-ai/saracura#4
related:
  - docs/action/specs/phase2c-data-governance-gate.md
  - SECURITY.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 3A annotation protocol and deterministic split gate

## Context

Phase 2C approved a metadata policy, not dataset bytes. Future first-party
human-original text is the only candidate for the initial PT-BR train and
evaluation families. Deterministic derivatives may later be created only from
accepted train atoms. Model-generated or model-assisted examples, customer
exports, AIOS traces, and unvetted public text remain quarantined or blocked.

The narrow workflow is `support-routing.v1`: choose the single queue that owns
the next action from `billing`, `technical_support`, `account_access`,
`subscription_cancellation`, or `order_delivery`. There is currently no human
contributor roster, base-state file, split plan, reviewed packet, training run,
calibration artifact, or quality evidence.

## Decision and scope

Phase 3A freezes the annotation protocol and implements a repository-only,
offline tool that validates future human-authored base-state commitments and
assigns their scenario families to train/dev/calibration/blind-test before any
variant or derivative is created.

This phase delivers:

1. a PT-BR annotation guide with priority and tie-break rules;
2. a bundled protocol manifest bound to the Phase 2C policy registry and guide;
3. a closed base-state JSONL contract;
4. deterministic stratified family splitting with golden vectors;
5. exact/normalized/near-duplicate and mechanical privacy-pattern prechecks;
6. test-only simulated inputs proving positive and negative behavior.

Phase 3A does not implement the final training-data artifact, independent review
records, adjudications, derivatives, dataset card, takedown ledger, training, or
blind-test access. Those belong to Phase 3B after real humans and contribution
grants exist. Every authorization and claim literal remains false.

## Non-goals

- Author, generate, download, translate, import, publish, or bundle dataset
  records.
- Use Codex, another model, a customer, AIOS, or a provider as a data author.
- Prove automatically that an opaque contributor ID belongs to a human.
- Approve privacy, rights, representativeness, statistical power, training,
  calibration, publication, or automation.
- Add dataset libraries, ML runtimes, network access, telemetry, redaction, or
  remote storage.
- Rebalance a frozen split after review, exclusion, or relabeling.

## Governance boundary

The tool validates declared structure, exact bytes, hashes, duplicate/leakage
signals, and deterministic assignments. It cannot prove that a contributor is
human, that a declaration is truthful, or that text contains no subtle personal
information. Maintainers verify those facts outside the validator before Phase
3B.

Contributor IDs are opaque public packet identifiers. The mapping to a verified
human and the consent/contribution grant stays outside the repository. A base
state carries only a SHA-256 reference to that external attestation. Two agents,
two accounts controlled by one person, or one person in two roles do not satisfy
future independence requirements.

Phase 2C controls are classified for Phase 3A as follows:

| Control | Phase 3A treatment |
| --- | --- |
| `record_provenance`, `human_author_identity` | declared in the base-state contract; human truth remains externally verified |
| `grouped_split` | mechanically enforced for the PT-BR-only base-state inventory and split plan |
| `cross_locale_family_link` | deferred; Phase 3A rejects every non-PT-BR input |
| `duplicate_scan` | mechanically enforced before assignment |
| `privacy_record_review`, `rights_record_review` | author declarations plus mechanical precheck only; independent approval deferred |
| `artifact_manifest`, `byte_hashes` | source and plan bytes are hashed, but final artifact manifest is deferred |
| `contamination_scan`, `shortcut_audit`, `takedown_lineage`, `independent_human_review` | pending for Phase 3B |

No Phase 3A output may describe all Phase 2C controls as completed.

## Annotation protocol

Add `docs/annotation-guides/support-routing.v1.md`. It defines the primary owner
as the queue that can perform the next material action. Topic mentions do not
create multilabel output. Apply these rules in order:

1. `account_access` when authentication, identity verification, credential
   recovery, or account reachability blocks every other requested action;
2. `billing` for a completed or attempted charge, payment, refund, invoice, or
   payment-method action, unless the only requested action is cancellation;
3. `subscription_cancellation` when the next action is to stop a subscription,
   renewal, or recurring plan and there is no separate completed charge/refund
   action;
4. `order_delivery` for shipment, tracking, delivery, missing package, damaged
   delivery, or physical-order fulfillment;
5. `technical_support` for malfunction, configuration, compatibility, or
   product-use help not blocked by account access.

Use `ambiguous` when equal ownership remains, required context is absent, or a
future independent review cannot resolve a label under the frozen rules. Use
`out_of_scope` when none of the five queues owns the action. These are review
outcomes and never forced into a five-class label.

The guide may contain clearly marked illustrative snippets. They have no record
IDs, are never imported by code, and are not dataset records or human-original
evidence.

## Protocol manifest

Add `benchmarks/manifests/support-routing-protocol.v1.json`. Its root has exactly
these fields:

- `schema_version`: literal `support-routing-protocol.v1`;
- `workflow_id`: literal `support-routing`;
- `workflow_revision`: literal `v1`;
- `locale`: literal `pt-BR`;
- `labels`: the five labels in the order above;
- `review_outcomes`: exactly `ambiguous`, `out_of_scope`, `rejected`;
- `guide_path`: literal `docs/annotation-guides/support-routing.v1.md`;
- `guide_sha256`: lowercase SHA-256 of the exact guide bytes;
- `policy_registry_path`: literal
  `benchmarks/manifests/training-data-source-policies.v1.json`;
- `policy_registry_sha256`: lowercase SHA-256 of the exact registry bytes;
- `base_state_schema`: literal `support-routing-base-state.v1`;
- `split_algorithm`: literal `support-routing-stratified-sha256.v1`;
- `split_proportions`: exact ordered array of objects
  `[{split=train,weight=40},{split=dev,weight=20},`
  `{split=calibration,weight=20},{split=blind_test,weight=20}]`; array order is
  the allocation and Hamilton tie order, independent of JSON object-key order;
- `normalization`: literal `support-routing-text-normalization.v1`;
- `unicode_behavior`: literal `python-3.11-3.13-vector-locked.v1`;
- `near_duplicate`: exact object `unit=word_trigram_jaccard`,
  `threshold_numerator=85`, `threshold_denominator=100`;
- `limits`: exact object `max_input_bytes=16777216`,
  `max_plan_bytes=8388608`, `max_states=10000`,
  `max_text_codepoints=2000`, `max_atoms_per_state=8`,
  `max_pair_comparisons=50000000`;
- `training_authorized`, `quality_claims_allowed`, and
  `publication_authorized`: literal `false`.

The manifest validator rejects unknown/missing fields, duplicate keys, reordered
labels/outcomes/split keys, changed literals, invalid bounds, and digest mismatch.
It reads only the two fixed repository-relative files. It makes no network call.

## Base-state JSONL contract

The split builder accepts one local UTF-8 file. It opens it without following
symlinks, snapshots at most 16 MiB into memory, verifies the same regular-file
metadata before and after the read, then hashes and parses that snapshot. FIFO,
device, socket, directory, symlink, growth, replacement, invalid UTF-8, BOM,
carriage return, empty line, missing final newline, comments, and more than
10,000 lines fail closed.

Each line is one RFC 8785 canonical JSON object followed by `\n`. Lines are
strictly sorted by `state_id`. Each object has exactly:

- `schema_version`: literal `support-routing-base-state.v1`;
- `state_id`: `state_` plus 32 lowercase hex characters;
- `family_id`: `fam_` plus 32 lowercase hex characters;
- `locale`: literal `pt-BR`;
- `candidate_label`: one of the five protocol labels;
- `text`: NFC, 12–2,000 Unicode code points, without Unicode control or format
  characters other than ordinary line-free spacing;
- `author_id`: `human_` plus 16 lowercase hex characters;
- `author_attestation_sha256`: 64 lowercase hex characters referencing external
  maintainer-verified evidence;
- `source_atom_ids`: 1–8 unique, sorted values matching
  `atom_[0-9a-f]{32}`;
- `authoring_mode`: literal `human_original`;
- `privacy_declaration`: literal `no_personal_data`;
- `rights_declaration`: literal `approved_first_party`;
- `created_at`: RFC 3339 UTC with seconds and terminal `Z`;
- `content_digest`: lowercase SHA-256 of the RFC 8785 canonical object with only
  `content_digest` removed.

Phase 3A allows exactly one state per family. State IDs, family IDs, and atom IDs
are globally unique, so no atom or family can cross splits. English or translated
records require a later protocol revision; they cannot enter this file.

Changing text, label, author, atoms, declaration, or timestamp creates a new
state and family only before a plan is frozen. After freeze, Phase 3B review may
exclude a state but cannot modify, replace, rerandomize, or rebalance it. A
requested relabel is exclusion-only in Phase 3A. Any later correction contract
must preserve the same global family identity and split across revisions and is
deferred to Phase 3B; prior holdout exposure can never be reset by a new ID.

## Pre-split scans

`support-routing-text-normalization.v1` is: require NFC; apply Unicode
`casefold`; map every code point whose Unicode category starts with `Z` and every
ASCII whitespace code point to one ASCII space; collapse consecutive spaces;
retain punctuation, symbols, and accents; trim leading/trailing spaces.

Exact raw-text duplicates and equal normalized text between distinct states are
blocking. Word tokens are maximal consecutive code points for which
`str.isalnum()` is true after normalization. A word trigram is each consecutive
three-token tuple. For two nonempty trigram sets, the Jaccard threshold is tested
without floats as `intersection * 100 >= union * 85`. Fewer than three tokens
produce no trigrams and score zero; equal normalized text was already blocked.
Near-duplicate comparison covers every distinct state pair and fails before
50,000,000 comparisons could be exceeded. Any pair at or above the threshold is
blocking before split assignment.

The mechanical privacy precheck blocks text matching versioned, tested patterns
for email, HTTP(S) URL, `www`, IPv4/IPv6, Brazilian CPF/CNPJ, payment-card-like
digit runs, common phone forms, bearer/API/private-key markers, or `@handle`.
Patterns are deliberately overinclusive. There is no automatic real-company or
person-name detector; the author declaration and later independent review own
that boundary. A clean scan is never reported as anonymization or privacy proof.

The implementation freezes these detectors as named pure functions and golden
positive/negative vectors: email accepts a 1–64 character local part plus DNS
labels; URL begins case-insensitively with `http://`, `https://`, or `www.`;
IPv4/IPv6 candidates are accepted only by `ipaddress.ip_address`; CPF/CNPJ
blocks 11 or 14 digits after removing only dot, slash, and hyphen separators;
payment-card-like blocks 13–19 digits after removing only spaces and hyphens;
phone blocks 10–13 digits when written with `+`, parentheses, spaces, or
hyphens; secret markers cover PEM private-key headers, `Bearer` credentials,
and `api_key`, `apikey`, `token`, or `secret` followed by `:` or `=` and a
non-space value; handle is `@` plus 2–30 ASCII letters, digits, or underscores.
Changing a detector or vector requires a new protocol revision.

NFC, `casefold`, Unicode category handling, and `isalnum` are tested with the
same committed golden vectors on Python 3.11, 3.12, and 3.13. If their outputs
differ, CI fails and the protocol must be revised; an implementation may not
silently accept version-dependent bytes.

## Deterministic split algorithm

The command accepts the validated canonical base-state snapshot, a public
64-lowercase-hex seed, and a create-if-absent output path. For each
`(locale, candidate_label)` stratum, compute:

```text
rank_sha256 = SHA-256(
  UTF-8("support-routing-stratified-sha256.v1\0" + seed + "\0" + family_id)
)
```

Sort by `(rank_sha256, family_id)`. Allocate the stratum using Hamilton largest
remainder for weights 40/20/20/20 in the fixed tie order `train`, `dev`,
`calibration`, `blind_test`. Sparse strata may leave splits empty; the tool does
not invent records or claim adequacy.

The canonical `split-plan.json` has exactly:

- `schema_version`, literal `support-routing-split-plan.v1`;
- `algorithm_id`, the protocol algorithm literal;
- `seed`;
- `protocol_manifest_sha256` and `source_file_sha256` over exact bytes;
- `assignments`, ordered by `family_id`, each with exactly `family_id`,
  `state_id`, `locale`, `candidate_label`, `split`, and `rank_sha256`;
- `strata`, exactly five objects in protocol-label order, including zero-count
  labels, each with exactly `locale`, `candidate_label`, `total`, `train`,
  `dev`, `calibration`, and `blind_test` integer counts;
- `plan_digest`, SHA-256 of the RFC 8785 canonical root object with only
`plan_digest` removed. File bytes are that canonical object plus exactly one
terminal `\n`; the digest excludes the terminal newline because it covers the
canonical object, while external file hashes cover exact bytes.

The output contains no text, author, atom, path, or timestamp. It is written to
a sibling temporary file, fsynced where supported, and atomically installed
create-if-absent. Existing output, including identical bytes, returns
`WRITE_CONFLICT`; nothing is overwritten or removed. A separate validation
command recomputes every digest, rank, allocation, count, and assignment from
the source snapshot.

The validator snapshots `--plan` with the same no-follow regular-file and
before/after identity checks used for base states, capped at 8 MiB. It requires
valid UTF-8, no BOM or carriage return, one RFC 8785 canonical root object,
exactly one final newline, no duplicate keys, no non-finite number, and no
trailing content. Symlink, FIFO, device, socket, directory, growth, replacement,
or oversized plan input fails before semantic validation.

Hamilton golden vectors cover zero/sparse strata, exact ties, remainders, all
five labels, and input sizes around each allocation boundary. Because input
JSONL must already be canonical and sorted, determinism tests compare identical
canonical snapshots rather than accepting arbitrary line order.

An entirely empty base-state file is invalid. Individual label strata may be
empty and remain present with zero counts. The allocation is a plumbing
mechanism, not statistical authorization. Planning
targets remain 1,000/500/500/500 independent PT-BR families, with
200/100/100/100 per label. Exclusions can reduce or skew accepted counts and do
not trigger rebalancing.

## CLI and public errors

Add `python -m benchmarks.first_party_gate` with:

- `validate-protocol` with no additional arguments;
- `build-split-plan --states <file> --seed <64hex> --output <file>`;
- `validate-split-plan --states <file> --plan <file>`.

Success prints one fixed sentence per command to stdout. Validation failures
return `2` and print exactly one bounded code to stderr:
`PROTOCOL_INVALID`, `BASE_STATES_INVALID`, `DIGEST_MISMATCH`,
`PRIVACY_BLOCKED`, `DUPLICATE_BLOCKED`, `RESOURCE_LIMIT`, `SPLIT_PLAN_INVALID`,
`WRITE_CONFLICT`, `ARGUMENTS_INVALID`, or `UNEXPECTED_FAILURE`. Argument parsing is custom and never
echoes supplied values. Exception text, paths, text, contributor IDs, URLs,
hashes, and counts are not printed on failure. Help contains metavariables only.

No command accepts a URL, provider, dataset, download, network, model, override,
authorization, or publication flag. Tests replace socket constructors and prove
the commands make no network call.

## Test provenance and packaging

Positive tests create simulated declarations in temporary directories. Their
strings are agent-authored test material, truthfully named as simulations, and
are never committed as JSONL artifacts, discovered by benchmark code, or
presented as human-original data. The production schema has no `test_only`
bypass.

The guide, protocol manifest, and `benchmarks` tooling are source-repository
assets and are not added to the installed wheel. The sdist may contain source
docs, tests, and benchmark code but no JSONL dataset artifact. Update package
inspection so zero artifact arguments fail and CI inspects the concrete wheel;
tests inspect the sdist for forbidden dataset extensions/packet basenames and
weights.

## Implementation scope

Expected changes:

- `docs/action/specs/phase3a-first-party-packet.md`;
- `docs/annotation-guides/support-routing.v1.md`;
- `benchmarks/manifests/support-routing-protocol.v1.json`;
- `benchmarks/first_party_packet.py`;
- `benchmarks/first_party_gate.py`;
- `benchmarks/validate_manifests.py`;
- `benchmarks/inspect_wheel.py`;
- `.github/workflows/ci.yml` so the newly built wheel and sdist are both
  inspected;
- focused tests plus README/CONTRIBUTING documentation.

No production JSONL or split-plan file is committed.

## Acceptance criteria

1. The guide freezes labels, priority rules, ambiguity/out-of-scope handling,
   human-independence limits, privacy/rights boundaries, and blind-test rules.
2. The protocol manifest is a complete closed contract and binds the exact guide
   and Phase 2C registry bytes offline.
3. Base-state parsing enforces canonical bytes, closed fields, one state per
   family, unique atoms, provenance declarations, content digests, file-snapshot
   safety, and all hard resource bounds.
4. Duplicate and privacy prechecks implement the exact versioned behavior above
   and block before assignment.
5. Split generation and validation are byte-for-byte deterministic, stratified,
   grouped, fail-closed, atomic, no-clobber, and covered by golden vectors plus a
   concurrent-writer test.
6. Every invariant has a focused negative test, including malformed Unicode,
   short-text comparison, symlink/FIFO/device rejection where supported,
   oversized state and plan inputs, digest tampering, atom/family reuse, all
   privacy categories, all Hamilton boundary cases, and bounded-error
   non-disclosure.
7. No committed file is a real or simulated production dataset, no output
   authorizes training/publication/claims, and no model-assisted text is
   represented as human-authored evidence.
8. Default installation remains lightweight and network-free. The wheel excludes
   data tooling/protocol assets as designed; wheel and sdist inspections receive
   concrete artifact paths and fail on zero inputs.
9. Ruff, format, mypy over `src benchmarks tests`, pytest, manifest validation,
   protocol validation, default-environment validation, build, wheel/sdist
   inspection, secret scan, and `git diff --check` pass.

## Validation commands

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src benchmarks tests
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.first_party_gate validate-protocol
uv run python -m benchmarks.validate_default_environment
uv build
uv run python -m benchmarks.inspect_wheel dist/*.whl dist/*.tar.gz
git diff --check
```

## Rollout and rollback

This is an offline repository change. Merge only after CI `quality` and
`secrets` pass. There is no service deploy, data migration, model download,
cache mutation, training, or remote artifact. Rollback is a Git revert.

## Next gate

Phase 3B starts only after maintainers name real human authors/reviewers, verify
contribution grants, and create a canonical base-state file outside the
repository. It defines independent reviews, disagreement truth tables,
adjudication, exclusions, final packet manifests, dataset card, takedown
lineage, contamination/shortcut evidence, and the exact transition—if any—to
research training. Calibration and one-time blind-test access remain later
gates.

## Open risks

- Human authorship and independence remain governance facts, not validator
  outputs.
- Frozen lexical priority may create shortcuts; Phase 3B must audit them before
  training.
- Quadratic near-duplicate scanning is intentionally bounded and may require a
  reviewed algorithm revision above 10,000 states.
- Unknown overlap with encoder pretraining cannot be excluded.
- Public attestation references must prove contribution authority without
  exposing unnecessary personal information.
