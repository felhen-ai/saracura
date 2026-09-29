---
title: v0.2 distillation-safe corpus and sealed PT-BR capsule
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/v02-distillation-safe-corpus.md
globalRef: qmd://saracura/docs/action/specs/v02-distillation-safe-corpus.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-29
sourceRefs:
  - github:#53
  - github:#55
related:
  - docs/action/specs/v02-model-evaluation-protocol.md
  - docs/action/specs/v02-first-owned-checkpoint.md
  - benchmarks/manifests/v02-model-evaluation-protocol.v1.json
supersedes: []
supersededBy: []
sensitivity: public
---
# v0.2 distillation-safe corpus and sealed PT-BR capsule

## Outcome

Replace the rights-blocked Phase 4E training packet with an independently
generated, reviewed, publication-capable corpus and prepare the sealed PT-BR
capsule required by the frozen v0.2 evaluation protocol. After both artifacts
are ready, revise the #45 data boundary to their immutable digests.

This document must freeze the smallest executable 1–2 day recovery path. It
does not authorize implementation, model download, corpus generation, GPU
rental, training, or held-out access until independent review returns GO.

## Mandatory boundaries

- No row, paraphrase, identity, prompt output, or derivative from the blocked
  Phase 4E packet may enter the successor corpus.
- No hosted-model output may enter training or sealed rows.
- The #44 protocol and #45 release architecture/candidate grid remain frozen.
- Raw rows, annotations, labels, paths, credentials, and provider payloads
  remain outside Git.

## Required design content

### Model roles and revisions

| Role | Model | Family | License |
| --- | --- | --- | --- |
| Training-corpus author | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Qwen | Apache-2.0 |
| Independent reviewer | `mistralai/Mistral-Small-3.1-24B-Instruct-2503@68faf511d618ef198fef186659617cfd2eb8e33a` | Mistral | Apache-2.0 |

Different Apache-2.0 model families remove hosted-output restrictions while preserving capable PT-BR generation and review. Author and reviewer are distinct model families; neither is a hosted API.

#### Author contract

- Model: `Qwen/Qwen3.5-9B` at immutable revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Preflight verifies that the
  downloaded snapshot and license bytes match the frozen identity; it does not
  resolve a moving revision at execution time.
- Role: training-corpus author only. It generates candidate PT-BR training rows from frozen prompts and self-hosted runtime.
- License: Apache-2.0 at the pinned revision, verified by the preflight command before any GPU allocation. No hosted model API call for authoring.
- Clean-room boundary: the author process cannot mount or read the Phase 4E
  packet, its prompts, rows, identities, or rejected outputs. The successor
  uses a new namespace, seed, plan, and prompt digests. The later three-lane
  sealer detects exact normalized identity/state-question/combined-content
  overlap under #44. This proves no lineage and no exact normalized reuse; it
  does not make an unverifiable claim to detect every semantic paraphrase.

#### Reviewer contract

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` at immutable revision
  `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Role: independent reviewer only. It reviews generated rows for PT-BR quality, privacy, and policy compliance.
- License: Apache-2.0 at the pinned revision, verified by the preflight command before any GPU allocation. No hosted model API call for reviewing.
- Review is blind: the reviewer receives generated content only, never Phase 4E outputs or identities.

### Corpus plan

The planner creates exactly 1,600 new slots under namespace
`saracura-v02-cleanroom-v1` before inference. It reuses only the public task
taxonomy and typed Choice contract, never historical row content or prompt
outputs. Seed `20260929` assigns exactly 1,360 `train` and 240 `internal_dev`
slots. Locale counts are exactly 960 PT-BR and 640 English: train contains 816
PT-BR and 544 English; internal-dev contains 144 PT-BR and 96 English. Exactly
300 bilingual pairs consume one slot per locale and remain in one split: 255
pairs in train and 45 in internal-dev. Remaining unpaired slots fill the fixed
locale totals. Within each split × locale, domains and option counts 2–8 differ
by at most one; gold positions differ by at most one within each locale ×
option-count cell. Scenario codes, criterion-role targets and all ordering are
fixed in the plan. Families, bilingual pairs and deterministic derivatives
remain in one split.

The twelve domains are exactly `email_triage`, `customer_support`, `finance`,
`accounting`, `commerce`, `operations`, `scheduling`, `document_routing`,
`browser_action`, `security_triage`, `content_moderation` and
`personal_productivity`. Phase B freezes the existing public scenario-code and
criterion-role enums and their allowed domain mapping in the manifest; live
execution cannot add a domain, scenario or role.

The sealed training capsule requires at least 1,200 accepted rows: at least
1,020 train and 180 internal-dev; at least 60% PT-BR and 20% English; every
train locale × option-count cell has at least 20 and every internal-dev cell at
least 10; every train domain has at least 60 and every internal-dev domain at
least 10; and at least 120 bilingual pairs are complete. Percentage floors use
the final accepted denominator, while the fixed per-cell/domain/pair minima are
checked directly by `accepted + unresolved`. No training holdout is created: evaluation
uses only #44's `public_dev` and independently prepared `sealed_ptbr_test`.
Every accepted row requires exact author/reviewer answer agreement, exact
closed semantic-attestation agreement, and local schema, privacy, fictionality,
length and duplicate gates. Review never rewrites a row.

Before generation, every planned or authored record must pass both renderers
with truncation disabled: the frozen Kev rendering function digest
`eca2a60af37c539c984e89cf920c53e8d1c93cff6e980dea1a24dd520f86e169`
and the candidate renderer derived exactly from `kev/model.py` revision
`9c41005b2180347c3c646dfc9e50c4428483ec6b` with `max_state=384`,
`max_branch=1024`, `max_packed=2048`, `strict=true`,
`option_isolation=false`, `special_embeddings=false`, `lora_targets=all` and
`head_dim=256`. These 384/1024/2048 limits and the 512-token ceiling are the
joint admission contract for both renderers. They do not describe Kev's full
standalone serving context. Phase B freezes the source-file hash and resulting
`candidate_rendering_digest`; live work cannot begin before both digests are in
the reviewed manifest. Every accepted training and sealed record is at most 512
candidate-rendered tokens and passes the Kev preflight. A render failure rejects
the slot and remains in the denominator.

### Self-hosted runtime and quantization

- Author and reviewer run self-hosted from pinned Apache-2.0 weights.
- No hosted model API is used at any stage.
- Runtime container and dependency digests are pinned and recorded in receipts.
- The frozen serving runtime is `vLLM` in an OCI image pinned by digest. The
  implementation phase records the exact image, vLLM, CUDA, tokenizer and
  quantization revisions in a reviewed manifest before the pilot.
- Quantization is bitsandbytes NF4 with BF16 compute. AWQ, GPTQ and calibration
  datasets are outside this recovery attempt. Quantized runtime tensors are
  derivatives of the pinned source weights only and never redefine model
  identity.

### Cloud GPU class decision

- Execution runs sequentially on rented cloud GPU infrastructure using pinned container/runtime digests (D2).
- Sequential serving avoids local GPU contention and does not require simultaneous residency.
- Primary GPU class: one NVIDIA A100 80 GiB or equivalent single CUDA device.
  A100 40 GiB, A6000 48 GiB and L40S 48 GiB are eligible only when the exact
  frozen NF4 runtime passes the same full-load preflight. Models load
  sequentially, never simultaneously.
- A 24 GiB device is not an automatic fallback. It becomes eligible only if
  the offline implementation manifest has already frozen NF4, the
  no-generation preflight loads the complete pinned 24B reviewer without CPU
  offload, and the pilot meets the same throughput and schema gates. Otherwise
  preflight returns `BLOCKED_RUNTIME` before generation.
- No GPU or generation begins before code/spec review and a positive rights/license readiness receipt (I5).
- The preflight command records the resolved GPU class, container digest, and runtime digest in the plan receipt before any GPU allocation.

### Pilot and acceptance math

#### Bounded pilot

A bounded pilot runs before the full corpus (D3). It preserves speed without repeating long failed protocol experiments.

#### Feasibility stop

The pilot applies a mathematical feasibility stop before the full corpus. If the pilot cannot achieve the minimum feasible acceptance threshold, the recovery stops with terminal `NO_GO`. No provider/model substitutions are allowed after pilot results (R4).

#### Acceptance floor

The training pilot is an explicit 140-slot prefix built before the remaining
plan: exactly ten slots per locale × option-count cell, with 119 train and 21
internal-dev slots assigned by seeded round-robin while preserving the final
plan totals. It contains exactly 21 complete bilingual pairs (42 slots); both
members occupy the prefix and share one split. A slot is accepted only when the author output,
blind reviewer decision and every local gate pass on the first settled attempt;
model errors and invalid outputs remain in the denominator. The pilot metric is
therefore `accepted_slots / 140`, not model top-1 accuracy.

Continue only when at least 106 of 140 slots are accepted, at least 53 of 70 in
each locale, and at least 15 of 20 in each option-count bucket. These are corpus
feasibility gates and support no model-quality claim. The full run stops
terminally when `accepted + unresolved < required` for the global corpus or any
required split/locale/cardinality cohort. At each 20 newly resolved slots it
also records the one-sided 95% Wilson lower bound; the bound is diagnostic and
cannot override deterministic impossibility or lower a minimum.

### Public implementation boundary and commands

Phase B may add only `benchmarks/v02_corpus.py`,
`benchmarks/manifests/v02-distillation-safe-corpus.v1.json`,
`tests/test_v02_corpus.py`, manifest routing, bounded documentation links and
the isolated optional-extra entries in `pyproject.toml`/`uv.lock`.
Default installation and CI remain network-free and import no ML runtime. Live
execution dependencies belong to an isolated optional extra.

Every command that can read raw content requires an explicit absolute
`--artifact-root` owned by the current user, mode `0700`, outside the repository,
Git common directory, every registered worktree and synchronized folders. The
root is never discovered, defaulted, persisted in public artifacts or printed.
Raw artifacts use mode `0600`; aggregate receipts contain only relative logical
artifact IDs and digests. Commands reject symlinks, path traversal, existing
destinations and cross-device replacement. Writes use a sibling temporary file,
`fsync`, and atomic create-if-absent publication.
Training and sealed lanes require distinct roots and distinct process
credentials; each command rejects a root whose device/inode or resolved ancestry
matches the other lane's receipt. The trainer receives only the training root.

The public offline CLI is `uv run python -m benchmarks.v02_corpus` and supports
the following separate create-only commands (IF1):

| Command | Purpose |
| --- | --- |
| `preflight` | Verify runtime, GPU, container, and model prerequisites without generating output. |
| `plan` | Produce the frozen execution plan with cardinalities, seeds, and cost estimates. |
| `pilot` | Run the bounded feasibility pilot with mathematical stop. |
| `generate` | Generate training corpus rows via the author model. |
| `review` | Review generated rows via the reviewer model. |
| `seal-training` | Seal the training capsule after generation and review. |
| `prepare-sealed` | Create the frozen sealed PT-BR candidate plan. |
| `generate-sealed` | Generate sealed candidate cases with the pinned Phi author. |
| `annotate-sealed` | Annotate sealed rows with separate model-role identities. |
| `adjudicate-sealed` | Adjudicate annotation disagreements with a third independent role. |
| `seal-test` | Rotate gold positions, enforce #44 gates, and create the private payload plus aggregate descriptor. |
| `verify-receipt` | Verify a sealed capsule receipt against frozen digests and ancestry. |

All commands are create-only. They produce artifacts; they do not modify training data, review decisions, or sealed results after creation.

### Closed artifacts and receipts

Private receipts bind exact values without raw content (IF2):

- Exact model revisions and licenses.
- Prompt and plan digests.
- Runtime and container digests.
- Seeds.
- Raw file digests (not raw content).
- Counts.
- Status.

Before any model bytes or content reach rented infrastructure, the positive
readiness receipt also binds the cloud provider legal/service name, instance
class, region, storage and log retention periods, deletion mechanism, encryption
at rest/in transit, account boundary and either verified zero retention or the
exact post-run deletion obligation. It explicitly authorizes that named
environment to process the corpus, authorizes training and public distribution
of adapter/head weights, and prohibits publication of raw rows, prompts,
provider payloads, base tensors, calibration or automation claims. Missing or
unknown fields yield `BLOCKED_DATA_RIGHTS`.

Receipts never contain raw rows, raw prompts, raw provider payloads, paths, or credentials.

### Sealed annotation role identities

The sealed-test authoring/annotation lane is separate from training with separate model-role identities and credentials (D5). This preserves independence and prevents the trainer from designing its own test.

| Lane | Identity | Role |
| --- | --- | --- |
| Training | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Training-corpus author |
| Training review | `mistralai/Mistral-Small-3.1-24B-Instruct-2503@68faf511d618ef198fef186659617cfd2eb8e33a` | Independent reviewer |
| Sealed author | `microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083` (MIT) | Creates original fictional PT-BR candidate cases only |
| Sealed annotator A | `ibm-granite/granite-3.3-8b-instruct@51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` (Apache-2.0) | Blind label A |
| Sealed annotator B | `allenai/OLMo-2-1124-7B-Instruct@470b1fba1ae01581f270116362ee4aa1b97f4c84` (Apache-2.0) | Blind label B |
| Sealed adjudicator | `HuggingFaceTB/SmolLM3-3B@a07cc9a04f16550a088caea529712d1d335b0ac1` (Apache-2.0) | Resolves committed A/B disagreements |

All four sealed roles run self-hosted, use separate prompt digests and process
identities, and cannot read the training rows or prompts. Annotators receive a
candidate case without its proposed label and are blind to one another. The
adjudicator runs only after both label receipts are atomically committed and
receives the case plus both labels in a deterministic, identity-blinded order.
A case is accepted on annotator agreement only when the agreed label equals the
sealed author's private semantic target. On disagreement, the adjudicator must
select exactly label A or label B; the case is accepted only when that selected
label equals the author's target. Agreement on another label, an adjudicator
choice outside A/B, schema/privacy/fictionality/renderer failure, non-exclusive
options or unresolved ambiguity rejects the original identity; no role rewrites
the case.

The sealed plan uses seed `20260930`, exactly 140 candidate identities and
exactly 20 candidates for each option count 2–8. Its first 21 identities are a
stratified pilot of three per option count. The pilot continues only with at
least 15 accepted, at least two accepted per option count, at least 14 direct
A/B agreements and at most seven adjudications. The full lane requires at least
100 accepted and at least 14 per option count. It stops when accepted plus
unresolved cannot meet either minimum; there are no replacement identities,
prompt changes or second batch. After the gold label is final, a deterministic
seeded cyclic rotation places it so gold-position counts differ by at most one
inside each option-count bucket; label receipts bind the pre-rotation semantic
option ID and the descriptor binds the rotation digest. At least 100 accepted
records must satisfy #44 exactly. The
descriptor discloses that authoring and labeling are model-only; an optional
human audit is diagnostic and may reject records but cannot rewrite labels or
tune prompts. The trainer never sees the sealed payload, mount or credential.

`prepare-sealed` creates the immutable 140-identity plan only.
`generate-sealed` executes the Phi author into a sealed-only artifact root.
`annotate-sealed` creates the two blind label receipts; `adjudicate-sealed`
resolves committed disagreements; and `seal-test` performs rotation, frozen
#44 gates and emits the private payload plus public aggregate descriptor. The
descriptor is owned by issue #55. Issue #45 later validates and republishes the
same bytes in its evidence-boundary commit; it must not synthesize a competing
descriptor.

The descriptor fixes permutation-selection seed
`saracura-v02-sealed-permutation-v1` and subset size 50. Selection follows
#44's length-framed identity-hash ordering and binds the resulting selection
digest before any candidate run.

### Invariants

#### I1 — No blocked Phase 4E content reuse

The successor has no Phase 4E content lineage, and exact normalized overlap is
zero under #44's identity, state-question and combined-content fingerprints.
No semantic non-overlap claim is made beyond the clean-room process evidence.

#### I2 — Self-hosted, no hosted API

Training author and reviewer run self-hosted from pinned Apache-2.0 weights with no hosted model API.

#### I3 — Sealed capsule isolation

The sealed capsule remains inaccessible to the trainer and satisfies every frozen #44 distribution and annotation gate.

#### I4 — Public Git content boundary

Public Git contains only schemas, code, aggregate counts, licenses, and cryptographic digests. Raw rows, annotations, labels, paths, credentials, and provider payloads remain outside Git.

#### I5 — Precondition gate

No GPU or generation begins before code/spec review and a positive rights/license readiness receipt.

### One-time boundaries

- One frozen recovery attempt per issue.
- No provider/model substitutions after pilot results.
- Terminal `NO_GO` on infeasibility.
- Immutable artifacts use a sibling temporary file plus an atomic create-if-absent operation; an existing revision is never overwritten.
- After `selection_commit` or `NO_GO`, candidate identity and plan are immutable under this protocol revision.

### Cost reporting

- Operational GPU cost is recorded with provider-reported cost and conservative debit.
- The operator is informed whenever cumulative spend crosses another USD 10.
- Cost is reported at run completion.
- GPU cost is operational substrate cost; it does not authorize, block, or stop work beyond the feasibility gate.

### Terminal outcomes

| Terminal | Meaning |
| --- | --- |
| `READY` | All artifacts sealed, validated, and review passed; #45 amendment authorized. |
| `NO_GO` | Feasibility stop proves acceptance floor unreachable. Recovery stops. |
| `BLOCKED_DATA_RIGHTS` | Rights/license readiness receipt negative. Stop before GPU spend. |
| `BLOCKED_SEALED_TEST` | The frozen sealed plan cannot produce a compliant capsule. Stop before the full training-corpus run and before #45 candidate training. |
| `BLOCKED_REVIEW` | Independent review returns negative. Stop before implementation. |
| `BLOCKED_RUNTIME` | The pinned models cannot pass the frozen self-hosted runtime preflight. Stop before generation. |

`BLOCKED_REVIEW` stops before implementation; data-rights, sealed-test and
runtime blocks stop before generation. A terminal result never triggers model,
provider, prompt or threshold substitution inside this protocol revision.

### Risk mitigations

| Risk | Mitigation |
| --- | --- |
| R1: Open models may have lower first-pass acceptance. | Bounded pilot proves feasibility and stops early if the full acceptance floor becomes impossible. |
| R2: Mistral 24B may not fit the selected GPU/runtime. | Sequential NF4 serving preflight may change only the eligible GPU class, never quantization or model identity. |
| R3: Model-only sealed annotation may be insufficiently independent. | Freeze distinct blind model identities and roles, require committed labels before adjudication, preserve an optional bounded human spot-audit without using it to tune. |
| R4: Corpus generation becomes another research loop. | One frozen recovery attempt, no provider/model substitutions after pilot results, terminal NO_GO on infeasibility. |

### Issue 45 amendment plan

After both the corpus and sealed capsule are ready, revise #45's data boundary to their immutable digests (D4, IF3). The amendment sequence:

1. Seal the successor sealed PT-BR capsule and record its descriptor digest.
2. Seal the successor training corpus and record its descriptor digest.
3. Submit a reviewed spec amendment replacing the blocked #45 packet digest with the successor authorization and descriptors.
4. #45 resumes only after the amendment is independently reviewed and merged.

The amendment replaces #45's historical `accepted-packet manifest digest` with
the new `v02-cleanroom-training-capsule.v1` descriptor digest and replaces its
`successor-grant digest` with the positive readiness-receipt digest. It updates
the `v02-training-descriptor.v1` field names/validation accordingly without
changing the #44 schemas, release architecture or candidate grid. The #55
sealed descriptor bytes become the sole sealed descriptor consumed and
republished by #45; #45 does not create a second logical artifact.
The amendment must also replace every residual reference in #45 to the Phase 4E
train/internal-dev lane, `v02-training-capsule-authorization.v1`, the historical
provider route and Phase-C ownership of sealed-descriptor creation. No stale
Phase 4E data identity may remain in the amended readiness or sealing flow.

### Acceptance criteria

1. The spec and manifest pin every model revision and license, corpus and sealed
   role, plan cardinality, seed, prompt boundary, runtime/quantization choice,
   artifact schema, stop condition and privacy boundary.
2. Offline tests prove closed schemas, duplicate-key rejection, deterministic
   planning, blind review/annotation envelopes, create-only resume behavior,
   path confinement, terminal stops and aggregate-only receipts without network,
   model weights, datasets or optional ML dependencies.
3. A positive readiness receipt verifies the frozen license bytes and explicitly
   authorizes self-hosted generation, cloud processing, training, and public
   distribution of Saracura-owned adapter/head weights while forbidding raw-row
   publication.
4. The accepted training capsule meets every frozen count and cohort minimum;
   `seal-training` proves clean-room generation and zero exact normalized
   identity, state-question and combined-content overlap against the blocked
   packet. If the private old fingerprint sets cannot be validated against the
   #53 receipt and #45 historical digest, sealing returns
   `BLOCKED_DATA_RIGHTS`; absence never skips the check.
5. The sealed descriptor proves at least 100 PT-BR Choice records, option counts
   2–8 with at least 14 each, balanced gold positions, two committed blind labels,
   adjudication of every disagreement, the 512-token envelope and #44's frozen
   permutation subset.
6. The aggregate descriptors and receipts validate from a fresh full clone;
   independent review returns GO before #45 is amended.

### Validation

```bash
git diff --check
uv run ruff check benchmarks/v02_corpus.py tests/test_v02_corpus.py
uv run ruff format --check benchmarks/v02_corpus.py tests/test_v02_corpus.py
uv run mypy benchmarks/v02_corpus.py tests/test_v02_corpus.py
uv run pytest -q tests/test_v02_corpus.py tests/test_docs.py tests/test_manifests.py tests/test_v02_evaluation.py
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.v02_evaluation validate-protocol
uv run pytest -q
uv build
```

### Rollout

| Phase | Scope |
| --- | --- |
| S | Freeze spec and licenses. |
| B | Offline runner and tests. |
| C | Runtime preflight plus frozen training and sealed pilots. |
| D | Full isolated sealed capsule and annotation. |
| E | Full training corpus. |
| F | Amend #45 bindings. |

### Rollback

Revert public code/spec through Git; preserve private create-only evidence;
terminate cloud GPU only after digest-verified copy and provider deletion proof.

### Phase-S forbidden scope

- Implementation code, model download, GPU rental and raw generation before
  this spec is independently reviewed and merged.
- Any implementation outside the bounded Phase-B files and isolated ML extra
  after Phase S.
- Any live identity, prompt, threshold or count not frozen by the merged spec
  and manifest.
- Raw data in public Git.
- Changing #44 or release model architecture.
- Calibration, confidence thresholds, automation authorization.
- Candidate checkpoint training; this issue creates data only.
- Reuse or transformation of any blocked Phase 4E output.
