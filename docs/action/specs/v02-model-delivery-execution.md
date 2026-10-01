---
title: v0.2 executable path to private training and Model Preview
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/v02-model-delivery-execution.md
globalRef: qmd://saracura/docs/action/specs/v02-model-delivery-execution.md
reviewCadenceDays: 14
lastReviewedAt: 2026-10-01
sourceRefs:
  - github:#45
  - github:#55
  - github:#46
  - github:#47
  - github:#48
related:
  - docs/action/specs/v02-first-owned-checkpoint.md
  - docs/action/specs/v02-distillation-safe-corpus.md
  - docs/action/specs/v02-grounded-generation.md
  - docs/action/specs/v02-choice-quality-acceptance.md
  - docs/action/specs/v02-model-evaluation-protocol.md
supersedes: []
supersededBy: []
sensitivity: public
---
# v0.2 executable path to private training and Model Preview

Critique revision: deep-r3. This revision retains B5/B6 and M1–M7 corrections,
and closes r2 B7: loaded LoRA must also be active with frozen scaling and
produce a measured functional effect. N8–N10 clarify continuity, receipt creator
and typed reload measurement. No approved design decision is reopened.

## Outcome and authorization

This is an executable closure plan, not a statement of readiness. Its immediate
outcome is one genuinely trained, privately exported and BF16-reloaded Saracura
checkpoint. Its complete dependency map also covers selection, comparison and
packaging. Current authorization is to finish this specification and Claude
critique; implementation and GPU operations remain paused. Model weights,
corpus, metrics, comparison results and release announcements must not be
published until the human explicitly agrees what to disclose.

No successful fixture, helper test, preflight, source review, reserved GPU,
optimizer smoke or descriptor alone counts as a trained model. The private
milestone requires completed optimization on the full accepted cleanroom
training capsule, best-epoch export, integrity verification and fresh-process
inference using the exported weights.

For the private milestone only, this spec takes precedence over the
sealed-before-training ordering in `v02-first-owned-checkpoint.md`,
`v02-distillation-safe-corpus.md`, `v02-grounded-generation.md` (both pilots,
full sealed then full training), and `v02-choice-quality-acceptance.md` (full
sealed before full training). It does not weaken any rights, corpus floor,
independence, architecture or final evaluation requirement. The approved route
is full **training-lane READY**, then private training. Global #55 READY still
requires both lanes. A later final candidate is admitted separately.

## Verified baseline and why progress stalled

Source baseline is main `8a2d729aed609976f0aada93ec0622607d78d74a`.
Recheck upstream before implementation, and record any relevant delta.

| Component | Actual state | Remaining work |
| --- | --- | --- |
| Frozen #44 evaluation protocol | Merged, closed issue | Execute; protocol delivery is not evaluation evidence |
| Historical Phase 4E corpus | #53 closed BLOCKED_DATA_RIGHTS | Never reuse rows or synthetic holdout; hash-only exclusions remain a prerequisite |
| Cleanroom planner, ledger, reducers, role prompts, constrained author decoding | Merged through PR #65 | Physical validation and first-settled-attempt accepted corpus |
| C8 author/reviewer runtime R2 | Private source independently reviewed; no accepted physical run on replacement host | Bootstrap a fresh host and verify actual bindings |
| Regional recovery R3/R18 | Unaccepted proposals and failed reviews | Preserve; do not make another generic capture/activation framework a prerequisite |
| Training corpus | Zero accepted C8 rows, calls, reservations and events at pause | Micro 28, pilot 140, full 1600 planned identities; strict gates |
| Training capsule sealer | Spec-only; main CLI has no `seal-training` | Implement deterministic sealer and receipt verification |
| Trainer | Only on preserved unmerged training branch; main has no v02_training.py | Reuse selectively, correct loader, joint rendering and closed module targeting |
| Private admission/export schema | Spec-only | Implement and bind actual training evidence before model load |
| CUDA smoke/full run/export/reload | Not performed | Physical smoke, full c1, fresh BF16 reload |
| Final readiness/evaluate/select | Some branch commands are stubs | Implement only in final phase, retain #44 validation |
| Public v0.2 inference backend | Absent; existing universal checkpoint validator is MiniLM Phase 4E | New explicitly selected backend; reuse typed contracts, not incompatible descriptor |
| Packaging/results/release | Issues #46/#47/#48 open | Private staged package and exact aggregate evidence, then disclosure alignment |

The loop resulted from treating source approval as physical readiness,
assuming spec-listed commands existed, coupling first training to the whole
sealed evaluation, and attempting increasingly broad environment recovery.
The replacement host has no demonstrated old virtualenv or teacher cache;
the current provisioning script assumes both. These are known tasks below,
not discoveries to defer until a GPU is rented.

All paused source and drafts remain recoverable. The existing training branch
has seven dirty files and no final acceptance; retain its saved patch and
hash inventory. The two parent-spec amendments are draft-only, not accepted
because their executor reported PASS. Conform them to this spec in P0, including
frontmatter dates and explicit successor precedence. Do not revert unrelated
work or overwrite historical receipts.

## Fixed inputs and minimal architecture

The student is `Qwen/Qwen3.5-4B-Base` at
`1001bb4d826a52d1f399e183466143f4da7b741b`. Architecture is the frozen
Qwen 3.5 4B pointer head, dimension 256, temperature 1.0. Base tensors remain
frozen; only LoRA and query/key head weights are trainable and exported.
NF4 with BF16 compute is for training; BF16 with unmerged LoRA scale 1 is for
reload and eventual comparison. No quantized reload substitutes for BF16 proof.

Frozen candidates are c1 r8/alpha16/dropout .05/lr2e-4, c2
r16/alpha32/dropout .05/lr1e-4, c3 r32/alpha64/dropout .05/lr5e-5.
Seed 20260929, AdamW weight decay .01, warmup .05, cosine schedule, gradient
clip 1, effective batch 32, up to 5 epochs, patience 2, best internal-dev
stratified macro accuracy, earliest epoch on tie. Private identity is
`private-c1-r8-v1`, distinct from final Phase E c1. No fourth candidate,
hyperparameter search, selection on sealed test or new calibration project.

The only renderer is pinned Kev source revision
`9c41005b2180347c3c646dfc9e50c4428483ec6b`, raw SHA-256
`d78fab645f29513a62816e594d10296b02bf166ba77835c55db5eda80c7f1978`.
Use its source functions, no Kev checkpoint tensors. Preserve state strings
as strings. Inputs use the exact joint causal row with `strict=true`,
`option_isolation=false`, max_state384/max_branch1024/max_packed2048 and
the frozen maximum 512 rendered tokens, no truncation. The head query is at
the row's decide index and each key at that same forward's ordered end-option
index. Separately encoding state and option branches is incompatible.

The actual meta-model inventory has 248 target Linear modules. Close every
name and shape against the pinned actual model; each LoRA rank has 496 adapter
tensors. Trainable LoRA counts are 16,232,448 / 32,464,896 / 64,929,792;
head query/key are each [256,2560], total head parameters 1,310,720. No
`all-linear` alias is accepted as an inventory. Validate semantic text config
values, not Python object identity against a config that the loader copies.

Teacher roles/pins remain those in the merged cleanroom policy: Qwen 9B
author and Mistral Small 24B reviewer for training; Phi mini author, Granite
and OLMo blind annotators, SmolLM adjudicator for sealed. Preserve actual
license evidence for all roles. Only author/reviewer weight downloads and
execution are necessary before private training; sealed prompts, plans,
model revisions and policy digests are frozen before training. Sealed
tokenizer/schema metadata must be included if existing preflight consumes it.
Do not claim that current six-role preflight already supports two weights.

No new scheduler, infrastructure abstraction, Kubernetes service, generalized
receipt framework, public hosting or deployment automation enters this path.
Use existing ledger/reducers, one owned GPU at a time and a trusted operator's
direct API/SSH observations. Source-bound evidence remains required.

## Phases, ownership and exit conditions

`P0 -> {P1,P2,P3} -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> STOP`.
P1/P2/P3 are independent source scopes and must all pass before P4. P8/P9
are outside the immediate private-training milestone; their dependencies are
specified now. Each code phase uses canonical SDD execution, conform, review,
targeted tests and final root QA. One dispatcher/session owns each phase;
no worker launches another orchestrator. A review of source never clears a
physical exit criterion. Keep one status record in existing GitHub issues and
private platform evidence, not a second roadmap.

### P0 — Freeze the implementation contract, no GPU

Owner: orchestrator. Scope: this spec, the two parent-spec amendments and
private evidence mappings; public docs have only generic logical roots.
Exit: genuine Claude GO with all blockers resolved, docs valid, draft
precedence coherent, preserved source inventory and phase contracts linked.
Before later implementation, integrate accepted source spec normally; a docs
merge alone cannot advance P1–P7. Final comparison protocol remains unchanged.

### P1 — Implement the deterministic training capsule sealer

Owner: corpus executor. Scope: `benchmarks/v02_corpus.py`, additive sealer
tests, relevant corpus spec interfaces. New CLI `seal-training --root ROOT
--plan PLAN --receipt RECEIPT --exclusions HASHSET --output-parent OUT
--artifact-id ID`. All arguments are explicit; roots never discovered.
Existing commands `validate-protocol`, `plan`, `verify-receipt` remain intact.

Read accepted identities from actual ledger reduction and first settled author
and reviewer events. Re-read their private case files and verify source hashes,
accepted status, planned split/locale/domain/cardinality/gold, renderer/tokenizer,
privacy, no duplication and every existing quality gate. Labels come from
accepted agreed planned gold, not renderer admission's fixed label zero.
Join only by exact frozen identity; incomplete/dangling/mismatched records fail.
Run `verify_receipt` against the same plan and event bytes; a string READY is
insufficient. Use existing canonical aggregate receipt v3 rather than inventing
a competing acceptance reducer. Emit in sorted frozen identity order.

Capsule must contain >=1200 accepted, train>=1020/dev>=180, PT-BR>=60%,
English>=20%, each train locale×cardinality>=20 and dev>=10, train
domain>=60/devdomain>=10, >=120 complete bilingual pairs, no split leakage.
Historical hash-only exclusion input must be validated against its existing
source/manifest/Phase A receipt digests, schema and normalization; pin its raw
SHA before accepting a capsule. Compare all four fingerprint axes: identity,
state-question and combined-content must have zero overlap under AC4;
option_multiset is descriptive, not an additional rejection gate. Never
open historical raw rows/holdout. Missing or unverifiable exclusions produce
BLOCKED_DATA_RIGHTS, not an empty exclusion set.

Output schema and ownership are specified below. Ownership 0700 root/0600
files, no symlink, no repo/worktree/sync ancestry, create-only atomic fsynced
publication. Exit: self-authored ledger fixtures exercise real reduction ->
sealing -> verification, rejected tampering/overlap and exact floors; no real
corpus claimed. Physical capsule acceptance is P5.

### P2 — Complete the minimum trainer and private inference proof

Owner: trainer executor. Scope: reuse `benchmarks/v02_training.py` and
training tests, additive `pyproject.toml` extra/lock, manifests only where
required to close actual tensor inventory; preserve existing extras and #44
validation. No evaluation or seal-readiness stub is accepted as delivered.

Commands NEW/branch-incomplete: `validate-private-admission`,
`smoke-private`, `train-private`, `verify-private-checkpoint`,
`reload-private`. Inputs are explicit paths to verified admission/capsule/base
snapshot/output root; model load is impossible before admission verification.
These are minimal checkout-only commands; do not pretend they are installed
today or reuse a final candidate descriptor as private eligibility.

Construct one joint row per accepted case, int64 token/index/label tensors,
bool attention mask, ordered option indices, real gold index and stratum.
Reject token overflow/invalid index/cardinality or missing label, never truncate.
Use actual Qwen text submodule, pinned tokenizer and closed module inventory;
PEFT must cover exactly the expected targets/tensors. Ragged batches mask padded
options out of loss and argmax; BF16 head inputs must match dtype. Only owned
parameters enter the optimizer; validate nonzero finite updates and unchanged
base. Seed before any trainable initialization. Best state copied independently
to CPU, not a reference mutated by later updates. Model/config caches are
explicit; no network access during training or request inference.

Exit: meaningful self-authored tiny causal CPU optimize/export/fresh-reload
test proves gold handling, same-forward indexing, finite gradient/update,
frozen base, dtype, ragged masking, best-state retention and create-only
export. Actual pinned meta/config/tensor tests prove architecture. Tiny/random
models are tests only. Run regression coverage for affected evaluation,
manifest validators and optional-install boundaries. Full suite only after
targeted passes, with owned SSD TMPDIR to avoid the observed ENOSPC failure.

### P3 — Make fresh-host runtime prerequisites explicit

Owner: private runtime executor. Scope: private R2 runtime bundle and a
create-only derived version; public source only for measured lane defects.
Reuse reviewed R2 source, source archive verifier, ledger, lane locks and
status reducer. Do not overwrite R2/R3 or complete R18 generic automation.

Before spend, produce a bootstrap bill of materials with pinned container,
Python/dependency wheel/hash inventory, source archive+18 runtime consumers,
four student tokenizer files, renderer, teacher license/tokenizer metadata,
download commands and disk requirements. Bootstrap creates the virtualenv
and model directories; it cannot assume the old pod's files exist. Derive a
training-only preflight mode that preserves all rights/tokenizer/prompt/source
checks and does not load four sealed-role weights. Test exact consumption:
existing preflight compiles schemas for six roles, so either retain their
metadata/tokenizers or narrowly split its role list. All changes are reviewed
with host-local patch and private tests before deployment.

Existing `run-pilot-role.py ROLE COHORT` and `status training` are private
interfaces. Freeze their actual ROLE_ORDER/COHORTS from reviewed source in
the private execution mapping; wrappers and grants must allow exactly the
requested cohort after its predecessor passes. No fictional public `generate`
command is a prerequisite. Existing status verifies only training lane;
ensure new receipts distinguish it from global #55 readiness.

Resolve hardcoded old region and old physical activation bindings through
one source-reviewed change using actual observed replacement region; never
fabricate previous approval evidence or blanket-enable environment grants.
Use trust in authenticated operator observation, not a self-asserted
`authenticated=true`. Preserve separate UIDs/0700 roots/credentials even
while only training roles execute. Exit: offline tests demonstrate clean-host
bootstrap dependencies, training-only role admission, role/cohort locks,
immutable source closure, stale host rejection and no sealed mount to trainer.

### P4 — Bootstrap and physically admit one owned host

Owner: operator/orchestrator. Inputs: all accepted P1/P2/P3 receipts, actual
cloud rights/provider/region/retention/deletion authorization, sufficient
funds and disk. On fresh authenticated API read record pod id, actual region,
image digest, startedAt and direct SSH endpoint. Never derive these from
name or nonexistent lastStartedAt. SSH verifies actual source/archive/runtime
hashes, OS/dependency versions, kernel boot id, GPU UUID/VRAM and filesystem.
Record combined identity before and after bootstrap; restart/change invalidates
physical admission. No role dispatch before both observations pass.

Run pinned bootstrap, download exact permitted author/reviewer weights and
student snapshot, verify license/inventory, loopback native schema constrained
server, lane credential/ancestry isolation and read-only preflight. Download
time and disk use are part of the budget. A stopped pod's native SSH null is
not an observation of live readiness. GPU >=24GiB; use existing compatible
80GiB route for teacher serving, not final systems measurement.

Exit: actual clean-start runtime works and training-only preflight passes with
zero model generations/reservations; write a real private physical receipt.
Stop host after bounded failure or idle source repair. Capacity strategy:
one restart of the owned compatible pod, then one checked alternate compatible
region/host with new rights/identity receipt. Exhaustion -> CAPACITY_UNAVAILABLE;
do not iterate provider frameworks or silently change architecture.

### P5 — Generate, review and seal the full accepted training corpus

Owner: operator with reviewed private driver. Run author then blind reviewer
for micro28, pilot140, then remaining planned identities; nested cohorts
settle the same identities once, never regenerate pilot rows in the full run.
Use `grounding_micro_pilot` and current reducer's exact frozen micro thresholds;
do not invent substitute micro acceptance. Pilot requires >=106/140 total,
>=53/70 per locale, >=15/20 each cardinality. The frozen full plan is
1360 train +240 internal_dev, two locales, 12 domains, option counts2..8.
Choice-agreement-and-all-gates.v2 requires author/reviewer/planned gold agreement
and every existing first-attempt gate. Diagnostics equality is not required.

No retry/replacement/prompt tuning under the same plan. Every reservation is
terminal or unresolved exactly as current ledger specifies. Mathematical
feasibility gate compares accepted plus unresolved against every remaining
floor before each dispatch; impossible -> terminal NO_GO and halt. Old C7
NO_GO remains historical. C8 cannot borrow old rows, reset receipts or recycle
identities. Driver transport failure cannot be interpreted as no dispatch.

Measure author/reviewer load time, latency, accepted yield and peak disk over
micro/pilot. Before full run project remaining wall time/cost conservatively
from actual role throughput and remaining identities, including model switching,
student smoke/full training/reload, storage and measured bootstrap. Record a
range and available runway; if insufficient -> FUNDING_REQUIRED, stop paid
runtime and ask for funding decision, not billing changes. No guessed timeline.

After full lane READY, create verified aggregate v3 receipt as training UID2001
via the existing training-user wrapper, then invoke P1
sealer. Recompute capsule hashes/counts and private backups on independent
owned storage; verify copies. Exit: real immutable accepted >=1200 capsule,
all floors/rights/exclusions proven. Do not mark #55 globally complete while
sealed lane is pending. Failure of quality is terminal evidence, not motivation
for endless prompt correction. A successor corpus policy requires explicit
design change, preserved failed evidence and new identities.

### P6 — Admit data and run bounded real CUDA smoke

Owner: operator with P2 code. Build admission from actual P5 capsule, verified
licenses/cloud authority, frozen sealed plan/prompt/model-role digests, source
commit and code bytes, base/tokenizer/renderer and closed inventory. No
placeholder digests or sealed payload/descriptor references. Existing native
physical admission is rechecked before model load. Trainer UID cannot access
sealed UID/root. Download student once explicitly if P4 cache is incomplete.

Smoke uses a deterministic bounded subset of train/internal_dev, <=32 train
rows and one optimizer step, export+BF16 fresh reload+Choice. It is tagged smoke
and cannot be the private full checkpoint or consume private-c1-r8-v1 identity.
Validate real peak VRAM, finite loss/gradients, actual updated LoRA/head and
unchanged base, complete tensor set and typed output. Project P7 time/cost from
measured step/load/reload times and number of actual batches, noting early-stop
uncertainty. Exit: real pinned CUDA smoke and runway sufficient for full run.

### P7 — Full private c1, export and fresh BF16 inference

Owner: operator with P2 code. Execute fixed optimization against only the full
accepted training and C8 internal_dev splits. Report train rows consumed,
epochs/steps, best epoch/internal-dev criterion, finite losses, parameter
updates, base freeze and measured duration/resources. Internal dev selects
epoch; it is neither #44 public_dev nor sealed evaluation.

Emit private checkpoint schema below using create-only adapter/head/config
files. Verify descriptors against actual safetensors names/shapes/dtypes/hash
and reject base/optimizer/foreign tensors. Start a new process, load exact base
in BF16 and unmerged exported adapter/head, and infer a deterministic subset of train and internal_dev examples,
with the ordered identity digest recorded in sample_identity_digest. Preserve raw state-string semantics through the
actual `DecisionRequest` contract: wrap serialized case state as a string
value in state.serialized_state, with checkout-only workflow
`universal-choice@v02-private-qwen-pointer.v1`. Its renderer reads that literal
member, maps each instruction/ordered criteria, and produces exactly the same
case row as training; do not parse the string as JSON. Map ordered option IDs
back to ChoiceCriterion IDs, finite raw scores/normalized ranking weights and
`DecisionResponse`. Status uncalibrated, abstained=true,
reason=uncalibrated_research, calibration=null, automation_allowed=false.
No test-set access. This checkout-only proof does not install a runtime default.

After fresh load, compare every live LoRA and head tensor with the exported
tensor's value after the declared BF16 cast; zero missing or unexpected keys
and exact name/shape/dtype inventory are mandatory. Traverse actual adapter
parameters rather than trusting a non-strict load return or success flag.
Record the compared inventory hash and matched tensor count. Prove these
checks reject dropped adapter keys, mismatched prefix or unapplied head on the
pinned Qwen meta/inventory test; exercise actual values on the real CUDA run.
Verify active adapter names equal exactly the exported adapter name. Every
target module must have adapters enabled and effective scaling alpha/r from
the frozen config (c1=2); no merged adapter or changed config is accepted.
Require these live checks, not an export-config declaration alone. On the same
deterministic subset (up to32 internal_dev identities in frozen sorted order),
compare active-adapter scores with scores while only LoRA is disabled and
the same trained head remains applied, in eval mode with identical inputs.
At least one valid-option score must differ by more than
`1e-6 + 1e-5 * abs(disabled_score)`; record maximum absolute difference and
functional_effect_verified=true. If no score differs, fail
IMPLEMENTATION_BLOCKED, not a quality-floor failure or permission to tune.
Restore exact active adapter state before further inference. Negative tests
reject inactive adapters, wrong alpha/r and a no-effect adapter forward.
These checks prove exported adapters participate; they do not claim quality.

Compute full internal_dev stratified macro accuracy through that same typed
DecisionRequest/renderer/criterion-ID adapter in BF16 after reload and
record it alongside NF4 best epoch accuracy, descriptive only, no new floor or
selection. Tensor/activation/functional checks reject a silent base-only
reload; this descriptive metric reports the precision/runtime behavior.

Exit: actual completed full optimization + immutable export + verified fresh
BF16 typed inference, two verified private copies and complete receipt.
This is the first **model trained** milestone. Keep issues #45/#55 open with
remaining final gates explicit; no final selection receipt, public score,
comparative claim or model upload. Publish neither metrics nor weights.

### P8 — Separate final admission, selection and one-time comparison

Owner: orchestrator/final evaluation executor. Scope: missing final training
commands, existing `benchmarks/v02_evaluation.py`, additive tests and #44/#45
readiness evidence. Before candidate training reopen final Phase B admission
under the frozen protocol: full independent sealed generation/annotation/
adjudication/rotation/disjointness/seal-test, aggregate descriptor and readiness
manifest merged before final Phase E attempts. The trainer sees no sealed
payload. Training-only private admission cannot replace this requirement.

Implement missing `seal-readiness`, real `smoke`, candidate train/export,
public-dev reports, `select`, private evaluator and one-time systems driver;
reuse already functioning validation rather than duplicating it. Four sealed
role execution/sealer are missing physical deliveries; existing role driver
and reduce_sealed are reused with frozen plan140/pilot21, accepted>=100 and
>=14/cardinality, two blind annotators plus adjudicator, frozen rotation/subset.
No training-aware prompt or plan changes after P7; keep roles/source-independent.
If a frozen policy cannot yield its cohort, terminal NO_GO, not relaxed floors.

Private identity is not automatically promoted into final c1. Execute the
three final frozen candidate identities under fresh final admission exactly
once. This deliberately incurs a separate c1 run to avoid ambiguous lineage;
no promise of promoting the private artifact. Existing #44 validator reads
selection receipts, not private checkpoint schemas; separation is an explicit
operator/review policy, not an invented automatic validator rejection.
Only public_dev chooses the final candidate, never private internal_dev metrics
or sealed results. Follow existing candidate attempt/code revision rules.

After select immutable winner, exactly one authorized sealed evaluator session
per frozen policy; use the same selected bytes and controls/inputs/denominators.
Quality and systems on dedicated Linux RTX4090 24GiB, CUDA BF16, unmerged
LoRA1, loopback Python HTTP, batch1/concurrency1. Candidate/Kev are the frozen
systems group; Laya/Julia are descriptive controls. Ten warmups then100 measured
sequential calls, exact deterministic50-row subset, p50/p95 nearest rank,
throughput, peak RSS/device, cold startup and artifact bytes; multi-decision
scenario remains explicitly measured with no encode-once claim for causal
pointer architecture. Accuracy>=.40,coverage>=.98,invalid=0,repeat=1,
order>=.95; otherwise NO_RELEASE and preserve negative aggregate evidence.
No retuning/reopening sealed payload because packaging reveals a weakness.

Exit: closed final selection receipt, one-time evaluation receipts, verified
aggregate/dev separation and bounded draft claim or no advantage claim. #47
publishes already accepted evidence later; it never reopens test payload.
Private P7 success stands even if P8 yields NO_RELEASE.

### P9 — Stage a usable private package; stop before disclosure

Owner: packaging executor. Scope: new explicit opt-in v0.2 backend and tests,
optional dependencies, reviewed download/install command, package docs/model
card and private staged distribution. Reuse `contracts/models.py` and engine
registry; existing `universal/checkpoint.py` MiniLM shapes cannot load Qwen.
Add a separate v0.2 closed descriptor loader with verified LoRA/head inventory,
same joint renderer and workflow contract; no default backend switch until
release scope explicitly allows it. Default dev/test remains lightweight.

Artifacts: adapter/head only, tokenizer assets or explicit pinned legal download,
base reference/license, exact source/model revisions/digests, model card PT-BR
Choice2..8/512tokens, hardware requirements, training source classes,
uncalibrated limitations/failure modes. No raw rows/prompts/private paths.
Provide one documented command that explicitly installs/downloads pinned assets
and runs one self-authored PT-BR example from a clean environment. Request
inference is fully offline, no remote fallback/download/telemetry. Missing or
corrupt snapshots fail before inference with actionable error. No confidence
or automation claim. Package artifact bytes must equal P8 benchmark bytes.

Exit: clean isolated environment can install private staged wheel/model bundle
without Felhen/private services and produce valid typed output; digests match
accepted final candidate; draft English/PT-BR front doors, model card,
release notes and aggregate claim are reviewable. Validate new backend tests,
optional dependency isolation, `uv build`, package install and offline run.

**STOP:** present exact disclosure bundle, destination plan (HF/GitHub/PyPI),
license and proposed claim to human. No uploads, public results, release,
benchmark/community submission or launch announcement before alignment.
After explicit approval, #46/#47/#48 govern publication, HF revision/digests,
PyPI quickstart, release, independent JevBench and two available index
submissions; external acceptance is not a release blocker. #50 calibration is
deferred. The release clock starts with a selected eligible package, not with
unfinished corpus infrastructure.

## Closed private artifact contracts

Implement these additive schemas in P1/P2, not a parallel generic framework.
All objects forbid unknown keys; IDs nonempty logical identifiers, SHA fields
lowercase64hex. RFC8785 canonical object hashes and raw-file hashes are
explicitly distinct. Relative paths cannot traverse or escape verified root.
Numbers are finite; booleans cannot replace integers; no null future digest.
Nested inventory is sorted unique `{path,sha256,bytes}`; tensor inventory is
sorted unique `{name,shape,dtype}`. Hash referenced actual bytes on every load.

| Schema | Exact top-level keys and meaning |
| --- | --- |
| `v02-cleanroom-training-capsule.v1` | `schema_version`, `artifact_id`, `plan_sha256`, `events_sha256`, `aggregate_receipt_sha256`, `corpus_source_revision`, `policy_sha256`, `renderer_sha256`, `tokenizer_inventory_sha256`, `exclusions_sha256`, `rights_receipt_sha256`, `rows_file`, `rows_sha256`, `counts`, `ancestry_sha256` |
| capsule row | `identity_id`, `split` (train/internal_dev), `locale`, `domain`, `state` (raw string), `instruction` (string), `options` (ordered array of `{id,description}`), `gold_index` (integer0..N-1), `case_sha256`, `author_event_sha256`, `reviewer_event_sha256`, `bilingual_pair_id` (string or null) |
| capsule counts | `accepted`, `train`, `internal_dev`, `locale` (exact pt_br/english counts), `train_locale_cardinality`, `dev_locale_cardinality` (each exact locale key -> keys2..8 counts), `train_domain`, `dev_domain` (exact frozen12 domain keys), `complete_bilingual_pairs` |
| `v02-private-training-admission.v1` | `schema_version`, `artifact_id`, `private_identity`, `capsule_sha256`, `aggregate_receipt_sha256`, `rights_receipt_sha256`, `license_inventory_sha256`, `environment_receipt_sha256`, `corpus_source_revision`, `trainer_source_revision`, `trainer_code_inventory_sha256`, `base_id`, `base_revision`, `base_inventory_sha256`, `tokenizer_inventory_sha256`, `renderer_sha256`, `module_inventory_sha256`, `candidate_config_sha256`, `renderer_preflight_sha256`, `sealed_plan_sha256`, `sealed_prompt_inventory_sha256`, `sealed_model_inventory_sha256`, `seed`, `training_authorized`, `private_only`, `publication_authorized` |
| `v02-private-checkpoint.v1` | `schema_version`, `artifact_id`, `private_identity`, `admission_sha256`, `run_receipt_sha256`, `base_id`, `base_revision`, `base_inventory_sha256`, `tokenizer_inventory_sha256`, `renderer_sha256`, `candidate_config_sha256`, `tensor_inventory_sha256`, `files`, `private_only`, `evaluation_pending`, `publication_authorized`, `automation_allowed` |
| private run receipt | `schema_version` (=v02-private-training-run.v1), `artifact_id`, `admission_sha256`, `private_identity`, `seed`, `epochs_completed`, `optimizer_steps`, `train_rows`, `internal_dev_rows`, `best_epoch`, `best_internal_dev_macro_accuracy`, `finite_updates`, `base_unchanged`, `initial_trainable_sha256`, `exported_trainable_sha256`, `duration_seconds`, `peak_gpu_bytes`, `reload_receipt_sha256` |
| private reload receipt | `schema_version` (=v02-private-reload.v1), `artifact_id`, `admission_sha256`, `exported_trainable_sha256`, `base_inventory_sha256`, `tensor_inventory_sha256`, `precision` (=bf16), `lora_merged` (=false), `sample_identity_digest`, `samples` (positive integer), `finite_scores`, `typed_contract_valid`, `matched_tensor_count`, `loaded_tensor_inventory_sha256`, `missing_tensor_keys` (empty array), `unexpected_tensor_keys` (empty array), `loaded_values_match_export` (=true), `active_adapter_names` (one exact export name), `adapters_enabled` (=true), `module_scaling_inventory_sha256`, `scaling_matches_config` (=true), `functional_sample_identity_digest`, `functional_tolerance` (=atol1e-6_rtol1e-5.v1), `functional_max_abs_difference` (finite positive number), `functional_effect_verified` (=true), `reload_internal_dev_macro_accuracy` (finite0..1), `training_best_internal_dev_macro_accuracy` (finite0..1), `automation_allowed` (=false) |

Admission flags: training_authorized/private_only=true,
publication_authorized=false. Checkpoint: private_only/evaluation_pending=true,
publication_authorized/automation_allowed=false. Private receipt references
noncyclic: export weights -> reload receipt -> run receipt -> checkpoint
descriptor. Reload binds weights/inventory, not yet-written checkpoint SHA.
Capsule/rights/ancestry/environment schemas already governed by corpus specs
and runtime are referenced unchanged; their digest alone does not authenticate
authority. Verify their actual provenance and existing closed validators.
Trainer receives verified training inputs and only frozen sealed metadata hashes.

## Source-root ownership, backups and digest mechanics

B5 closure is intentionally colocation, not a new authenticated transport
system. P5 sealer and P6/P7 trainer run as the same training UID (2001 on the
reviewed host) on the same owned filesystem. The original aggregate receipt
remains at the original receipt parent, whose actual ancestry is its binding;
the training ledger/cases remain at their original root. The trainer has read
access only to this training material and its own outputs, never sealed UID2002
root. CLI inputs name original receipt, frozen plan and verified events, not a
digest-only copy. Admission verification invokes existing verify_receipt as
that UID, recomputes the reducer and sealer's row/gold/floor checks against
original files before model load. It binds the actual receipt-parent ancestry
and capsule root separately. P1/P2 tests explicitly run the owner/path checks;
copied receipt, different UID or ancestry mismatch are rejected.

Backup copies in P5/P7 are byte-verified archival copies only. Verify their
raw file SHA inventory and permissions; do not invoke verify_receipt on a copy
or claim it is a newly admitted live corpus. Keep original receipt bytes and
binding evidence unchanged. Stopping/restarting a host is allowed only if
actual receipt/corpus device-inode ancestry remains valid and fresh physical
observation re-admits the environment. Never rewrite old ancestry fields.
Host loss or migration that invalidates original ancestry is explicitly
IMPLEMENTATION_BLOCKED for this route; no automatic import or hidden
transport prerequisite. Preserve backups. Recovery would require a separately
reviewed import/re-attestation increment against original ledger/plan/source
proof, with a new destination receipt, before training resumes. This is a
declared contingency and not part of the minimal critical path.

`ancestry_sha256` in the capsule is SHA over RFC8785 canonical bytes of the
original aggregate receipt ancestry object. Descriptor's capsule-root ownership
and original receipt-parent are both checked live by admission; identical
digests alone do not authenticate their origin. renderer_preflight_sha256 binds
a closed self-authored/new private report with schema_version
v02-renderer-preflight.v1, artifact_id, capsule_sha256, candidate_renderer_sha256,
kev_rendering_contract_sha256, accepted_records (exact capsule count),
candidate_all_records_pass=true and kev_all_records_pass=true. Recompute both
preflights for all rows; do not hash an unverified claimed boolean.

Capsule row locale literals remain pt_br/english from the source planner and
reducers; only DecisionRequest locale maps them to pt-BR/en. Sealer maps case
question -> row instruction losslessly and leaves raw state unparsed.
Micro thresholds are >=24/28, >=12 per locale and >=3 per cardinality, exactly
the current GROUNDING_MICRO_PILOT constants. Current role dispatch literals
are training_author and independent_reviewer, with micro/pilot/full cohorts;
private wrappers/grants and exact commands remain in the execution mapping.

Define exported_trainable_sha256 as SHA over RFC8785 canonical bytes of the
sorted inventory of the adapter/head safetensors files only, entries
{path,sha256,bytes}; each sha256 is raw file bytes. Checkpoint files inventory
must contain those same entries plus owned config/licenses, no optimizer/base.
Initial trainable digest uses SHA over a canonical name/shape/dtype/value-byte
hash inventory before any update. Loaded comparison uses exported values cast
to the declared runtime dtype, never safetensor file bytes compared to runtime
parameter bytes. Bind all compared keys to tensor_inventory_sha256 and the
same count; r8 adapter496 + head2 =498 matched tensors. Finite-value and
unchanged-base checks remain mandatory. No descriptor/run/reload digest cycle.

## Failure, resume, cost and discovery limits

| Condition | Permitted action | Terminal/stop boundary |
| --- | --- | --- |
| Missing local prerequisite before spend | Complete the exact P1/P2/P3 item, recheck only affected gate | No GPU while repairing source |
| Same criterion fails twice | Classify evidence, split phase or change bounded strategy; consolidate correction | No third identical retry |
| Auth/provider transient before dispatch | Existing operational fallback once, observe actual status | No billing/auth settings changes |
| Capacity unavailable | One compatible restart then one verified alternate, new receipt | CAPACITY_UNAVAILABLE |
| Reservation exists, response unknown | Preserve reservation and investigate actual ledger/transport; follow frozen settlement | Never blindly redispatch same identity |
| Micro/pilot/floor impossible or invalid teacher output | Current reducer decides NO_GO; preserve all evidence | No reset/replacement/tuning same plan |
| Invalid license/cloud rights/exclusion source | Correct authoritative evidence if available | BLOCKED_DATA_RIGHTS before upload/train |
| Source/host binding mismatch | Stop dispatch, derive new source bundle and independent physical proof | Old physical receipt never copied as new |
| Disk/cost insufficient | Before run, resize only within authorized scope or stop; preserve files | No deletion of ambiguous data; funding decision if needed |
| OOM on smoke | Use fixed microbatch1 plus accumulation32 if valid frozen effective batch; clear owned process | If still insufficient, compatible larger GPU; no architecture/data truncation |
| Full run interrupted | Resume only exact same code/base/data/config/seed/optimizer/RNG/step bytes with private state | If no valid state, IMPLEMENTATION_BLOCKED; no silent fresh private identity retry |
| NaN/infinite or unexpected tensor/base update | Stop and preserve implementation evidence | No LR/rank/seed tuning; completed identity never retried |
| BF16 reload/export fails | Fix measured exporter/loader only, verify immutable original weight bytes | Different learned weights require new reviewed design/identity |
| Final frozen quality floors fail | NO_RELEASE, preserve private model and negative evidence | No claim or test reopening |

Private optimizer resume state is a separate 0600 operational artifact, never
part of exported weights. Same run binding includes source inventory, data,
config, base, initialization seed, optimizer/scheduler, RNG states and step.
Write atomic step checkpoints only as needed for long-run interruption; no
generic resumable distributed training system. If code changes alter actual
training, an old state is not resume-compatible. Separate source revisions:
corpus generation, trainer and final admission, never a single ambiguous SHA.

Refresh funds/rate/storage before any paid operation. Last observed spend was
zero while paused; balance is not an authorization to fund account. Report each
additional USD10 crossing privately. No invented full training duration: first
estimate after pilot, second after real CUDA smoke. Keep a measured remaining
cost reserve for reload and backup. Before P5, stop owned paid host during lengthy source repair; no unattended
bill while critics work. From first P5 ledger event through P7 export and
backup, default to one continuous host session: parent ancestry includes
container overlay parents and a stop/start can invalidate it. Reserve that
continuous cost in P5/P6 projections, including bounded model switches and
reload. P4 may perform one empty-directory stop/start ancestry probe before
any generation; stable actual parent/root ancestry is required to permit
later planned pauses. If not measured stable, do not assume it. Critical
source repair after P5 either fits the measured runway or stops with preserved
backup and IMPLEMENTATION_BLOCKED requiring the declared recovery increment. Record all actual costs, even
failed bootstrap. Hard constraints are available funds and frozen cohort gates.

Each phase has a fixed admission checklist above. Newly discovered issues must
name phase/criterion, actual evidence, classification, corrective scope and
effect on remaining cost/path. Blocking is limited to evidence required for
that phase; later packaging or generic automation cannot block private training.
An uncertainty has one bounded probe and a declared stop branch, not an
unlimited infrastructure project. Update #55 for corpus/host, #45 for trainer,
#46/#47/#48 for later deliverables; do not close an issue for partial scope.

The private `platform.md` records current phase, source delivery vs physical
proof, accepted rows/calls/training steps, last actual cloud observation,
blocker/next action and real receipt paths. Public issue comments contain
sanitized technical state and spec links, no raw data/provider body, private
path/host/account identifiers or undisclosed metrics. Exact execution paths,
role/cohort literals, hashes, provider identity and receipts are mapped in
private evidence before any command. No secrets in stdout, argv, logs or Git.


## P1 explicit sealing inputs — execution handoff completion

This additive input contract closes the P1 handoff gap exposed by execution:
the seven original CLI arguments cannot supply rights/renderer/tokenizer
provenance required by the closed capsule. Add required `--sealer-inputs FILE`;
existing original arguments and all frozen gates remain. No new rights policy,
authentication framework, corpus lane or inference architecture is introduced.

FILE is an owned regular0600 canonical JSON under the original training root,
created by the trusted data operator after real P4/P5 evidence, never a
self-authorizing READY flag. Exact top-level keys of
`v02-training-sealer-inputs.v1` are `schema_version`, `artifact_id`,
`corpus_source_revision`, `policy_sha256`, `environment_grant_file`,
`environment_grant_sha256`, `license_files`, `runtime_lock_file`,
`runtime_lock_sha256`, `renderer_source_file`, `renderer_source_sha256`,
`tokenizer_directory`, `tokenizer_inventory_file`,
`tokenizer_inventory_sha256`, `exclusions_sha256`,
`historical_phase_a_receipt_file`, `historical_phase_a_receipt_sha256`,
`historical_source_sha256`, `historical_manifest_sha256`.
File/directory references are safe relative paths beneath the explicitly named
training root; license_files has exactly the six MODEL_ROLES keys and one
relative actual license-file path each. No symlink/escape, unknown keys, null
digest or booleans as numbers. Source revision is40hex; digests64hex.

Use existing `v02-environment-grant.v1` as the rights receipt, preserving every
existing validator and mandatory license/distribution/cloud/retention flag.
Distribution-rights clearance is legal capability, not current disclosure
permission; checkpoint publication_authorized remains false. Rehash actual
grant bytes and all six license files, invoke `validate_environment_grant`,
and bind the resulting canonical grant digest to original ledger's existing
environment-binding, both training role bindings and every accepted-role
reservation. Never approve content by trusting the new input file alone.
Rehash runtime lock bytes, require its canonical digest to equal grant's
runtime_lock_digest, and its public_source_integrity.source_commit to equal
corpus_source_revision. Require source inventory/archive digest fields in that
existing source proof and its nonzero files_verified; actual source closure
was independently observed/reviewed before generation, not fabricated by sealer.

policy_sha256 is SHA256 over RFC8785 canonical JSON bytes of the literal
TRAINING_ACCEPTANCE_POLICY identifier string already bound by aggregate v3. The capsule rights_receipt_sha256 is raw SHA of
the unchanged environment-grant file, and policy_sha256 binds that exact frozen policy identifier, not a new object. Read the pinned renderer bytes from the named file,
verify exact source hash before `load_pinned_renderer`; load verified tokenizer
from named directory/inventory via existing factory and check inventory equals
grant tokenizer inventory canonically. Recompute both required renderer
preflights for every accepted case; no claimed booleans accepted.

Pin/recheck raw exclusions_sha256, closed historical exclusion schema/source
bindings and original historical Phase A rights receipt metadata (not raw rows).
Require raw Phase A receipt hash to equal exclusion source's
phase_a_receipt_sha256, and its existing packet_manifest_sha256 to match historical_manifest_sha256.
The Phase A receipt has no source-row hash field: require exclusions metadata
historical_source_sha256 and historical_manifest_sha256 to equal the exact
trusted operator pins in this input manifest, derived from the independently
reviewed exclusion producer/source binding. Do not invent a Phase A field or
read historical raw rows to reconstruct it. The known historical block remains blocked; these hashes
are exclusions only, never historical data authorization. Hash-set type/count,
normalization, literal-string alias sets and three mandatory zero-overlap axes
remain required. Fourth option axis is descriptive; absent historical fourth
set is explicitly unavailable, never invented zero overlap.

Private fixtures may construct self-authored valid grant/renderer/tokenizer/
events to exercise real validators, clearly test-only; they do not confer
operational authorization. CLI and reusable verify_training_capsule API must
reject changed rights/source/tokenizer/renderer/reservation bindings and any
missing actual input before create-only publication. P2 receives this verified
capsule API once P1 is integrated. No raw historical rows/holdout are read.


### P1 immutable historical authority and actual runtime-lock shape

PB-1/PB-2 correction is bounded to the prior input delivery delta. The
historical exclusion artifact is NOT authorized by arbitrary operator pins.
Require the code/spec-frozen historical packet manifest hash
3e5dccc8bb551cf4840046b20712a52c9409c9424f20333185f3072e8dd6c239,
historical accepted-train-dev source hash
bc444c3a233e8e3397ff35a38360a117caa2bbe9e2bcaa69b8b7b77cb384018e,
original #53 Phase A receipt hash
75b4fa2c3758610a1e494360a8bd1236a76a2d7f841e998ad0d9a00f218ef015,
and exact immutable exclusion file raw hash
878ad80644570ef107fe7dd5ac8841ece39f7870d3062aac6bda38ce8062ddbc.
Input fields and metadata must equal these actual independently inspected
anchors, not replace them. The inspected producer binds the train-dev source
only, source_records=1304; each primary and literal-alias set has1304 unique
sorted lowercase64hex hashes. The #45 accepted total1552 includes other
historical lanes, and is not the count of this exact train-dev file. Do not
read or include historical holdout/rejected rows. This exclusion scope protects
train-dev non-overlap; cleanroom process prohibition on all historical reuse
continues, without claiming hash-level coverage of unopened holdout.

The closed saracura-historical-exclusion-source.v1 keys are schema_version,
status, historical_manifest_sha256, historical_source_sha256,
phase_a_receipt_sha256, source_records, normalization, identity_source_field,
question_source_field, option_source_field, sets,
literal_serialized_state_alias_sets, historical_holdout_opened,
raw_rows_uploaded, limitation. Literal status is DERIVED_NOT_SEALING_READY
(producer history; actual sealer now adds the independent gate), source_records
is integer1304; flags historical_holdout_opened/raw_rows_uploaded=false.
sets has exactly identity/state_question/combined_content; alias sets exactly
state_question/combined_content. Fixed normalization is
`v02_evaluation._normalise NFC-casefold-whitespace; RFC8785; SHA256`;
source fields are task_id/instruction/`criteria; IDs excluded by canonical
combined fingerprint`. Limitation is a nonempty provenance note, not a permit.
Every raw hash, schema key, set uniqueness/count/order and source/receipt bind
must pass before any capsule. Rehash raw #53 receipt and require its actual
packet_manifest_sha256 to equal the frozen #45 digest and historical blocked
status. No derivative trust based solely on the new input manifest.

The actual private-runtime-lock.v1 top-level keys are schema_version,
source_commit, public_source_integrity, code_sha256, cohort_bindings,
cpu_offload_gb, dependency_lock_sha256, driver, dtype, gpu, gpu_memory_mib,
historical_measurements, historical_model_metadata,
historical_model_metadata_sha256, host, image_digest, installed_versions,
max_model_len_by_role, max_num_seqs, max_output_tokens, native_json_grammar,
port, public_runner_sha256, python, quantization, renderer_source_sha256,
request_body_logging, teacher_input_ceiling_by_role, timeout_seconds,
tokenizer_inventory. All are existing production private fields; P3 preserves
this outer interface. P1 validates exact keys and consumed source proof, while
the original private runtime validates role/native/schema internals before
generation. Missing/changed schema returns BLOCKED_DATA_RIGHTS, not discovery.
public_source_integrity has exactly source_commit (40hex), archive_sha256
(64hex), source_inventory_digest (64hex), files_verified (positive integer,
not bool). Both lock and proof source_commit equal corpus_source_revision.
code_sha256 is a unique basename-to-64hex map over actual reviewed consumers;
installed_versions a nonempty string-to-string map; role maps have six exact
MODEL_ROLES keys. Cohort bindings retain the existing original reviewed shape.
Use source/runtime lock from actual source-reviewed/physically observed bundle,
not a self-asserted source success.

Every input field ending *_sha256 is raw-file SHA except policy_sha256 (the
RFC8785 policy identifier string hash) and historical_*_sha256 (fixed raw
source-file bindings above). runtime_lock_sha256 checks raw lock bytes;
grant.runtime_lock_digest hashes RFC8785 canonical parsed lock. Grant digest
used by ledger bindings is canonical parsed grant, not raw grant SHA.
Input tokenizer_inventory_sha256 checks raw inventory file bytes; capsule,
private admission and checkpoint tokenizer_inventory_sha256 are the existing
VerifiedTokenizer.inventory_digest over its canonical normalized inventory.
Do not confuse these namespaces. Source_inventory_digest follows the existing
source archive verifier's canonical sorted [{path,sha256}] inventory semantics.
All digest/commit regexes are literally `[0-9a-f]{64}`/`[0-9a-f]{40}`.

Reusable P1 API is verify_training_capsule(capsule_path, *, root, plan_path,
receipt_path, exclusions_path, sealer_inputs_path), returning verified descriptor
and rows with actual hash/reduction/provenance checks. A named private verifier
result may be used but cannot replace those checks. P2 optimization core is
independent; P2 private CLI/admission integration explicitly depends on this
P1 API and cannot clear P2-C3/P2-C4 or admit P6 until that integration passes.
P2-C3 additionally binds actual capsule/base/tokenizer/renderer/source/license/
environment bytes; aggregate READY alone is insufficient.
