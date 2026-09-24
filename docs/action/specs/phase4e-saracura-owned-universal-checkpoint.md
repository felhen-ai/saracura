---
title: Phase 4E Saracura-owned universal checkpoint
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
globalRef: qmd://saracura/docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-23
sourceRefs:
  - https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - https://github.com/huggingface/sentence-transformers
  - https://openrouter.ai/google/gemini-2.5-flash
  - https://openrouter.ai/meta-llama/llama-3.3-70b-instruct
  - https://openrouter.ai/api/v1/models/meta-llama/llama-3.3-70b-instruct/endpoints
  - https://openrouter.ai/terms
related:
  - AGENTS.md
  - README.md
  - docs/README.pt-BR.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/action/specs/phase3b-synthetic-research-training.md
  - docs/action/specs/phase4b-human-ptbr-calibration.md
  - docs/action/specs/phase4d-experimental-universal-choice.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E Saracura-owned universal checkpoint

## Context

Phase 4D proved that Saracura can execute a universal local decision model, but
its only installed universal backend reconstructs the Laya architecture and
loads the exact Laya tokenizer and checkpoint. That validates Saracura's typed
runtime and local safety boundary; it does not create a Saracura universal
model. An e-mail pilot on that backend would primarily evaluate Laya quality.

Before the mailbox pilot, Saracura therefore needs an independently trained
universal checkpoint. It may start from a general-purpose pretrained encoder,
as normal downstream model development does, but it must not consume Laya
weights, tokenizer bytes, logits, temperatures, labels, generated decisions,
or hidden representations. Laya remains an external benchmark adapter only.

The reviewed general-purpose base is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at immutable
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. The repository already
records its exact safetensors snapshot, Apache-2.0 declaration, 384-dimensional
BERT architecture, and local descriptor verifier. The public model card states
that it supports 50 languages and maps text to 384-dimensional vectors. This
phase treats that encoder as an attributed foundation dependency, not as a
Saracura-authored artifact.

## Objective

Produce the first reproducible, local, research-only Saracura universal Choice
checkpoint and execute it through the typed `v1alpha1` API. The checkpoint must
rank arbitrary ordered criteria in PT-BR and English without a task-specific
label head, use Saracura-owned training code and decision-layer weights, preserve
complete lineage, and remain explicitly uncalibrated and automation-disabled.

The phase is successful when a sealed synthetic-only checkpoint trained on the
MacBook runs locally, meets the precommitted unseen-family synthetic holdout
gate below, and is compared blindly with the Laya control. It is not successful
merely because contracts, fixtures, or an untrained head pass unit tests.

## Definition of a Saracura-owned model

For this phase, a model may be called `Saracura-owned universal checkpoint`
only when all conditions below hold:

1. Its decision architecture, renderer, objective, training loop, selection
   rule, and checkpoint schema are implemented in this repository.
2. Every trainable decision tensor is initialized by the Saracura training run;
   no tensor is loaded from a decision-specialized model or competitor.
3. The only pretrained tensors are from the pinned general-purpose MiniLM
   encoder. Their origin, revision, license declaration, and hashes remain
   explicit in every training and runtime manifest.
4. No Laya, Jev, TypeSafe, mDeBERTa, or other decision-model output, score,
   probability, label, embedding, or tensor is used as a teacher, target,
   feature, initialization, acceptance oracle, or hyperparameter-selection
   signal. The only provider-authored targets permitted are the independently
   reviewed synthetic rows produced under the exact closed Phase 4E exception
   below; their lineage remains model-generated and synthetic-only.
5. Training examples have record-level provenance. Synthetic examples remain
   visibly `model_generated_or_assisted`, cannot be relabeled human-original,
   and cannot support production, calibration, or representative-quality claims.
6. The emitted checkpoint contains only Saracura decision tensors and an exact
   binding to the separately verified base encoder. It does not repackage the
   base encoder or any Laya byte.

The product name `Saracura` describes the complete engine and model family. The
foundation encoder continues to be attributed by its own identity; “trained
from scratch” must not be claimed.

## Architecture: Saracura Universal Ranker v0

The first independent architecture is a shared multilingual bi-encoder ranker,
not the Laya option-marker head.

For each Choice question:

1. A context renderer creates one canonical length-framed text from locale,
   domain, instruction, and RFC 8785 state JSON.
2. A criterion renderer creates one canonical length-framed text from each
   criterion ID and description, preserving declared order.
3. The pinned MiniLM base mean-pools one context embedding and a batched set of
   criterion embeddings.
4. Separate Saracura context and criterion projections map 384 dimensions to
   192 dimensions using `Linear -> GELU -> LayerNorm`.
5. L2-normalized projected embeddings are scored by scaled cosine similarity.
   The learned positive scale is parameterized as a bounded log-scale. There is
   no per-label bias because labels are dynamic.
6. Cross-entropy over the options trains the selected criterion. Batches may
   contain different option counts; padding is masked before loss and scoring.

The base encoder is frozen in v0. This keeps training reproducible on the
MacBook and makes ownership boundaries auditable: the resulting checkpoint is
the Saracura projection ranker plus scale, layered over a separately attributed
foundation model. Fine-tuning base layers or adding adapters is a later model
revision and must outperform this baseline under the same blind protocol.

This design intentionally differs from Phase 4D: it does not use option-marker
tokens, a Laya decision transformer, Laya tensor names, or a Laya tokenizer. It
also supports cached criterion embeddings for stable high-volume schemas in a
later optimization without changing the public decision contract.

## Input and capacity contract

The runtime accepts only `choice` questions under a new exact workflow revision
`universal-choice@phase4e-saracura-ranker.v1`, locales `pt-BR` and `en`, 1–10
questions, and 2–8 criteria per question. Criterion order remains semantic.
`BackendCapabilities` gains an immutable set of supported dynamic workflow
references. Compiled backends declare none, Laya declares only
`universal-choice@phase4d-laya.v1`, and Saracura Universal declares only the
Phase 4E reference. `WorkflowRegistry.validate` requires both universal tier and
an exact capability match. Crossed backend/revision combinations fail with
`WORKFLOW_UNSUPPORTED` before snapshot, tokenizer, or model access; CLI
prevalidation receives the selected backend capability instead of accepting any
universal revision globally.

All strings and keys must already be NFC. Semantic fields are UTF-8-byte-length
framed; separator text can never change field boundaries. Context and every
criterion must fit independently without truncation:

- context: at most 128 MiniLM tokens including structural tokens;
- criterion: at most 96 MiniLM tokens including structural tokens;
- state: at most 200 Unicode code points and 800 UTF-8 bytes;
- instruction: at most 120 Unicode code points and 480 UTF-8 bytes;
- criterion description: at most 120 Unicode code points and 480 UTF-8
  bytes for this workflow, within the broader API bound;
- criterion batch: at most 20;
- model residency: one verified base plus one Saracura checkpoint per process.

Every generated row is rendered and tokenized with the exact verified MiniLM
snapshot before it can be accepted or counted. A row outside any source or
token limit is recorded as rejected in its preassigned split and cannot be
moved, shortened, or regenerated under the same task identity. Runtime overflow
returns `CAPACITY_EXCEEDED` before inference. No input, question, criterion, or
token sequence is silently shortened. The initial e-mail pilot will therefore
use explicitly selected subject and preview fields rather than pretend that an
arbitrarily long body was evaluated.

## Synthetic universal training corpus

The Phase 3B exception and `synthetic-research-policy.v1.json` remain byte for
byte unchanged. Phase 4E adds a second closed exception with its own ID,
workflow `phase4e-saracura-universal-synthetic.v1`, policy manifest, typed model,
and validator. The source-policy registry validator must accept exactly the two
known exception IDs and reject duplicates, unknown exceptions, changed Phase 3B
content, inheritance, or wildcard fields. `benchmarks.validate_manifests` must
route the new manifest by its exact schema version.

The Phase 4E exception uses the same already authorized OpenRouter author and
reviewer identities unless a current preflight invalidates their availability,
pricing ceiling, or provider policy. The operator's current authorization for
this execution is bounded to the USD 17.00 new-spend safety envelope below.
This guards against runaway spend; it is not a product budget or a reason to
abandon useful work. It may be increased by explicit operator agreement when
execution evidence shows that more is needed:

- author: `google/gemini-2.5-flash`;
- reviewer: `meta-llama/llama-3.3-70b-instruct`.

Provider routing is pinned to `Google` for the author and `CoreWeave` for
the reviewer, with `allow_fallbacks=false` and `require_parameters=true`.
OpenRouter must not silently route a request to an endpoint that ignores the
reviewed structured-output contract.

Before any provider request, an immutable public-seed plan assigns every task
ID, scenario-family ID, split, locale, domain, difficulty axes, option count,
balanced gold-option position, and a closed semantic generation target. The
target precommits one scenario code and one canonical selected-first sequence
of distinct criterion roles. It is sent only to the author, which must
materialize content matching it; it is never sent to the reviewer. The plan
also assigns cross-locale pair IDs and keeps paired tasks, deterministic
derivatives, and any planned criterion-order permutation inside one family and
split. The provider can never mint or change an identity, split, axis, count,
option position, or semantic target.

The exact domains are `email_triage`, `customer_support`, `finance`,
`accounting`, `commerce`, `operations`, `scheduling`, `document_routing`,
`browser_action`, `security_triage`, `content_moderation`, and
`personal_productivity`. The closed axes are explicitness
(`explicit`, `implicit`), negation (`absent`, `present`), distractor overlap
(`low`, `high`), urgency (`normal`, `urgent`), locale (`pt-BR`, `en`), and
option count (2–8). The planner balances gold position within every
locale/option-count cell.

The author receives planned identities and axes and produces only fictional
decision content plus a closed semantic attestation: instruction, state, 2–8
criterion descriptions, selected index, one scenario code, one ordered logical
role per criterion, and the selected role. The scenario code is selected from
`action_required`, `informational_only`, `suspected_abuse`,
`missing_information`, `deadline_risk`, `policy_violation`,
`duplicate_record`, `topic_routing`, `rule_eligibility`, `urgency_priority`,
`threshold_approval`, `reconciliation_mismatch`, `fulfillment_exception`,
`access_risk`, `schedule_conflict`, and `content_safety`. Criterion roles are a
distinct ordered subset of `matches_rule`, `contradicts_rule`,
`irrelevant_to_rule`, `insufficient_evidence`, `unsafe_action`,
`premature_action`, `overbroad_action`, and `duplicate_action`; the selected
role is always `matches_rule`. Paired PT-BR/English outputs must emit the same
attestation before local reordering. The local pipeline binds every
planner-owned field, moves the authored selected criterion and its authored role
together to the planned gold position, and derives the selected criterion ID;
it never fabricates a semantic attestation. The author response must exactly
restate the precommitted semantic target and make that target independently
inferable from the generated rule, facts, and options without writing role or
scenario labels into task text. Legacy full records are accepted only when
their planner-owned and semantic fields match exactly. Provider-authored `state.summary` is
limited to 180 characters so the complete framed state remains within the
200-code-point source contract. A generated row that still violates the source
or tokenizer contract is durably resolved as rejected under its original task
identity; a settled request must never be left unresolved or blindly retried.
Every settled provider call is therefore closed by one immutable, atomic
call-level resolution record covering all task identities in that call. If
response parsing, local validation, review resolution, or post-response
persistence cannot produce the normal accepted/rejected outcomes, that single
record resolves every still-unresolved identity as a content-free rejection,
retains only its planned split plus the settled author/reviewer lineage already
available, and uses a bounded reason code. Resume expands and validates both
legacy per-task records and call-level records, rejects duplicate or partial
coverage, and never resends an identity already bound to a settled call. Raw
provider content and reasoning are never written to this fallback record.
This includes a response whose reported cost settles the reservation into the
terminal overspend state: its journal and task lineage are exposed to the same
fallback path before the budget error is re-raised.

Author and reviewer receive the same concise, closed definitions for criterion
roles. They must derive each role from the rule, facts, and option meaning, and
must not write the role name into criterion text. The selected option directly
matches the rule; each distractor must unambiguously be exactly one of direct
contradiction, irrelevance, insufficient evidence, unsafe action, premature
action, overbroad action, or duplicate action. Rejection evidence distinguishes
scenario-code disagreement from criterion-role disagreement without retaining
provider reasoning or raw rejected content.

They also receive the same concise scenario codebook: required next action,
information only, suspected abuse, missing facts, deadline risk, policy breach,
duplicate record, topic routing, rule eligibility, urgency priority, approval
threshold, reconciliation mismatch, fulfillment exception, access risk,
schedule conflict, or content safety. The author response schema binds
`selected_index=0` and the exact planned scenario and selected-first role list
with inline single-value JSON Schema enums before transport; it avoids schema
composition keywords and schema references that are not portable across the
pinned provider route. State and criterion constraints are also materialized
inline, while the same Pydantic models remain the authoritative local validator.
The reviewer receives the codebook
but never the planned target. Author-side rejection evidence distinguishes a
target mismatch, prohibited semantic-label leakage into task text, and other
closed schema, state, option-cardinality, gold-position, attestation, criterion
identity, cross-locale, or typed task-contract failures without retaining
rejected content. Schema and task-contract codes may append a bounded,
sanitized Pydantic field-path/error-type signature; input values and exception
payloads remain prohibited. Field paths come only from a static local allowlist;
provider-controlled or otherwise unknown locations collapse to `unknown`.

The Wilson stop is evaluated at each ten-author-batch boundary once at least 20
planned tasks have resolved. Deterministic impossibility is always
authoritative; a Wilson projection cannot stop a cohort with fewer than 20
observations. This prevents both an eight-sample premature stop and needless
spend after a statistically impossible global acceptance trajectory.

The reviewer receives only `task_id`, `family_id`, locale, domain, instruction,
state, and the ordered criterion IDs/descriptions. It does not receive split,
gold position, selected criterion, difficulty axes, pair metadata, the author's
attestation or reasoning, or any field derived from the answer. It independently
selects one criterion or rejects the task and independently emits the same
closed semantic attestation shape. It also validates natural language, internal
sufficiency, option exclusivity, fictionality, privacy, and absence of sensitive
patterns. Only exact answer agreement and exact author/reviewer attestation
agreement are accepted; the reviewer output never rewrites the row. A paired
family counts only when the two author attestations and two independent reviewer
attestations are byte-equivalent after canonicalization.

The planner creates 1,600 candidate task slots, including 300 cross-locale
paired families distributed proportionally across train, dev, and holdout.
Generation proceeds in seeded
plan order until every slot is resolved or a budget fails. Every accepted task
remains in its preassigned split; no accepted task is discarded because a
minimum was already reached. The corpus gate requires at least 1,200 accepted
tasks across at least 12 domains, with minimums of 840 train, 180 dev, and 180
holdout. Every accepted task is one family member; at least 240 accepted tasks
form 120 PT-BR/English paired families. Each split must contain every domain,
both locales, option counts 2–8, negative/implicit cases, and at least 100
accepted families. At least 60% of accepted tasks are PT-BR and at least 20%
are English:

- 70% synthetic train;
- 15% synthetic dev;
- 15% synthetic holdout.

Dev and holdout must each contain at least 10 accepted tasks in every one of
the 14 locale × option-count cells; train must contain at least 20 in every
cell. An empty or under-minimum cell fails the corpus gate. The fixed
`stratified_macro_accuracy` has no missing-cell fallback, reweighting, or merge.

All members of a scenario family and cross-locale family stay in one split.
Exact and normalized duplicates, criterion permutations, and deterministic
derivatives stay with their source family. Split assignment is fixed from a
public seed before provider calls. Generation must stop rather than weaken a
minimum, mint a replacement identity, or move a family after seeing outcomes.
The packet records acceptance/rejection counts by split, locale, domain,
difficulty, option count, and reason so the agreement filter's easy-task bias is
visible.

At each ten-batch boundary after at least 20 tasks are resolved, the runner
computes for the global corpus, every split, and every locale × option-count
cell: current accepted count, unresolved planned slots, observed acceptance,
and the one-sided 95% Wilson lower bound. The deterministic capacity check is
always active: current accepted plus all unresolved slots must still be able to
meet every required minimum. A cohort-specific Wilson projection becomes
authoritative only after that cohort has at least 20 resolved outcomes; an
unobserved or smaller cohort records its descriptive rate but cannot trigger a
Wilson stop. Once eligible, the runner stops early when current accepted plus
`floor(unresolved * Wilson lower bound)` cannot meet the minimum within the
remaining stage budget. The calculation, cohort observation count, eligibility,
and stop reason enter the immutable ledger; no target is weakened after a stop.

Each planned cross-locale family is authored in one provider request containing
both task IDs, shared semantic axes, and separate locale slots. The author must
produce semantically equivalent, independently natural PT-BR and English tasks;
neither task is declared a translation of the other. The two authored semantic
attestations must match exactly. Each member is then sent alone to the reviewer
without the author attestation, pair ID, or sibling content, and its independent
reviewer attestation must match the authored one. Both remain in the same split
even if only one is accepted, and the pair counts toward the paired minimum only
when both are accepted. The 300 planned pairs provide rejection headroom; the
fixed acceptance minimum remains 120 complete pairs.

The provider lane keeps explicit network authorization, ZDR/data-collection
controls, no secrets in argv/logs/artifacts, atomic no-clobber writes, resumable
cost ledger, bounded retries, and record-level author/reviewer response hashes.
The Gemini author request disables optional model reasoning with
`reasoning_effort=none`; constrained generation must not spend the bounded
response budget on hidden reasoning. The pinned CoreWeave Llama reviewer route
is a non-reasoning endpoint and does not advertise `reasoning_effort`, so its
request omits that unsupported parameter while `require_parameters=true`
prevents silent rerouting. Neither response may contain or persist a reasoning
trace.
The current Phase 4E execution has a USD 17.00 new-spend safety envelope,
partitioned into four nonfungible stages: USD 5.00 corpus author, USD 10.00
corpus reviewer, USD 0.75 comparison author, and USD 1.25 comparison reviewer.
This envelope prevents an unattended runaway; it is not a hard project ceiling
and may be raised through explicit operator agreement. At the reviewed ceilings
of USD 0.30/M input and USD 2.50/M output for the author, and USD 0.71/M input
and USD 0.71/M output for the reviewer, the planned maximum payload and retry
envelope must calculate to no more than each stage limit before the first
request. For the frozen 300-pair/1,000-single plan, the conservative aggregate
preflight is currently USD 4.1572324 for the author and USD 9.99871629 for the
reviewer. These whole-run bounds must fit before transport begins; request-level
checks and ledger debits remain independently authoritative during execution.
Provider-reported costs and conservative local worst-case debits both enter the
new Phase 4E ledger; the larger debit governs. Prior Phase 3B/4C spend remains
immutable historical evidence and is not reset or charged to this new
authorization. If a price, model, provider policy, stage budget, or
accepted-corpus minimum fails, the run stops and requires a new explicit
authorization; it cannot borrow between stages. Raw provider payloads remain
ignored local artifacts and are not published. Public training data or weights
remain unauthorized until a separate rights and provider-terms review.

No e-mail, customer export, AIOS trace, private Felhen record, web crawl, social
post, Laya example, or TypeSafe output enters this corpus.

## Training, selection, and artifacts

The training command consumes only an immutable accepted packet and the exact
verified MiniLM snapshot. It refuses network access, cache discovery, arbitrary
model IDs, environment-selected model paths, or a packet whose source-policy
exception is absent.

The packet keeps evaluation content physically separate. Its canonical sealed
payloads are `accepted-train-dev.jsonl`, `accepted-holdout.jsonl`, and
`holdout-identities.json`. The identity payload contains only the fields needed
to prove split/family isolation and precompute the fixed holdout baselines; it
contains no instruction, state, criterion description, review, token, mask, or
embedding. Before the checkpoint digest, pre-holdout gate descriptor, and
irreversible release claim exist, validation and embedding derivation may read
the train/dev payload and the holdout identity payload, but must not open,
deserialize, digest-check from bytes, render, tokenize, or derive the holdout
payload or holdout tensors. The packet manifest may expose their precommitted
digests. After the claim, the full packet validator must bind the holdout
payload back to the sealed identities and plan before descriptor-bound holdout
embeddings are verified and scored. A failed post-claim validation consumes the
one-time holdout release and remains a failed experiment.

The one-time release claim lives in Saracura's canonical per-user state
registry, not beneath a caller-selected output directory. Its key is the
immutable packet manifest digest alone: a second extraction device, ABI,
capsule copy, output path, or checkpoint cannot reopen the same holdout. Its
payload binds the exact embedding capsule, selected checkpoint, and pre-holdout
gate descriptor. Tests may redirect the registry through an internal injected
path, but the production command exposes no application-level registry
argument. This is a fail-closed normal-operation guard, not a tamper-proof DRM
boundary against a user who deletes or relocates local state.

The fixed v0 search budget is one declared configuration, not an open sweep:

- seed fixed in the policy manifest;
- AdamW, declared learning rate and weight decay;
- maximum 40 epochs;
- early stopping on synthetic-dev stratified macro accuracy with declared
  patience;
- model selection only by dev metric and then lower epoch on an exact tie;
- synthetic holdout opened once after the checkpoint digest is frozen;
- no retry or hyperparameter change after seeing holdout results in this model
  revision.

Before training, the verified encoder produces frozen train/dev context and
criterion embeddings once on the explicitly selected CPU device. The
pre-holdout embedding capsule binds their token IDs, attention masks,
base-encoder identity, rendering revision, device/runtime ABI, float32 embedding
digests, and the identity-only holdout descriptor; it contains no holdout text,
token, mask, embedding, or `holdout.safetensors`. Projection training then runs
twice on CPU with PyTorch deterministic algorithms, fixed thread count and
seed. Both runs must produce byte-identical safetensors checkpoint bytes and
canonical metrics. After checkpoint, gate, and claim are frozen, the same
verified encoder opens the holdout payload once, derives its tensors in memory,
and scores them without creating a reusable pre-claim holdout capsule. A
failure is a reproducibility blocker, not a tolerance-based pass.

`stratified_macro_accuracy` is the unweighted arithmetic mean of the 14 fixed
cell accuracies formed by locale (`pt-BR`, `en`) times option count (2 through
8). The accepted-corpus gate guarantees every cell meets the fixed dev/holdout
minimum before embeddings may be extracted. Domains and difficulty axes are
reported but do not participate in selection.
No alternative grouping, F-score, micro average, or post-generation cell
definition is permitted in revision v1.

Before holdout access, the training manifest freezes these success gates:

- selected dev macro accuracy must exceed both the deterministic untrained
  cosine baseline and the deterministic random-projection baseline by at least
  0.03 absolute;
- holdout overall accuracy must be at least the greater of `0.45`, expected
  random accuracy plus `0.20`, and the precomputed most-frequent-gold-position
  baseline plus `0.15`;
- holdout PT-BR accuracy must be at least `0.40` and English accuracy at least
  `0.35`;
- every option-count bucket 2–8 must exceed its own expected random accuracy;
- zero non-finite logits, missing choices, extra choices, silent truncations,
  family overlaps, or deterministic-repeat mismatches.

The two dev-improvement gates are evaluated immediately from the frozen
descriptor. If either fails, the command seals a pre-holdout failed-experiment
result and stops without creating a release claim or opening holdout content.
Only a checkpoint that passes both dev gates may claim and observe holdout.

Expected random accuracy is the arithmetic mean of `1 / option_count` over the
accepted precommitted rows. The most-frequent-gold-position baseline is computed
over the accepted holdout task IDs after acceptance is sealed but before any
holdout text is released to model code. All task IDs, positions, baselines, and
denominators enter the release descriptor. Failing any gate preserves the
artifact as a failed experiment but does not satisfy Phase 4E or register its
runtime checkpoint.

The no-clobber training capsule contains exact canonical manifests, the
Saracura-only safetensors checkpoint, epoch ledger, dev selection report,
holdout report, conformance vectors, source packet receipt, human-readable
dependency versions, code digest, encoder identity, and architecture descriptor.
Conformance vectors are a separate safetensors file containing a deterministic
14-cell selection of context/criterion embeddings, option masks, and expected
ranker logits; they contain no rendered text or raw example. Raw training rows
remain outside Git and the wheel.

The checkpoint tensor set is closed. It contains context projection, criterion
projection, their LayerNorm tensors, and bounded log-scale only. It must not
contain an encoder tensor, Laya-compatible tensor name, optimizer state, raw
example, tokenizer byte, absolute path, hostname, username, secret, or provider
payload.

## Installed runtime and response semantics

Add the exact opt-in backend ID `saracura-universal`. It receives three explicit
operator paths: verified MiniLM encoder snapshot, sealed Saracura training
capsule, and later an optional compatible calibration artifact. Phase 4E itself
accepts no calibration and returns the same safe research semantics as Phase 4D:

- `status=uncalibrated`;
- `score_semantics=ranking_weights`;
- `abstained=true`;
- `reason=uncalibrated_research`;
- `calibration=null`;
- `automation_allowed=false`.

The model reference is exact: `id=saracura/universal-ranker`,
`revision=phase4e-saracura-ranker.v1.<training_manifest_sha256>`, and
`checkpoint_sha256=<checkpoint.safetensors SHA-256>`. The complete lowercase
manifest digest is used, so two training capsules cannot share a request model
revision. The backend description separately identifies the foundation encoder
ID, revision, snapshot digest, tokenizer revision, architecture revision, and
synthetic-only disposition. The runtime must never label the Laya adapter or
its weights as Saracura-owned.

The exclusive CLI form is `saracura decide --backend saracura-universal` plus
request, verified encoder snapshot, sealed training capsule, and explicit
device. It rejects `--calibration`, Laya snapshot arguments, MiniLM compiled
checkpoint arguments, and mixed backend artifacts. `describe-backend` accepts
the same encoder/capsule/device triple and reports the identities above without
local paths.

Before optional ML import or encoder access, the CLI opens the capsule through
the descriptor-bound verifier, validates its canonical training manifest,
derives the complete model revision, and compares it with `request.model`.
Only a matching revision may proceed to encoder/checkpoint verification. The
public example uses a documented placeholder until Phase 4E.3 produces the real
manifest digest; it must not pretend to be executable earlier.

Optional ML imports remain delayed. Default install, import, help, schema
validation, and invalid requests remain lightweight and offline. The backend
uses descriptor-bound reads, exact file ledgers, no symlinks/hardlinks/extras,
current-UID restrictive modes, tensor key/shape/dtype/finite checks, explicit
CPU or MPS, and no CPU fallback from MPS.

## Comparative evaluation

After the checkpoint and synthetic holdout report are sealed, run a comparison
on a separately generated, precommitted comparison set whose families are
absent from train/dev/holdout. The plan must constrain complete inputs to the
intersection of the Saracura 128/96-token envelopes and the Laya envelope;
this intersection is a planner target, not a post-outcome exclusion.
Capacity is checked with the explicit verified tokenizer snapshot for each
backend before scoring. Any unexpected unsupported result remains reported and
is never dropped from denominators. Loading the Laya tokenizer/control requires
its own `universal-local` extra and snapshot only in the comparison process; it
is never a dependency of Saracura training, verification, or inference.
Saracura is the candidate; Laya is an external control. They run in separate
fresh processes because each backend owns a resident-model boundary. Control
outputs cannot change Saracura weights, renderer, thresholds, or dataset.

The comparison planner precommits 200 candidate slots and requires at least 150
accepted tasks, both locales, all option-count buckets 2–8, and all 12 domains.
Its provider calls use only the dedicated USD 0.75 author and USD 1.25 reviewer
stage authorizations above. Comparison generation begins only after the
checkpoint and holdout report are sealed. Failure to reach its minimum or fit
its stage budget reports insufficient comparison evidence; it never spends a
training-stage balance or reopens model selection.

The report separates:

- synthetic task accuracy by locale, domain, difficulty, and option count;
- unsupported/capacity outcomes;
- cold load, warm latency, throughput, peak RSS, and artifact size;
- deterministic repeat checks;
- source, training, checkpoint, and code digests;
- incomparable cells and uncertainty.

A lower result than Laya is reported, not repaired by post-holdout tuning in the
same revision. No “better”, “PT-BR optimized”, calibrated-confidence, production,
or automation claim is allowed from synthetic evidence alone.

## Roadmap and mailbox boundary

This spec replaces the old Phase 4D next-gate wording:

- Phase 4E: Saracura-owned universal checkpoint and blind control comparison;
- Phase 4F: read-only internal e-mail shadow pilot using Saracura as the primary
  candidate and Laya only as an optional blinded control;
- later gate: human-original fit/evaluation, calibration, selective-risk policy,
  and separately authorized mailbox actions.

Phase 4F corrections may become private first-party evidence only under a new
reviewed policy. They are not public training data by default and are never
reused as evaluation after entering fit or training.

## Implementation phases

### Phase 4E.1: contracts, policy, and deterministic architecture

Implement the closed source-policy exception, universal task schemas, renderer,
variable-option ranker module, checkpoint schema, deterministic fake-encoder
tests, workflow registration, and roadmap/documentation correction. Update the
exact current-product and safety exception in `AGENTS.md`, the Phase 4D next
gate and execution report, both READMEs, packaging allowlists/exclusions,
`.gitignore` for `phase4e-plan.json`, `phase4e-accepted.jsonl`,
`phase4e-cost-ledger.json`, `phase4e-training-manifest.json`,
`phase4e-checkpoint.safetensors`, `phase4e-holdout-report.json`, and comparison
counterparts, plus a CI job that exercises fake transport and fake encoder
paths without MPS or real weights. No provider call, model acquisition,
training run, runtime registration, or real weight is allowed in this phase.

Phase 4E.1 documentation describes the workflow and backend as planned and
unsupported. `AGENTS.md`, README feature lists, and examples may declare them
installed only in Phase 4E.3 after a real checkpoint passes the holdout gate.

### Phase 4E.2: offline-first corpus and training lane

Implement plan generation, provider author/reviewer protocol, privacy and
duplicate gates, immutable packet, local training, dev selection, one-time
holdout, capsule verification, and adversarial tests. Reuse audited Phase 3B
provider/cost primitives where their contracts are identical by extracting a
parameterized shared module while keeping Phase 3B constants, bytes, behavior,
and tests unchanged; do not loosen or silently fork them. Generation requires
the explicit `local-minilm` extra and descriptor-bound receipt for the verified
tokenizer snapshot so exact capacity rejection occurs before acceptance. Run a
small fake-transport exercise before any network call. Tests monkeypatch socket
construction to prove that packet validation, training, capsule verification,
and the installed backend have no network path.

The immutable packet schema for this phase is v2 and uses the physically split
train/dev, holdout, and identity-only payloads defined above. There is no
fallback to the v1 combined `accepted.jsonl` in the production training path.

### Phase 4E.3: real checkpoint and runtime

With the operator's existing authorization and credential boundary, generate
the real synthetic packet within the new cumulative budget, train locally on the
verified MiniLM snapshot, seal the checkpoint, add the `saracura-universal`
runtime/CLI, and execute device-specific CPU and MPS conformance. Embedding
capsules and deterministic-repeat gates are ABI-bound to their extraction
device; CPU/MPS differences are reported and never required to be byte-identical
across devices. Any provider-model disappearance,
policy failure, insufficient accepted corpus, budget exhaustion, missing base
snapshot, or training failure stops this phase; it never falls back to Laya.

### Phase 4E.4: blind comparison and public evidence

Freeze an independent comparison plan, execute Saracura and optional Laya
control without training feedback, publish only sanitized aggregate evidence,
update bilingual documentation, and state the exact limitations. The repository
and wheel still contain no dataset, encoder, or checkpoint bytes unless a later
publication review explicitly changes that boundary.

## Acceptance criteria

1. Repository terminology distinguishes Saracura engine, Saracura checkpoint,
   generic foundation encoder, and Laya external control.
2. The Saracura checkpoint verifier rejects any encoder or Laya tensor and binds
   exact training code, packet, architecture, and encoder revision digests.
3. Renderer and ranker tests cover PT-BR/English, 2/8/20 options, repeated
   descriptions, order preservation, NFC, length framing, capacity rejection,
   non-finite values, and variable-option masks.
4. Provider tests prove explicit network opt-in, pinned host/models, ZDR/data
   controls, cumulative budget, no secret leakage, no-clobber resume, and
   reviewer blindness to the author answer.
5. Split tests prove family/cross-locale isolation and zero train/dev/holdout or
   comparison overlap under exact and normalized duplicate scans.
6. Training is reproducible for the declared environment, freezes the encoder,
   emits only the closed Saracura tensor set, selects once on dev, and opens the
   holdout only after the checkpoint digest is fixed.
7. The real training capsule contains an actually trained checkpoint; a fixture
   or random initialized head cannot satisfy the phase.
8. `saracura-universal` runs locally on the real checkpoint and returns only
   uncalibrated abstained rankings with automation disabled.
9. Laya is not required to train or run `saracura-universal`; deleting the local
   Laya snapshot cannot affect Saracura checkpoint verification or inference.
10. Default install remains offline/lightweight; build artifacts contain no
    data, weights, provider payloads, credentials, absolute paths, or binaries.
11. Full QA, wheel/sdist inspection, CPU/MPS conformance where available, and a
    read-only technical review pass before integration.
12. The final report states synthetic-only limitations and does not claim
    calibrated confidence, production quality, representative PT-BR, training
    from scratch, or authority for any external action.

## Validation commands

```bash
git diff --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python benchmarks/validate_default_environment.py
uv build
uv run python benchmarks/inspect_wheel.py <wheel> <sdist>
```

Phase-specific real commands must be added by Phase 4E.2 with explicit input
and output directories under ignored `.artifacts/`; secrets must enter through
the existing 1Password-backed environment boundary and never through argv.

The checkout-only operational entry point is `python -m
benchmarks.phase4e_pipeline`. It exposes the following no-clobber stages:

```bash
uv run python -m benchmarks.phase4e_pipeline plan \
  --output .artifacts/phase4e/plan.json

# OPENROUTER_API_KEY is an ephemeral environment value supplied by `op run`.
# It is never accepted as an argument, written to an artifact, or printed.
uv run --extra local-minilm python -m benchmarks.phase4e_pipeline corpus \
  --plan .artifacts/phase4e/plan.json \
  --snapshot <verified-minilm-snapshot> \
  --work-dir .artifacts/phase4e/corpus-work \
  --packet .artifacts/phase4e/accepted-packet \
  --allow-network

uv run --extra local-minilm python -m benchmarks.phase4e_pipeline extract \
  --packet .artifacts/phase4e/accepted-packet \
  --snapshot <verified-minilm-snapshot> \
  --device cpu \
  --output .artifacts/phase4e/embedding-capsule

uv run --extra local-minilm python -m benchmarks.phase4e_pipeline train \
  --packet .artifacts/phase4e/accepted-packet \
  --snapshot <verified-minilm-snapshot> \
  --embeddings .artifacts/phase4e/embedding-capsule \
  --device cpu \
  --output-parent .artifacts/phase4e/training \
  --output-name saracura-universal-v0

uv run --extra local-minilm python -m benchmarks.phase4e_pipeline verify \
  --packet .artifacts/phase4e/accepted-packet \
  --embeddings .artifacts/phase4e/embedding-capsule \
  --training .artifacts/phase4e/training/saracura-universal-v0
```

`corpus` is the only network-capable stage. It requires the literal
`--allow-network` flag, reads only `OPENROUTER_API_KEY`, uses the pinned HTTPS
host with redirects and proxies disabled, persists the reservation ledger
before each request, and records response digests rather than credentials. Its
append-only work directory supports deterministic resume of completed task
identities; an unresolved pre-send reservation fails conservatively instead of
silently repeating a possibly charged request. Once a request is settled, its
entire task set is resolved by one atomic call-level record even when parsing,
validation, review resolution, or ordinary result persistence fails; restart
must skip those identities without another provider call. Author batching never
separates a planned cross-locale family, and every reviewer request contains
exactly one validated row. At each required ten-batch boundary the command
persists and applies the Wilson stop projection. It seals the packet only after
all 1,600 slots are resolved and every corpus minimum passes. `plan`, `extract`,
`train`, and `verify` construct no socket.

Revision v0 extraction and training are CPU-only so descriptor-bound
re-derivation is byte-exact across processes. MPS is reserved for later runtime
latency/conformance checks against the CPU-authored checkpoint; it cannot
produce a second embedding capsule or a second holdout release.

## Rollout and rollback

All runtime behavior is opt-in and research-only. Existing fixture,
`minilm-routing`, and `laya-universal` paths remain unchanged. No service,
mailbox, production runtime, account setting, or external automation is
modified. Rollback is a Git revert of the Phase 4E commits plus recoverable
removal of the explicitly named local Phase 4E artifact directories after their
digests and reports have been recorded. Provider spend is not recoverable.

## Risks

- **Synthetic tasks teach provider style rather than real decisions:** keep
  synthetic-only labeling, use heterogeneous domains and difficulty axes, and
  require later human-original evidence.
- **A thin trained head is presented as a foundation model:** explicitly name
  and attribute the frozen MiniLM base and prohibit “from scratch” claims.
- **Dual-encoder quality trails joint encoders:** preserve the Laya blind control
  and report the result; do not silently copy the competitor architecture.
- **Short context misses important e-mail content:** fail on capacity and defer
  the exact preview/body selection to Phase 4F rather than silently truncate.
- **Provider terms or models change:** re-run live preflights and stop on drift;
  do not substitute a model without a revised spec and manifest.
- **Publication rights are unclear:** keep corpus/checkpoint local and
  unpublished until a dedicated rights and release review.
- **MacBook training is unstable or too slow:** fixed frozen-encoder v0 keeps the
  trainable surface small; failure blocks the real checkpoint rather than
  falling back to Laya or a remote runtime.
