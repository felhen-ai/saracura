---
title: Phase 4C universal decision backend bake-off
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4c-universal-backend-bakeoff.md
globalRef: qmd://saracura/docs/action/specs/phase4c-universal-backend-bakeoff.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs:
  - https://github.com/NandhaKishorM/laya
  - https://github.com/wfzyx/von
  - https://github.com/rupeshpoojary9/poorjev
  - https://github.com/sgoedecke/system-one
  - https://github.com/Shalimov04/open-jev
  - https://github.com/razorback16/openjev
  - https://typesafe.ai
related:
  - AGENTS.md
  - SECURITY.md
  - docs/action/specs/phase2b-encoder-candidate-gate.md
  - docs/action/specs/phase2c-data-governance-gate.md
  - docs/action/specs/phase3b-synthetic-research-training.md
  - docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
  - docs/action/specs/phase3d-typesafe-native-control.md
  - docs/action/specs/phase4a-opt-in-minilm-backend.md
  - docs/action/specs/phase4b-human-ptbr-calibration.md
  - docs/action/specs/phase4c2-local-candidate-adapters.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4C universal decision backend bake-off

## Context

Saracura currently proves a deliberately narrow path: a frozen multilingual
MiniLM encoder plus a five-label support-routing head. That path can be very
fast for a known recurring workload, but it cannot accept a new question and a
new set of choices without another trained head. Treating every use case as a
pack would make the project look like a collection of classifiers rather than a
general decision primitive.

Existing projects solve generality in materially different ways:

- Laya and Von jointly encode instructions, choices, and state, then score one
  marker per choice. Their public API accepts new choice sets, but their useful
  quality still depends on broad training or task-specific fine-tuning.
- poorjev converts each choice into a zero-shot NLI pair. It needs no
  task-specific head, but work grows with `questions × choices` and probability
  quality is not inherited automatically from entailment scores.
- `system-one` asks a causal language model for constrained next-token logits.
  It inherits broad language capability and can act as a teacher or fallback,
  but is much heavier than an encoder and is not calibrated by construction.
- Shalimov's OpenJev distills a teacher into one fixed-task student. That is a
  useful compiled specialization, not the universal layer.
- DiffusionGemma OpenJev uses parallel masked answer slots, but its weight and
  hardware footprint make it unsuitable for Saracura's first local MacBook
  lane.
- TypeSafe Jev is a valuable external control, but its implementation and
  training are proprietary and cannot define Saracura's architecture.

The key distinction is between **API generality** and **model generality**. A
backend may accept arbitrary labels while still performing poorly on unseen
domains or languages. This phase measures both rather than selecting an
architecture from README claims.

## Truth snapshot

- The default Saracura install is lightweight and offline.
- The only executable learned backend is a fixed five-label PT-BR research
  workflow. Its synthetic training evidence does not support a general quality
  claim or automation.
- Phase 2B already defines explicit, pinned, safetensors-only encoder
  acquisition with no request-triggered download.
- Phase 2C quarantines model-generated or model-assisted examples from canonical
  train, dev, calibration, and blind-test evidence.
- Phase 3B authorizes a sealed synthetic-only research exception. It can support
  only its own frozen workflow. Phase 4C does not inherit that exception and may
  not read its lineage until the data-policy registry is explicitly revised.
- Phase 3C already owns the cumulative OpenRouter authorization and its durable
  reservation/settlement journal. Phase 4C cannot create a parallel budget.
- Phase 3D already owns the native TypeSafe Jev protocol, fixed endpoint/model,
  cost accounting, and single-use predecessor claim. Phase 4C references that
  lane rather than respecifying it.
- Phase 4B defines the human PT-BR gate. Until qualifying human evidence exists,
  all universal-backend conclusions remain research recommendations.
- The current backend protocol encodes state once independently of questions.
  That is an invariant of the compiled backend, not a neutral assumption for a
  universal model: Laya-, Von-, NLI-, and prompt-style candidates are all
  question-conditioned.

## Decision and outcome

Build a checkout-only, fail-closed architecture bake-off before changing the
runtime backend protocol or public API. Phase 4C produces:

1. a closed benchmark scenario, candidate, run-plan, and result contract;
2. an immutable candidate registry that describes exact implementations and
   acquisition disposition without downloading anything;
3. deterministic offline reference adapters that prove scheduling, metric,
   redaction, and comparison semantics without pretending to be models;
4. opt-in local reference adapters for eligible pinned candidates;
5. separately budgeted remote controls for Qwen and native Jev;
6. a versioned decision record that recommends a universal backend, rejects all
   candidates, or records insufficient evidence.

The expected product shape, if supported by evidence, is one public decision
API with two execution tiers:

- `universal`: question-conditioned decisions for new schemas and domains;
- `compiled`: optional specialized heads for stable, high-volume workloads.

Recipes and cookbook examples demonstrate use cases. They are not required
model packs and do not constrain what a caller may ask the universal tier.

## Non-goals

This phase does not:

- alter the `v1alpha1` runtime API, `Backend` protocol, workflow registry, or
  global safety invariant;
- add dynamic labels, Boolean, ordinal, score, HTTP serving, runtime model
  downloads, automatic remote fallback, or user-selected model paths;
- train, fine-tune, calibrate, quantize, publish, or redistribute model weights;
- approve a third-party dataset, use private Felhen/AIOS/customer data, or turn
  model-generated cases into human evidence;
- claim that agreement with a teacher is correctness;
- spend provider credit, download weights, or acquire datasets during the
  contract-only first increment;
- select a production architecture using synthetic evidence alone.

## Candidate classes

The registry identifies candidate **implementations**, not marketing names.
Each candidate has one closed architecture class:

| Candidate class | Reference | Role in bake-off | Initial disposition |
| --- | --- | --- | --- |
| `compiled_encoder_head` | current Saracura MiniLM head | speed and specialization reference | local, existing artifacts only |
| `option_marker_encoder` | Laya multilingual | primary universal hypothesis | local only after exact safe snapshot and adapter review |
| `option_marker_encoder` | Von | independent reference for the same family | local only after exact safe snapshot and adapter review |
| `zero_shot_nli` | poorjev-style adapter | no-task-training baseline | local only after exact safe snapshot and adapter review |
| `constrained_causal_lm` | Qwen/system-one-style | broad teacher/control | remote first; optional local later |
| `native_remote` | TypeSafe Jev | external product control | remote, separately authorized |
| `masked_diffusion` | DiffusionGemma OpenJev | heavyweight comparison | documented only; excluded from Mac execution |

The harness must not import an upstream repository as trusted code. A local
candidate is either implemented through reviewed Saracura-owned adapter code
over verified bytes. Running an upstream environment is outside this spec and
requires the separate isolation review described in Phase 4C.2. This is not a
runtime plugin mechanism.

## Evidence lanes

Every scenario and result declares exactly one lane:

### `synthetic_research`

Phase 4C may read Phase 3B synthetic lineage only after a reviewed revision of
`training-data-source-policies.v1.json` extends the `synthetic_experiment`
exception to workflow revision `universal-bakeoff-plan.v1`. Until then the only
authorized input is self-authored deterministic fixtures committed in the plan.
The Phase 3B `synthetic_holdout` is never an input to candidate ranking.

Allowed conclusions depend on whether a real candidate executed. Phase 4C.1
reference adapters may report only declared analytic work and failure counts.
Later real-candidate runs may report execution compatibility, behavioral
difference, latency, throughput, memory, and teacher agreement under a reviewed
plan. Accuracy, calibration, risk coverage, locale quality, and production
selection are never allowed conclusions in this lane.

### `external_control`

Uses only a separately approved artifact bound to an allowed Phase 2C source
policy and immutable byte manifest. The official MASSIVE `pt-PT` policy is not
itself an artifact approval and remains a separate native intent-classification
control. It cannot be relabeled as PT-BR or mapped into the support workflow
without a later reviewed spec.

### `human_evaluation`

Uses a Phase 4B-compatible, human-original, provenance-complete artifact with
frozen split and access controls. This is the only lane that may support the
final PT-BR architecture recommendation, calibration evidence, or comparative
quality claim. English and any other locale require their own evidence slices
and denominators.

A bake-off may not read the Phase 4B blind-test split. Comparative candidate
claims require a separate, independently authored human evaluation split, a
precommitted primary metric, multiplicity correction for the exact candidate
count, and a power analysis sized for that count. The Phase 4B minimum of 100
blind records does not satisfy this comparative purpose.

The harness accepts metrics through a closed allowlist per evidence lane and
adapter authority. It never attempts to prevent overclaiming through a denylist
of convenient field names.

## Evaluation matrix

The closed benchmark plan expresses a matrix without requiring every candidate
to support every cell. Unsupported cells are explicit structured results and
never silently omitted.

### Axes

- locale: `pt-BR`, `en`, and a documented `pt-PT` external-control axis whose
  state is `unreachable_until_artifact_manifest`;
- domain: exercised `support` and `inbox_triage`; `browser_action` and
  `agent_tool_routing` are documented future axes and are not exercised until
  their product semantics receive a reviewed spec;
- novelty: `seen_taxonomy`, `unseen_taxonomy`, and `unseen_domain`;
- questions per state (`Q`): 1, 10, and 50;
- choices per question (`K`): 2, 5, 20, and 77;
- independent states per batch (`N`): 1, 8, and 32;
- candidate execution: `compiled`, `joint_encoder`, `pairwise_nli`,
  `constrained_lm`, or `native_remote`.

`Q`, `K`, and `N` are declared dimensions, not inferred from duplicated rows.
Each measured scenario has an exact state count and an exact question/choice
shape. A candidate may impose a reviewed token or choice cap; exceeding it must
return `unsupported_capacity`, not truncate without evidence.

### Metrics

For `synthetic_research` with Phase 4C.1 `fixture_only` reference adapters, the
allowed metric IDs are exactly:

- `declared_sequence_count`;
- `declared_attention_work`;
- `declared_state_count`;
- `declared_question_count`;
- `declared_choice_count`;
- `unsupported_count`;
- `failed_count`.

`declared_attention_work` is the sum of squared declared sequence lengths, with
lengths derived deterministically from scenario bytes. It is an analytic shape
model, not a FLOP, latency, tokenization, or hardware measurement. Sequence
count alone is not a cost proxy and must never be presented as a scaling
advantage. Every Phase 4C.1 work number carries
`metric_authority=declared_analytic_model`. The compiled reference's reusable
state encoding is a declared premise being modeled, not a measured result.

Phase 4C.1 cannot report load time, warm-up time, latency, throughput, RSS,
device memory, tokens, stability, ablation, teacher agreement, quality, winner,
or recommendation. A later real-candidate plan must add an exact metric
allowlist appropriate to its execution and evidence lane.

`human_evaluation` may additionally report top-1 accuracy, macro-F1, Brier
score, ECE with frozen bins, selective risk/coverage, abstention rate, grouped
confidence intervals, and PT-BR/English parity. These metrics require a
precommitted primary metric, practical margin, sample-size analysis, and
untouched evaluation split. Fit and evaluation records must be disjoint.

## Phase 4C.1: closed contracts and offline harness

Add the following checkout-only resources:

- `benchmarks/manifests/decision-backend-candidates.v1.json`;
- `benchmarks/manifests/universal-bakeoff-plan.v1.json`;
- `benchmarks/universal_bakeoff.py`;
- `tests/test_universal_bakeoff.py`;
- new routes in `benchmarks/validate_manifests.py` and manifest-routing tests in
  `tests/test_manifests.py`.

### Candidate registry contract

The registry root has exactly `schema_version`, `reviewed_at`, and `candidates`.
The schema is `decision-backend-candidates.v1`. Candidate IDs are unique and
match `^[a-z][a-z0-9-]{2,63}$`. Each candidate has exactly:

- `id`, `display_name`, and `architecture_class`;
- `role`: `specialized_reference`, `universal_candidate`, `teacher_control`,
  `product_control`, or `documented_exclusion`;
- `source_type`: `first_party`, `open_source`, or `remote_service`;
- `source_url`, `source_revision`, `source_license`, and
  `license_evidence_url`;
- `model_id`, `model_revision`, and `weight_format`;
- `execution_boundary`: `existing_verified_local`, `review_required_local`,
  `review_required_remote`, or `excluded`;
- `data_transmission`: `none`, `provider_request`, or `not_applicable`;
- `claimed_local_mac`, `claimed_dynamic_choices`, `claimed_ptbr`, and
  `claimed_question_conditioned` booleans plus `claim_evidence_state`, one of
  `upstream_readme`, `upstream_source_reviewed`, or `measured`;
- `acquisition_state`: `existing`, `not_reviewed`, `excluded`, or
  `not_applicable`;
- `exclusion_reason`, nullable only when not excluded;
- `notes`, a 20–500 character string limited to printable Unicode without
  control characters.

`display_name` is 3–80 characters. `exclusion_reason` is either null or 20–300
printable characters and is non-null exactly for excluded candidates.
`source_license` is the closed enum `Apache-2.0`, `MIT`, or `unknown`.

Open-source repository revisions are immutable 40-character lowercase Git
SHAs. The first-party Saracura row and remote-service TypeSafe row use
`source_revision=null`; no manifest attempts to predict the commit that will
contain itself or invent an opaque service revision. Model revisions are
immutable 40- or 64-character lowercase hex digests when reviewed model bytes
exist. A missing reviewed local model revision is `null` and forces
`acquisition_state=not_reviewed`; remote services use `not_applicable`, and
documented exclusions use `excluded`. No null is replaced with a branch,
marketing version, or provider alias. URLs are HTTPS, contain no
userinfo/query/fragment, and never point to mutable `latest`, `main`, `master`,
`resolve/main`, or `tree/main` resources.

The registry has exactly these seven IDs and fixed Phase 4C.1 identities:

| ID | Source identity | Architecture and role | Boundary |
| --- | --- | --- | --- |
| `saracura-compiled` | first-party Saracura; Apache-2.0; source revision null | `compiled_encoder_head`; `specialized_reference` | `review_required_local`; acquisition `not_reviewed` |
| `laya-multilingual` | `NandhaKishorM/laya@573e5b62696ba441230cd6be71d593331b5d23af`; Apache-2.0 | `option_marker_encoder`; `universal_candidate` | `review_required_local`; acquisition `not_reviewed` |
| `von-option-marker` | `wfzyx/von@2656a69be2ef0ebf2f3058302e52791f99cbd8de`; Apache-2.0 | `option_marker_encoder`; `universal_candidate` | `review_required_local`; acquisition `not_reviewed` |
| `poorjev-nli` | `rupeshpoojary9/poorjev@7e684e95db13b90238c63e6ab39e1a016263a168`; MIT | `zero_shot_nli`; `universal_candidate` | `review_required_local`; acquisition `not_reviewed` |
| `qwen-system-one` | `sgoedecke/system-one@ebde2a2db7067b920dfe51e9ce785613e66613d5`; license unknown | `constrained_causal_lm`; `teacher_control` | `review_required_remote`; acquisition `not_applicable` |
| `typesafe-jev` | TypeSafe remote service; source revision null; license unknown | `native_remote`; `product_control` | `review_required_remote`; acquisition `not_applicable` |
| `diffusiongemma-openjev` | `razorback16/openjev@e04794ab36e4f7e6040c2547baecdb2737ce2e79`; license unknown | `masked_diffusion`; `documented_exclusion` | `excluded`; acquisition `excluded` |

Identity, capability, boundary, acquisition, and evidence fields are fixed by
an ID-bound truth table in the validator. In Phase 4C.1 all `model_id` and
`model_revision` values are null. Local and repository-backed candidates use
`weight_format=not_reviewed`; the remote TypeSafe service and documented
exclusion use `not_applicable`. Capability booleans are upstream claims unless
the validator fixes a stronger evidence state; none is silently promoted to
measured evidence.

Phase 4C.1 records the research disposition only. It contains no file
allowlists or acquisition URLs and cannot download or execute candidates.
Phase 4C.2 must extend the registry through a new schema revision with exact
file names, sizes, SHA-256 values, loader/tokenizer class review, and license
evidence before any local bytes are acquired.

### Run-plan contract

The run-plan root has exactly `schema_version`, `plan_id`,
`generator_revision`, `evidence_lane`, `candidate_registry_sha256`,
`dataset_artifact`, `remote_budget`, `scenarios`, and `report_policy`. Schema is
`universal-bakeoff-plan.v1` and the generator revision is
`phase4c.1-fixture-generator.v1`.

- Phase 4C.1 fixes `evidence_lane=synthetic_research`,
  `dataset_artifact=null`, and remote budget to zero requests and zero USD.
- Scenarios use only self-authored deterministic fixture specifications
  committed in the plan and materialized by the fixed generator revision; they
  contain no private, customer, operational, or copied third-party text.
- Each scenario declares locale, domain, novelty, `N`, `Q`, `K`, an input seed,
  and authorized metric IDs.
- The candidate registry digest is SHA-256 of exact registry bytes.
- Unknown fields, duplicate JSON keys, duplicate IDs, unapproved metrics,
  non-finite numbers, inconsistent dimensions, and unbounded strings fail
  closed.

Phase 4C.1 has exactly eight scenarios rather than the full Cartesian product:

| # | Locale | Domain | Novelty | N | Q | K | Input mode |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `pt-BR` | `support` | `seen_taxonomy` | 1 | 1 | 5 | `self_authored` |
| 2 | `pt-BR` | `support` | `seen_taxonomy` | 8 | 1 | 5 | `self_authored` |
| 3 | `pt-BR` | `support` | `unseen_taxonomy` | 1 | 10 | 5 | `self_authored` |
| 4 | `pt-BR` | `inbox_triage` | `unseen_taxonomy` | 8 | 10 | 20 | `rule_generated` |
| 5 | `en` | `support` | `seen_taxonomy` | 1 | 1 | 5 | `self_authored` |
| 6 | `pt-BR` | `support` | `seen_taxonomy` | 32 | 1 | 2 | `rule_generated` |
| 7 | `pt-BR` | `inbox_triage` | `unseen_taxonomy` | 1 | 50 | 20 | `rule_generated` |
| 8 | `pt-BR` | `support` | `seen_taxonomy` | 1 | 1 | 77 | `rule_generated` |

`rule_generated` inputs are deterministic strings produced from the committed
generator version and scenario seed; they are not model-generated. The three
dimensions are checked against the materialized scenario shape before a report
is produced.

### Offline reference adapters

Phase 4C.1 implements only deterministic reference adapters:

- a `compiled_reference` adapter representing one reusable state encoding and
  per-question fixed-head scoring;
- a `joint_encoder_reference` adapter representing one sequence per
  state/question with all choices packed together;
- a `pairwise_nli_reference` adapter representing one sequence per
  state/question/choice;
- a `constrained_lm_reference` adapter representing one prompt per
  state/question with all choices packed together.

They calculate `declared_sequence_count` and `declared_attention_work` from
canonical scenario bytes framed with explicit byte lengths, never ambiguous
delimiters, and may produce only a deterministic placeholder ranking explicitly
labeled `deterministic_placeholder`. They must be named `*_reference_adapter`,
never `backend` or `model`, and their report status is `fixture_only`. They exist
to prove computation-shape and contract consistency; they are not proxies for
quality, FLOPs, or measured latency.

The command is:

```bash
python -m benchmarks.universal_bakeoff validate
python -m benchmarks.universal_bakeoff run-reference --output <run-name>
```

`validate` accepts no path, URL, candidate, override, network, or secret input.
`run-reference` accepts only a new child name under the repository-local
`.artifacts/universal-bakeoff/` root. It creates that directory exclusively with
mode 0700 and exactly `result.json`, `report.md`, and
`artifact-manifest.json`, each mode 0600. It never overwrites, performs no
network call, and writes canonical JSON through sibling temporary files
followed by atomic create-if-absent promotion. The final manifest records size
and SHA-256 for the other two files plus its own canonical self-digest. Public
errors use bounded codes and do not echo local paths, scenario text, URLs, or
exception messages.

The report contains the exact registry and plan digests, tool revision, runtime
environment without usernames or paths, adapter results, and an explicit
`conclusion_authority=none`. Raw prompts, states, provider responses,
credentials, and model artifacts are never report fields.

## Phase 4C.2: eligible local candidate adapters

This phase requires a separately reviewed registry v2 before implementation.
For each candidate it must:

1. pin repository and model revisions plus exact allowlisted files, sizes, and
   SHA-256 values;
2. verify source and weight licenses independently;
3. allow only safetensors and inert reviewed tokenizer/config files;
4. reject pickle, `.bin`, remote code, `auto_map`, arbitrary paths/URLs,
   symlinks, hardlinks, unexpected files, and mutation during load;
5. acquire only through an operator command with `--allow-network` into a
   private cache and never from a decision request;
6. construct model/tokenizer classes from verified bytes in Saracura-owned
   adapter code;
7. require a frozen conformance vector proving parity with the upstream family
   before a negative conclusion may be attributed to that architecture;
8. emit the same bounded result contract as Phase 4C.1.

Running upstream code in a built environment requires its own reviewed spec
with an explicit isolation contract: no operator credentials, no home
directory, and no network after build. It is not an option in Phase 4C.2.

Initial MacBook smoke execution should prioritize one safe option-marker
candidate and one NLI baseline. Von and Laya are the same architecture family;
running both is useful only if their pinned weights or training provenance make
the comparison independent. A blocked or unsafe checkpoint is a valid result.

## Phase 4C.3: controlled remote comparisons

Remote comparisons require a reviewed plan revision with exact providers,
models, immutable provider-visible version where available, request and token
caps, retry policy, timeout, and redaction policy. They extend the existing
cumulative authorization and stage allocations; they reuse the Phase 3C
reservation/settlement journal and single-use predecessor claim registry. A
new ceiling requires a new operator authorization recorded in the plan
revision. Native TypeSafe execution remains governed by Phase 3D; Phase 4C
references it and does not restate its endpoint or protocol.

The runner accepts no key flag, path, secret reference, or base URL. It reads
exactly the provider environment variable allowed by Phase 3B or Phase 3D and
sends it only in the Authorization header to the literal governed host. Values
are never copied into repository files, arguments, reports, or logs. Remote
inputs must be self-authored, synthetic, and non-sensitive. Provider failures
never trigger an automatic fallback.

## Phase 4C.4: architecture decision record

The ADR is produced only after the intended evidence lanes are complete. It
must compare:

- generality on unseen taxonomies and domains;
- PT-BR behavior and cross-locale parity;
- quality and selective-risk evidence where authorized;
- scaling with `N`, `Q`, `K`, state length, and choice-description length;
- cold/warm latency, throughput, memory, artifact size, and MacBook usability;
- calibration stability across question type and option count;
- supply-chain, licensing, data, and operational complexity;
- whether compiled specialization materially improves stable high-volume work.

Possible dispositions are `recommend`, `reject`, or `insufficient_evidence`.
Synthetic-only evidence forces `insufficient_evidence` for production
selection. A recommendation must name the exact evidence artifacts and known
unsupported cells. It does not itself modify runtime code.

## Invariant review

The existing statement “state encoding must not depend on questions or
criteria and must execute once per request” remains binding for the current
runtime and compiled backend throughout Phase 4C. The bake-off is explicitly
allowed to model question-conditioned computation because every credible
universal candidate requires it.

After Phase 4C.4, a separate reviewed runtime spec may replace the global rule
with scoped invariants:

- compiled backends encode each independent state once and reuse it;
- universal backends may jointly encode state, question, and choices;
- the engine exposes computation and capacity limits instead of promising a
  physically impossible shared encoding;
- no backend may silently truncate questions, choices, or state beyond its
  declared policy.

Until that spec is merged, Phase 4C code stays under `benchmarks/` and is not
registered with `DecisionEngine`.

## Security and privacy invariants

- Default installation and import remain free of ML, hub, dataset, and provider
  dependencies.
- Validation and reference runs are offline and cannot accept URLs or secrets.
- No request-triggered downloads, telemetry, arbitrary model paths, dynamic
  code, pickle weights, or remote fallback are introduced.
- Candidate failures and reports never expose local paths, credentials, input
  text, prompts, provider bodies, or raw exception messages.
- Repository artifacts contain no weights, acquired datasets, raw model
  responses, provider credentials, or private Felhen/AIOS/customer data.
- Exact dependency and model acquisition remains operator-triggered, pinned,
  verified, and opt-in.
- An upstream license is recorded as evidence, not inferred from a package name
  or model card; uncertain licensing fails closed.

## Acceptance criteria

Phase 4C.1 is complete when:

1. canonical candidate and plan manifests validate offline with strict closed
   schemas, duplicate-key rejection, immutable revision rules, and exact-byte
   digest binding;
2. malicious fixture tests reject mutable revisions, unknown fields, bad URLs,
   unapproved metrics, non-zero remote budgets, dimension mismatches, duplicate
   IDs/keys, and conclusion-authority escalation;
3. all four reference adapters deterministically express their distinct work
   shapes with both declared sequence count and declared attention work, and
   report only `fixture_only`/`declared_analytic_model` results;
4. output is confined to an exclusively created 0700 directory beneath
   `.artifacts/universal-bakeoff/`, contains the exact three-file 0600
   allowlist and a verified self-digest manifest, and a second run cannot
   replace it;
5. tests prove no network function is reached and no credential environment is
   read by validate/reference commands;
6. wheel and sdist contain no new weights, datasets, run reports, secrets, or
   third-party source trees;
7. the default dependency graph is unchanged;
8. the full repository validation passes.

Phase 4C as a whole is complete only when the ADR records a disposition under
the evidence rules above. Phase 4C.1 may merge independently without implying
that later phases or a model choice are complete.

## Verification commands

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.universal_bakeoff validate
uv run python benchmarks/validate_default_environment.py
uv run python -m benchmarks.encoder_gate validate-registry
uv run python -m benchmarks.first_party_gate validate-protocol
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
git diff --check
```

Tests must monkeypatch socket and provider access to raise if called. No
verification command in Phase 4C.1 uses `--allow-network`, a provider key, a
model cache, or a paid endpoint.

## Rollout and rollback

Phase 4C.1 has no production deployment and does not modify the runtime package.
Rollout is a normal repository release of checkout-only benchmark code. Rollback
is a Git revert of its commit. Operator-created reports are immutable local
artifacts and remain operator-managed; rollback never deletes caches or output
directories.

Later local candidates remain behind optional dependencies and explicit
acquisition. Later remote controls remain behind exact run-plan budgets and
explicit execution. None becomes an automatic runtime fallback.

## Risks and mitigations

- **README benchmark bias:** re-run under one Saracura contract; never compare
  upstream headline numbers as if datasets and hardware were equivalent.
- **Teacher agreement mistaken for quality:** label the metric explicitly and
  allow it only in a later real-candidate plan and never in Phase 4C.1.
- **Synthetic overfitting:** require human-original evidence for final PT-BR
  selection and keep family-level splits.
- **Architecture favored by a narrow matrix:** include unseen taxonomy/domain,
  PT-BR and English plus large `Q/K` shapes in Phase 4C.1, then add permutation
  and ablation only when real candidates execute.
- **High-cardinality token overflow:** require explicit unsupported results and
  measure limits; never silently truncate choices.
- **Unsafe third-party code or weights:** pin and verify exact bytes, prefer
  Saracura-owned adapters, and require a separate isolation spec before any
  upstream environment can execute.
- **Provider drift or cost:** isolate remote controls, record provider/model
  identity, cap requests/tokens/USD, and default the budget to zero.
- **False promise of one state encoding:** scope that optimization to compiled
  backends and measure universal candidates according to their real computation.
- **Premature product expansion:** keep all Phase 4C code checkout-only until a
  separate runtime spec changes the public boundary.
