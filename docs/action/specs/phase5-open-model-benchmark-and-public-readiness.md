---
title: Phase 5 open-model benchmark and public readiness
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md
globalRef: qmd://saracura/docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-27
sourceRefs:
  - https://langwatch.ai/compare/jev-vs-all
  - https://huggingface.co/jaredpalmer/kev-4b
  - https://huggingface.co/shisa-ai/shisa-de-1
related:
  - README.md
  - docs/README.pt-BR.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/decisions/0004-model-first-product-direction.md
  - docs/action/specs/phase4c-universal-backend-bakeoff.md
  - docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md
  - docs/action/specs/phase4f1-shadow-evaluation-and-email-reference.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 5 open-model benchmark and public readiness

> **Current direction:** ADR 0004 supersedes this document's original
> runtime-and-lab positioning and makes a Saracura-owned Model Preview the next
> public milestone. The Gate A/Gate B manifest contracts below remain
> historical implementation contracts until a separately reviewed increment
> revises their schemas; they do not override the model-first roadmap filter.

## Outcome

Saracura will become a model-independent local decision runtime and evaluation lab with a PT-BR-first, bilingual evidence lane. The project will retain its compiled high-volume tier, stop treating the current Saracura-owned MiniLM bi-encoder projection ranker as the presumed product foundation, and evaluate modern open decision models before training another checkpoint.

The first public milestone is a usable developer preview backed by a reproducible local comparison and an explicitly named external checkpoint. The later model milestone is a Saracura-owned checkpoint that can be submitted to independent evaluators such as LangWatch. Neither milestone may claim that Saracura beats Jev or another model without comparable held-out evidence.

## New evidence

- LangWatch currently reports Jev across all 11 primary tasks while the strongest open models cover only 8 to 10 of them. Per-task comparisons, contamination markings, confidence intervals and hardware labels matter more than an unweighted average.
- The same comparison reports weak out-of-distribution results for the Laya family on tool routing, complaint routing, typed decisions and web-agent actions. Laya-typed's stronger typed-decision result is marked as trained on related data.
- Kev-4B declares a TypeSafe-style typed-decision contract from a Qwen 3.5 4B base with a LoRA adapter and pointer head. It is the first external candidate in the size range relevant to a local MacBook lane. Its checkpoint, base-model and adapter licenses remain upstream declarations pending the Phase 5B acquisition review.
- Shisa DE-1 shows that restricted-logit decision readout can be fast and accurate, but its roughly 48 GiB BF16 artifact and datacenter benchmark hardware make it a comparison reference rather than the first MacBook candidate.
- Saracura's first live read-only e-mail pilot restored 84% eligibility through deterministic projection but produced only 2/21 agreement with an independent operator label. This is a private, non-publicly-reproducible quality observation rather than a statistically powered benchmark, and it falsifies readiness of the current checkpoint for automation without creating a public comparative claim.

## Product decision

Saracura keeps one typed decision contract with three explicit roles:

1. `universal`: a pluggable question-conditioned backend for new schemas. The first Phase 5 candidate is Kev-4B or a later evidence-supported open model in the same practical local class. No individual checkpoint becomes a permanent dependency of the API.
2. `compiled`: a specialized encoder/head for stable, repeated, high-volume workflows where measured throughput and quality justify specialization.
3. `control`: remote or heavyweight candidates used only for comparison. Jev remains an external product control, and 25B-27B open models remain comparison references until local hardware evidence supports another disposition.

The current `saracura-universal-ranker.v0` MiniLM bi-encoder projection ranker remains a historical research baseline and a possible narrow fast path. Phase 5 makes no parameter-count claim for it. It is not the default quality candidate and must not absorb more tuning work until a frozen comparison identifies a task on which its footprint or throughput advantage is valuable and its quality is adequate.

## Public positioning

Use this bounded description until a checkpoint passes the release gates:

> Saracura is an open, local-first decision model project for fast typed decisions at scale, with PT-BR-first evaluation and a language-neutral API.

Do not describe the project as a Jev replacement, the fastest decision model, PT-BR optimized, production-ready, calibrated, or better than another candidate unless the corresponding evidence gate has passed. External models may be named inside detailed benchmark records as controls under the same protocol, but no individual third-party checkpoint is the public positioning anchor or recommended Saracura model.

## Independent benchmark strategy

An external evaluator should be able to add Saracura without bespoke reverse engineering. The submission bundle must contain:

- a public immutable checkpoint revision under an approved license;
- a public repository commit with a one-command, noninteractive inference path;
- a closed request/response schema for typed decisions, with an explicit statement of the supported subset; the current Saracura release supports only its declared Choice workflows;
- deterministic temperature-zero execution and stable option-order semantics;
- an artifact manifest with model, tokenizer, code, base-model and adapter digests;
- a model card disclosing architecture, training sources by class, known contamination, supported languages, context limit, hardware, quantization and limitations;
- a container or pinned environment that does not download mutable code at request time;
- self-reported results on public development sets plus a clear invitation to run undisclosed held-out sets;
- a maintainer contact and a short integration note mapping Saracura output to the evaluator's expected probability distribution.

Contact LangWatch only when that bundle exists. The first message should offer a ready-to-run artifact and ask whether it fits their intake, not ask them to design or debug the integration. Their decision and schedule remain external dependencies and are never part of Saracura's release gate.

## Evaluation protocol

### Dataset lanes

- `public_dev`: openly redistributable or self-authored cases used for integration, examples and prompt/readout selection. Scores are development evidence only.
- `sealed_ptbr_test`: human-authored or independently adjudicated PT-BR cases held outside the public repository until a declared evaluation release. Training, teacher calls, retrieval and threshold selection cannot read this lane.
- `sealed_bilingual_test`: paired or semantically equivalent PT-BR and English cases used to measure cross-locale parity. Translation alone is not automatically an independent label.
- `external_hidden`: evaluator-owned data that Saracura maintainers never inspect before the run.

Every durable result binds the exact dataset revision, checkpoint revision, code commit, hardware class, quantization, seed, batch/concurrency settings and prompt/readout contract.

### Required metrics

- accuracy or balanced accuracy plus macro-F1 for multiclass decisions;
- AUROC or catch rate at a declared false-alarm budget for binary ranking tasks;
- Brier score and ECE for probabilistic outputs, with calibration fitted on records disjoint from the test set;
- risk-coverage and abstention curves where abstention is supported;
- coverage, capacity failures and invalid-output rate;
- cold load time, warm p50/p95 latency, items/s, decisions/s, peak RSS, model bytes and hardware;
- option-order reversal sensitivity, deterministic repeat stability and PT-BR/English parity.

No single aggregate average determines the winner. A candidate disposition is made per task and per deployment envelope.

## Release gates

### Gate A: usable developer preview

The project may stop calling itself only an internal study when all of the following are true:

1. a clean machine can install the public package and execute a real open checkpoint through one documented command;
2. the checkpoint is pinned, license-reviewed and never downloaded implicitly by a request;
3. PT-BR and English examples complete without private Felhen code, data or configuration;
4. the public development benchmark is reproducible and reports quality, latency, throughput, memory and limitations without unsupported comparative claims;
5. the default path remains non-automating, and uncalibrated scores are not presented as confidence;
6. CI validates the lightweight default install and an opt-in model contract without downloading weights in ordinary tests.

Gate A permits a `0.1.0` developer preview using a third-party open checkpoint. It does not imply a Saracura-owned model.

### Gate B: Saracura-owned release candidate

1. a Saracura-owned checkpoint and model card are public under an approved license;
2. the checkpoint passes the sealed PT-BR and bilingual test lanes at thresholds frozen before reading test results;
3. calibration is fitted and evaluated on disjoint records and is bound to the exact checkpoint and configuration;
4. the release demonstrates at least one defensible advantage: better PT-BR quality, materially better local throughput/footprint, or a better quality-cost frontier for a declared workflow;
5. a LangWatch-style submission bundle passes from a clean environment.

Gate B authorizes an external benchmark submission and a public model claim limited to the measured advantage. It still does not authorize automatic high-risk actions.

## Phase graph and target elapsed time

| Phase | Deliverable | Target elapsed time | Dependency |
| --- | --- | --- | --- |
| 5A | Strategic reset, closed candidate/evaluation contracts and release ladder | 1 day | none |
| 5B | Pinned Kev-4B acquisition, opt-in out-of-process research adapter and managed local systems runs | 2-3 days | 5A |
| 5C | Frozen PT-BR/English dev and sealed-test protocol with 300-500 reviewed cases | 3-5 days, parallel with 5B | 5A |
| 5D | Baseline comparison, targeted Qwen 3.5 4B LoRA/pointer-head training and disjoint calibration | 3-5 days | 5B, 5C |
| 5E | Public checkpoint, model card, clean-room submission bundle and `0.1.0` release | 2-3 days | 5D |

With continuous execution, Gate A is targeted in 5-7 calendar days and Gate B in 10-15 calendar days. A credible external submission can be sent immediately after Gate B; publication by an external benchmark depends on its maintainer and is not schedulable by Saracura.

## Phase 5A implementation contract

Phase 5A updates the public direction without changing runtime behavior:

- record the new architecture priority in a reviewed ADR while preserving the two-tier decision API;
- update English and PT-BR public positioning consistently;
- add a closed, validation-tested Phase 5 candidate manifest that identifies Kev-4B as `planned_acquisition`, the current Saracura checkpoint as `historical_baseline`, Jev as `external_control`, and heavyweight open models as `comparison_only` without claiming local support;
- add a closed public-readiness policy manifest containing Gate A and Gate B criteria and the exact evidence classes each gate permits;
- add validators and offline tests for both manifests;
- make no network call, model download, dataset acquisition, runtime backend registration or quality claim.

### Closed Phase 5A manifest contracts

`benchmarks/manifests/phase5-open-model-candidates.v1.json` has exactly `schema_version`, `reviewed_at`, `allowed_dispositions`, `candidate_claim_vocabulary` and `candidates`. `schema_version` is exactly `phase5-open-model-candidates.v1`; `reviewed_at` is an ISO calendar date in `YYYY-MM-DD`; `allowed_dispositions` is exactly `historical_baseline`, `planned_acquisition`, `external_control`, `comparison_only` in that order; and `candidate_claim_vocabulary` is exactly `historical_research_baseline`, `candidate_for_evaluation`, `external_product_control`, `comparison_reference` in that order. Each candidate has exactly:

- `id`: stable lower-kebab identifier;
- `display_name`: non-empty display label;
- `disposition`: one allowed disposition;
- `architecture_class`: one of `bi_encoder_projection_ranker`, `option_marker_encoder`, `constrained_causal_lm`, `pointer_head_causal_lm`, `native_remote` or `mixture_of_experts_readout`;
- `source_url`: HTTPS public reference; the private historical baseline points to the public Saracura repository, never a private artifact;
- `declared_license`: upstream-declared SPDX identifier or `unknown`;
- `license_review`: exactly `pending`, `not_applicable_private_baseline` or `external_service` in Phase 5A; no value means approved;
- `revision`: immutable 40- or 64-character lowercase hexadecimal revision only when acquisition has already been reviewed, otherwise `null`;
- `revision_state`: `private_verified`, `unpinned`, `external_service` or `comparison_reference`;
- `local_execution`: `historical_only`, `planned`, `not_supported` or `not_applicable`;
- `evidence_authority`: `private_quality_observation`, `upstream_claims_only`, `external_control_only` or `comparison_context_only`;
- `allowed_claims`: an ordered subset of `candidate_claim_vocabulary`;
- `notes`: one bounded public sentence with no local path, digest or private identifier.

Disposition consistency is exact:

- `historical_baseline` requires `architecture_class: bi_encoder_projection_ranker`, `revision: null`, `revision_state: private_verified`, `license_review: not_applicable_private_baseline`, `local_execution: historical_only`, `evidence_authority: private_quality_observation`, and may allow only `historical_research_baseline`;
- `planned_acquisition` requires `revision: null`, `revision_state: unpinned`, `license_review: pending`, `local_execution: planned`, `evidence_authority: upstream_claims_only`, and may allow only `candidate_for_evaluation`;
- `external_control` requires `revision: null`, `revision_state: external_service`, `license_review: external_service`, `local_execution: not_applicable`, `evidence_authority: external_control_only`, and may allow only `external_product_control`;
- `comparison_only` requires `revision: null`, `revision_state: comparison_reference`, `license_review: pending`, `local_execution: not_supported`, `evidence_authority: comparison_context_only`, and may allow only `comparison_reference`.

Phase 5A intentionally records no acquired immutable revision. A future manifest version may populate `revision` only after a reviewed acquisition and must reject branch names, tags, `main`, URLs, paths and mixed-case digests.

`benchmarks/manifests/phase5-public-readiness.v1.json` has exactly `schema_version`, `claim_vocabulary` and `gates`. `schema_version` is exactly `phase5-public-readiness.v1`. `claim_vocabulary` is exactly `research_only`, `usable_developer_preview`, `third_party_checkpoint_supported`, `saracura_owned_checkpoint`, `ptbr_measured`, `bilingual_measured`, `calibrated_for_declared_protocol`, `comparative_claim_for_declared_task`, `external_submission_ready`, `production_ready`, `automation_authorized`. Each gate has exactly `id`, `status`, `required_evidence`, `allowed_claims` and `forbidden_claims`; `status` is `not_met` in Phase 5A.

- Gate A has `id: gate_a_developer_preview` and requires exactly these ordered evidence classes: `clean_machine_install`, `pinned_license_reviewed_acquisition`, `public_dev_quality_report`, `local_systems_report`, `offline_ci_contract`, `public_ptbr_en_examples`, `non_automation_safety_contract`. It allows exactly `research_only`, `usable_developer_preview` and `third_party_checkpoint_supported` and forbids every other claim.
- Gate B has `id: gate_b_saracura_checkpoint` and requires every Gate A evidence class in the same order followed by exactly `saracura_owned_checkpoint`, `sealed_ptbr_test`, `sealed_bilingual_test`, `disjoint_calibration_test`, `clean_external_submission_bundle`. It allows the Gate A claims plus `saracura_owned_checkpoint`, `ptbr_measured`, `bilingual_measured`, `calibrated_for_declared_protocol`, `comparative_claim_for_declared_task` and `external_submission_ready`. It forbids `production_ready` and `automation_authorized`.
- Gate B's allowed claims must be a strict superset of Gate A's. Every vocabulary entry must occur in exactly one of each gate's `allowed_claims` or `forbidden_claims`; duplicates and omissions fail validation. `calibrated_for_declared_protocol` requires the disjoint-calibration evidence class; `comparative_claim_for_declared_task` requires sealed test evidence; no gate in this revision may allow `production_ready` or `automation_authorized`.

`benchmarks.validate_manifests` routes both exact schema-version strings above to their dedicated validators. Any integer, unknown revision or misplaced copy under `src/saracura` fails. The existing Phase 4E.4 `187/198` versus `139/198` statement remains historical synthetic evidence with its existing scope and integrity markers; Phase 5A must not broaden it into an unqualified superiority claim or remove its validation markers.

README tests reject ungated positioning terms in the Phase 5 overview: `Jev replacement`, `fastest decision model`, `production-ready`, `PT-BR optimized` and unqualified `calibrated`, plus the PT-BR equivalents `substituto do Jev`, `modelo de decisão mais rápido`, `pronto para produção`, `otimizado para PT-BR` and unqualified `calibrado`. Historical evidence sections may retain scoped factual wording when tests bind it to an explicit `synthetic`, `research-only` or `not production evidence` qualifier.

## Safety and provenance invariants

- Default installation remains lightweight, offline and free of model weights.
- Model acquisition is explicit, pinned, digest-verified and separate from inference.
- No upstream repository code is imported or executed as trusted code.
- Private e-mail contents, provider identifiers and private evaluation records never enter the public repository or model artifacts.
- Teacher-generated data is labeled as such and cannot become human evidence by review wording alone.
- Test records are not used for training, retrieval, threshold selection, temperature fitting or prompt/readout selection.
- The public runtime never gains an automatic remote fallback, arbitrary URL, arbitrary model path or request-triggered download.
- A model's declared absence of training data is not treated as proof of non-contamination.

## Rollout and rollback

Each phase is an independent PR and can be reverted without invalidating earlier immutable evidence. Candidate support remains opt-in until its own reviewed phase passes. If Kev-4B fails the MacBook envelope or PT-BR development threshold, Phase 5 records that result and evaluates the next qualified candidate; it does not silently change the selected checkpoint. If no candidate passes Gate A, Saracura remains a research alpha and publishes the failed comparison rather than weakening the gate.

## Validation for Phase 5A

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src benchmarks tests
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.validate_default_environment
uv run python -m benchmarks.encoder_gate validate-registry
uv run python -m benchmarks.first_party_gate validate-protocol
uv build
uv run python -m benchmarks.inspect_wheel dist/*.whl dist/*.tar.gz
```
