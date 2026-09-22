---
title: Phase 4C.2 local universal candidate adapters
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4c2-local-candidate-adapters.md
globalRef: qmd://saracura/docs/action/specs/phase4c2-local-candidate-adapters.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs:
  - https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af
  - https://huggingface.co/convaiinnovations/laya-multilingual/tree/052592a15d198d9ad47da779604259b10b47b7aa
  - https://huggingface.co/jhu-clsp/mmBERT-base/tree/c5955035435e2bf121cde7f3c8863ef52ff35d82
  - https://github.com/rupeshpoojary9/poorjev/tree/7e684e95db13b90238c63e6ab39e1a016263a168
  - https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli/tree/8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c
  - https://huggingface.co/microsoft/mdeberta-v3-base/tree/a0484667b22365f84929a935b5e50a51f71f159d
  - https://huggingface.co/datasets/facebook/xnli/tree/b8dd5d7af51114dbda02c0e3f6133f332186418e
  - https://huggingface.co/datasets/nyu-mll/multi_nli/tree/da70db2af9d09693783c3320c4249840212ee221
related:
  - AGENTS.md
  - SECURITY.md
  - docs/action/specs/phase4c-universal-backend-bakeoff.md
  - docs/action/specs/phase2b-encoder-candidate-gate.md
  - docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4C.2 local universal candidate adapters

## Context

Phase 4C.1 established a closed, offline harness for comparing computation shapes, but it executed no learned universal candidate. Phase 4C.2 adds two opt-in local candidates on the MacBook: one option-marker encoder and one pairwise NLI baseline. They remain checkout-only benchmark adapters and never become `DecisionEngine` runtime backends in this phase.

The option-marker candidate is the exact Laya multilingual checkpoint because its public contract supports PT-BR and dynamic choices. The poorjev default checkpoint, `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, is English-only, so running it on the PT-BR matrix would conflate architecture with language coverage. The NLI candidate therefore uses poorjev's pairwise architecture with the separately pinned multilingual `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` checkpoint. It is named `mdeberta-nli`, not `poorjev-nli`, and results may not be presented as poorjev product results.

## Current truth

- `decision-backend-candidates.v1` records seven research dispositions but contains no acquisition allowlist, model files, or executable adapter authority.
- The default installation is offline and lightweight. Torch, Transformers, Safetensors, Platformdirs, and Psutil exist only in opt-in extras.
- The existing encoder acquisition lane proves bounded anonymous downloads, exact-byte verification, private cache permissions, immutable promotion, symlink and hardlink rejection, and descriptor-bound loading for flat snapshots. Phase 4C.2 needs a new recursive implementation because Laya's allowlist contains `encoder/` and `tokenizer/` subdirectories.
- The Phase 4C.1 plan contains eight self-authored deterministic computation-shape scenarios across PT-BR and English with `N`, `Q`, and `K` dimensions. Its generated inputs are byte-framed analytic fixtures rather than natural-language states, instructions, and choices, so Phase 4C.2 must preserve the eight dimension tuples while replacing their materialization with a new textual generator.
- Laya multilingual ships without fitted temperature buckets and explicitly warns that its raw probabilities are over-confident. No Laya confidence may be described as calibrated in this phase.
- The chosen multilingual NLI model claims cross-lingual transfer to 100 languages, but Portuguese is not part of its XNLI evaluation list. PT-BR behavior is therefore measured research evidence, not an upstream guarantee.

## Goal

Implement a registry-v2 acquisition and execution lane that can safely download, verify, load, and smoke-test the exact Laya multilingual and multilingual DeBERTa NLI artifacts on CPU or Apple MPS under an explicit dtype policy. Produce immutable local reports containing compatibility and systems-performance evidence without publishing input text, weights, quality claims, calibration claims, or an architecture winner.

## Non-goals

This phase does not modify the installed runtime API, register a universal backend, train or fine-tune weights, fit temperature maps, use private or customer data, read Phase 4B blind-test data, execute upstream packages, publish model files, add provider calls, choose a production architecture, or claim PT-BR quality. It does not run Von because Laya already represents the same option-marker family and no independent checkpoint provenance has yet justified the extra acquisition. It does not run the poorjev default English-only checkpoint.

## Selected identities

### Laya multilingual option marker

| Axis | Pinned value |
| --- | --- |
| Candidate ID | `laya-multilingual` |
| Source implementation | `NandhaKishorM/laya@573e5b62696ba441230cd6be71d593331b5d23af` |
| Source license | Apache-2.0 |
| Model ID | `convaiinnovations/laya-multilingual` |
| Model revision | `052592a15d198d9ad47da779604259b10b47b7aa` |
| Model-card license | Apache-2.0 |
| Declared base encoder | `jhu-clsp/mmBERT-base` |
| Base revision reviewed | `c5955035435e2bf121cde7f3c8863ef52ff35d82` |
| Base model-card license | MIT |
| Weight tensor dtypes | 169 `F16` tensors and one `F32` tensor in the pinned Safetensors header |
| Loader | Transformers `ModernBertModel` plus a Saracura-owned decision head compatible with the pinned public source contract; no import of `laya` |

Exact acquisition allowlist:

| File | Bytes | SHA-256 | Role |
| --- | ---: | --- | --- |
| `encoder/config.json` | 1,938 | `83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4` | closed encoder config |
| `rl_agent_config.json` | 472 | `25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6` | closed decision-head config |
| `tokenizer/tokenizer_config.json` | 502 | `424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7` | reviewed tokenizer metadata |
| `tokenizer/tokenizer.json` | 34,363,188 | `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f` | inert fast-tokenizer graph |
| `model.safetensors` | 643,835,514 | `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204` | encoder and decision-head weights |

The loader validates the closed config values needed to construct the model, including `model_type=modernbert`, hidden size 768, 22 layers, 12 heads, vocabulary 256,000, two decision-head layers, `max_len=1024`, `head_max_len=256`, and identity temperatures only. Unknown executable metadata, `auto_map`, remote code, or an unexpected architecture fails closed. The tokenizer is constructed directly from the verified tokenizer JSON with explicit special tokens; the shipped file is never rewritten. The built-in encoder implementation comes from the exact locked Transformers version and is recorded in every report; Saracura owns only the decision head and orchestration around it.

### Multilingual DeBERTa NLI

| Axis | Pinned value |
| --- | --- |
| Candidate ID | `mdeberta-nli` |
| Architecture reference | `rupeshpoojary9/poorjev@7e684e95db13b90238c63e6ab39e1a016263a168` |
| Reference source license | MIT |
| Model ID | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |
| Model revision | `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c` |
| Model-card license | MIT |
| Declared base encoder | `microsoft/mdeberta-v3-base` |
| Base revision reviewed | `a0484667b22365f84929a935b5e50a51f71f159d` |
| Base model-card license | MIT |
| Weight tensor dtypes | 202 `F16` tensors and one `I64` tensor in the pinned Safetensors header |
| Loader | Transformers `DebertaV2ForSequenceClassification`; no import of `poorjev` |

Exact acquisition allowlist:

| File | Bytes | SHA-256 | Role |
| --- | ---: | --- | --- |
| `config.json` | 1,066 | `4e4430c95100d613df80fa01276f231931e1b535557cf62fbb3cc50323e50cca` | closed sequence-classification config |
| `tokenizer_config.json` | 1,256 | `df9fd3a4482cc4a866788244824573fab8e74dadd1609d4f0e5997f9e96921dc` | reviewed tokenizer metadata |
| `special_tokens_map.json` | 286 | `9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9` | closed special-token map |
| `tokenizer.json` | 16,331,396 | `3aca3ce69a0a35aeb144a52c4f1d41c4246b8785f8f398315cc8fb6b24057810` | inert fast-tokenizer graph |
| `model.safetensors` | 557,652,046 | `65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e` | NLI weights |

The loader requires `model_type=deberta-v2`, `architectures=[DebertaV2ForSequenceClassification]`, hidden size 768, 12 layers, 12 heads, vocabulary 251,000, and the exact label map `entailment=0`, `neutral=1`, `contradiction=2`. It constructs a fast tokenizer directly from verified bytes and never reads `training_args.bin`, ONNX files, SentencePiece code, hub metadata, or a mutable cache. The built-in model implementation comes from the exact locked Transformers version and is recorded in every report.

## Provenance and research-use disposition

The large-file sizes and SHA-256 values above come from the immutable Hugging Face model-revision API and LFS metadata. The small-file values were independently recomputed from the immutable raw URLs on 2026-09-22. Acquisition recomputes every SHA-256 over downloaded bytes before promotion, so API metadata is never the final trust decision. Registry tests also query no network and use only the reviewed constants.

License evidence is recorded on separate axes for source code, checkpoint card, declared base model, tokenizer provenance, and disclosed training datasets. The Laya checkpoint card is Apache-2.0 and the mmBERT base card is MIT, but mmBERT documents a Gemma 2 tokenizer; its independent tokenizer licensing provenance is not established by this review. The NLI checkpoint and mDeBERTa base cards are MIT, while its disclosed XNLI and MNLI training sources have mixed or incomplete license metadata. These uncertainties do not authorize redistribution or commercial use. They permit only local, non-redistributed research execution in Phase 4C.2 and force `licensing_disposition=research_only_unresolved_provenance`. A production recommendation, packaged weights, hosted redistribution, or commercial claim remains blocked pending human legal/provenance review.

## Dependency and dtype policy

Add a checkout-only `universal-local` extra containing `platformdirs>=4.3,<5`, `psutil>=6.1,<7`, `safetensors>=0.5,<1`, `torch>=2.5,<3`, `transformers>=4.48,<5`, and direct `tokenizers>=0.21,<1`. Its lockfile resolves the implementation versions; every report records Python, Torch, Transformers, Tokenizers, Safetensors, OS version, architecture, logical device type, process thread count, batch size, attention implementation, and execution dtype without hostnames, serials, usernames, or paths. The default dependency group remains unchanged.

Both checkpoints contain predominantly FP16 weights, but this source-contract increment constructs all model parameters as FP32 on both `cpu` and `mps` and explicitly casts verified FP16 parameter tensors into them during loading. This matches the pinned Laya construction path and avoids an unreviewed half-precision MPS optimization. Laya's only F32 source tensor must be the key `temperature` with shape `[3]`; it remains F32. mDeBERTa's only I64 tensor must be `deberta.embeddings.position_ids` with shape `[1,512]`; the loader validates that exact auxiliary tensor and its monotonic `0..511` contents, removes it from the parameter state dict because Transformers 4.57.6 owns the equivalent non-persistent buffer, and then applies strict loading to every remaining key. Any other non-FP16 tensor, missing auxiliary tensor, or remaining state-dict mismatch fails closed.

Reports emit the complete `source_weight_dtypes` count map rather than a misleading scalar, plus `execution_dtype=fp32`. Laya attention is fixed to `sdpa`; mDeBERTa is fixed to `eager` because the pinned DeBERTaV2 implementation rejects SDPA under the locked Transformers 4.57.6 runtime. For MPS, `PYTORCH_ENABLE_MPS_FALLBACK` must be absent or exactly `0`; the run records `mps_fallback_enabled=false` and fails if an operation cannot execute on MPS. No automatic dtype, quantization, device, attention, or backend fallback is allowed. Preflight requires bounded free disk and estimated available memory for the selected candidate and FP32 construction before model allocation.

## Registry v2

Add `benchmarks/manifests/decision-backend-candidates.v2.json` and route it through the manifest validator. The root contains exactly `schema_version`, `reviewed_at`, `supersedes_registry_sha256`, `predecessor_map`, and `candidates`; `schema_version` is `decision-backend-candidates.v2`, `supersedes_registry_sha256` is `587f745dabffacfb4ba5fa676b88fb1b3768db701ed2121fd0007531ed6adf71`, and `predecessor_map` is exactly `{"poorjev-nli":"mdeberta-nli"}`. Candidate entries retain the compatible Phase 4C.1 identity fields and add exactly:

- `model_source_url`, `model_license`, `model_license_evidence_url`, `base_model_id`, `base_model_revision`, `base_model_license`, and `base_model_license_evidence_url`;
- `tokenizer_provenance`, `training_data_provenance`, and `licensing_disposition`;
- `loader_family`: `laya_option_marker`, `mdeberta_nli`, or `not_executable`;
- `acquisition_state`: `eligible`, `existing_reference`, `deferred`, `excluded`, or `not_applicable`, replacing the v1 enum rather than adding an overlapping execution state;
- `weight_format`: `safetensors`, `not_reviewed`, or `not_applicable`;
- `dependency_contract`: `universal-local.v1` for eligible candidates and `null` otherwise;
- `source_weight_dtypes`, an exact non-empty dtype-count map only for eligible checkpoints;
- `files`, an ordered non-empty list only for eligible candidates, with exact `path`, `bytes`, `sha256`, and `role`;
- `acquisition_contract_digest`, the lowercase SHA-256 of the canonical candidate acquisition subset for eligible candidates and `null` otherwise;
- `conformance_state`: `pending`, `source_contract_reviewed`, or `not_applicable`;
- `conformance_vector_sha256`, non-null only when `conformance_state=source_contract_reviewed` and `null` otherwise;
- `architecture_attribution`: `positive_compatibility_only` or `not_applicable`.

The v2 registry keeps the seven conceptual candidates but renames `poorjev-nli` to `mdeberta-nli` through an explicit predecessor map. Only `laya-multilingual` and `mdeberta-nli` are eligible for research-only acquisition. `saracura-compiled` remains an existing specialized reference; Von and all remote candidates remain deferred; DiffusionGemma remains excluded. For the five non-eligible candidates, the new scalar/object model, provenance, dependency, dtype, file, and digest fields are `null`, while `loader_family=not_executable`, `conformance_state=not_applicable`, and `architecture_attribution=not_applicable`. Their compatible v1 fields remain byte-for-byte unchanged except the deliberately replaced `acquisition_state`, which follows the v2 truth table. Registry v1 remains the immutable source for the Phase 4C.1 analytic plan, while registry v2 is the sole source for real local candidate execution. The validator binds every ID to an exact truth table, verifies the v1 predecessor digest and all unchanged shared fields, and rejects unknown fields, duplicate keys, mutable revisions, missing provenance evidence, duplicate file paths, path traversal, non-allowlisted extensions, mismatched roles, and any eligible candidate without exact files.

File URLs are not stored as a second source of truth. Acquisition constructs each literal URL from the exact `model_id`, 40-character `model_revision`, and percent-encoded allowlisted path using `https://huggingface.co/<model_id>/resolve/<model_revision>/<path>`. The per-candidate `acquisition_contract_digest` is SHA-256 over the RFC 8785 canonical acquisition subset: ID, model identity, weight format, dtype map, loader family, file allowlist, and `dependency_contract=universal-local.v1`. Notes, review dates, deferred candidates, and unrelated registry edits are excluded. Cache paths include this digest, so a changed acquisition contract creates a new immutable snapshot rather than invalidating or colliding with an earlier one.

## Acquisition boundary

Add a checkout-only CLI under `benchmarks.universal_local`:

```bash
python -m benchmarks.universal_local validate-registry
python -m benchmarks.universal_local acquire --candidate laya-multilingual --allow-network
python -m benchmarks.universal_local acquire --candidate mdeberta-nli --allow-network
python -m benchmarks.universal_local verify --candidate laya-multilingual
python -m benchmarks.universal_local freeze-conformance --candidate laya-multilingual --output <new-name>
python -m benchmarks.universal_local run --candidate laya-multilingual --device mps --warmups 1 --iterations 3 --output <new-name>
```

The CLI accepts no URL, revision, file list, destination path, token, proxy, credential, or arbitrary path. `--output` is a validated new child name, not a path, and resolves only under the command's fixed artifact root. Acquisition uses anonymous HTTPS to the constructed immutable URLs, disables environment proxies, and permits redirects only when the scheme remains HTTPS and the normalized host is exactly `huggingface.co`, `hf.co`, or a subdomain of one of those two suffixes. A signed query is accepted only on a validated redirect target, is never persisted or logged, and never changes the allowlisted file identity; fragments and userinfo always fail. A bounded preflight verifies TLS, final content length or content range, free disk, and cumulative byte budget before the first large body is accepted.

Phase 4C.2 implements a new recursive snapshot writer rather than reusing the flat Phase 2B promotion functions. It creates each path component by directory file descriptor with `O_DIRECTORY`, `O_NOFOLLOW`, owner-only mode 0700, and identity checks before descending. Files are created by descriptor at mode 0600, with one link, exact size, and exact digest. The final tree contains exactly the allowlisted nested directories and files plus one root completion marker bound to `acquisition_contract_digest`. Promotion is atomic and never overwrites. Partial downloads remain under a non-final random staging directory; this increment may restart rather than resume a failed large file, but it must never treat a partial file as valid. A changed contract resolves to a different final directory, while a conflicting directory for the same digest fails closed with a bounded recovery instruction.

Every later load receives a descriptor-bound verification receipt created from open directory and file descriptors after recursively validating ownership, modes, exact tree shape, sizes, hashes, link counts, and directory identity. Symlinks, hardlinks, special files, unexpected files or directories, world/group permissions, mutation during verification, and stale or artisanal receipts fail closed. The offline `verify` command exercises this path without importing ML dependencies. Model classes receive verified bytes or descriptors and may not rediscover a hub cache or resolve caller paths.

## Adapter contract

Add two checkout-only adapters under `benchmarks/`, never under `src/saracura/backends/`.

The Laya adapter materializes one sequence per independent state and question, renders all choices into option-marker slots, batches compatible sequences, and scores only verified marker positions. Its input construction is a Saracura-owned implementation of the pinned public architecture contract around the Transformers built-in `ModernBertModel`; it does not reimplement ModernBERT attention. It inserts token IDs, rather than joining delimiter-sensitive strings, in this exact order: `CLS`; tokens for the fixed prefix `choice question: ` followed by the unchanged instruction; `SEP`; for each choice in declared order, the tokenizer's `MASK` ID followed by tokens for one leading space plus the unchanged choice description; `SEP`; tokens for the state frame `S<subject_utf8_byte_count>:<subject>B<body_utf8_byte_count>:<body>`; final `SEP`. Decimal lengths have no leading zero except the value zero, and the parser consumes the declared UTF-8 bytes before reading the next tag. The arbitrary `choice_id` never enters model text. `CLS`, `SEP`, and `MASK` are verified structural IDs, so this template preserves the repository byte-framing invariant without a delimiter carve-out.

The adapter preserves Unicode data and returns `unsupported_input` if any `subject`, `body`, instruction, or choice description contains any tokenizer-declared special-token literal rather than silently rewriting or interpreting the value structurally. The 256-token head budget includes prefix, instruction, markers, and choice descriptions. Each description is initially capped at 48 tokens as in the pinned source; if the head still exceeds 256, the adapter applies the pinned equal-share rule but requires at least four description tokens plus the marker for every option. A violation returns `unsupported_capacity`; no choice is removed. Any accepted description truncation sets `supported_with_choice_truncation` and reports retained and dropped counts. The remaining room up to 1,024 contains state tokens. When a non-empty state exceeds that room, right-side state tokens are dropped, the run remains `supported_with_state_truncation`, and original, retained, and dropped token counts are reported. Zero retained state tokens returns `unsupported_capacity`; no truncation is hidden. If both forms occur, the closed status is `supported_with_state_and_choice_truncation` and both counters remain populated.

The NLI adapter materializes one premise/hypothesis pair per state, question, and choice. The Phase 4C.2 textual generator produces a closed state with `subject` and `body`; the premise is exactly `S<subject_utf8_byte_count>:<subject>B<body_utf8_byte_count>:<body>`. Each hypothesis is exactly `I<instruction_utf8_byte_count>:<instruction>D<description_utf8_byte_count>:<description>` followed by the locale-specific assertion `A descrição acima é a opção correta.` or `The description above is the correct option.` Other locales are unsupported. The arbitrary `choice_id` is excluded. Decimal lengths follow the same canonical rule, values remain unchanged data, and every tokenizer-declared special-token literal in a source field causes `unsupported_input`, so quotes or delimiter-looking text cannot reframe a segment.

The NLI tokenizer dynamically pads each microbatch to its longest member under a hard pair maximum of 512 tokens. The complete hypothesis is mandatory; after accounting for pair-special tokens and the hypothesis, the adapter preserves the complete subject frame and truncates only the right side of the body payload to the remaining premise budget. A hypothesis that cannot fit, a subject frame that cannot fit, or zero retained body tokens returns `unsupported_capacity`; accepted body truncation reports original, retained, and dropped token counts. It batches at most 16 pairs, reads the exact entailment index from the validated truth table, upcasts the three logits to FP32, computes a three-label softmax per pair, selects the entailment probability, and then performs FP32 linear normalization of the non-negative entailment probabilities by their sum across choices. A non-finite or zero sum returns `numerical_error`. This exact rule is frozen in the conformance vector because it matches the pinned poorjev implementation and intentionally differs from the Transformers zero-shot pipeline default. The resulting values are `uncalibrated_ranking_weights`, never calibrated probabilities or threshold authority.

Both adapters require an explicit `--device cpu|mps`; MPS unavailability fails without fallback. Only one model may be resident per process. They run under `torch.inference_mode()`, synchronize MPS before and after each measured region, fix the process thread count before model construction, and expose no dynamic code, pickle, hub, telemetry, or remote fallback. A run records process RSS and, on MPS, both current and driver-allocated MPS memory. Memory units are normalized explicitly across macOS and Linux.

## Conformance and attribution

Phase 4C.2 does not execute upstream packages because the parent spec requires a separate isolation review for that route. Add `benchmarks/fixtures/universal-local-conformance.v1.json`, with exactly two candidate entries bound to registry-v2 candidate identity and acquisition contract digest. Each entry covers canonical input rendering, token IDs, marker or pair positions, tensor shapes, output schema, scoring rule, and a public card example. The final registry stores the exact vector-file SHA-256, and every report stores both the vector digest and pass/fail state. Passing these vectors establishes local source-contract compatibility but not numerical parity with the upstream package.

The registry/acquisition increment first lands both eligible candidates as `conformance_state=pending` with a null vector digest; `acquire` and `verify` work, but `run` fails closed. After that merge, the operator acquires the verified snapshots. During the loader increment, a maintainer-only `freeze-conformance` command reads those receipts and writes a new named candidate fragment only under `.artifacts/universal-local-conformance/`; it accepts no arbitrary path and never overwrites. The reviewed fragments are combined into the committed two-entry fixture, the registry changes both candidates to `source_contract_reviewed` with the fixture digest, and plan v2 binds that final registry. CI validates the fixture digest, closed schema, registry linkage, and renderer logic using fake tokenizers; it does not regenerate real token IDs or contact a model host. The local freeze receipt records the candidate acquisition digest and locked library versions without cache paths or checkpoint bytes.

Until a later isolated parity run records matching logits within a declared tolerance, a successful vector run reports `conformance_state=source_contract_reviewed` and `architecture_attribution=positive_compatibility_only`. A successful real run may show that the exact candidate can execute under Saracura's boundary. A failure or weak result may not be generalized into a negative claim about Laya, poorjev, option-marker encoders, or NLI as an architecture family.

## Real-candidate run plan and metrics

Add `universal-bakeoff-plan.v2.json` with exactly `schema_version=universal-bakeoff-plan.v2`, `plan_id=phase4c2-local-candidates`, `generator_revision=phase4c.2-text-fixture-generator.v1`, `evidence_lane=synthetic_research`, `candidate_registry_sha256`, `predecessor_plan_sha256`, `dataset_artifact=null`, `remote_budget`, `generator_contract`, `scenarios`, and `report_policy`. `predecessor_plan_sha256` is `535165cdf6eacc3bd73bafecaf2ef2c82050890d3525045667623177614be08d`; `candidate_registry_sha256` binds the exact new v2 bytes; and `remote_budget` contains exactly zero `requests`, `tokens`, and `usd`. The predecessor digest binds the exact v1 plan, but v2 materialization is intentionally different.

Plan v2 preserves the eight v1 `(locale, domain, novelty, N, Q, K)` tuples, not their analytic byte fixtures. The new generator creates deterministic, self-authored natural-language records with exact `subject`, `body`, `instruction`, `choice_id`, and `description` fields. PT-BR scenarios contain native Brazilian Portuguese; the English scenario contains English. `generator_contract` contains exactly `reserved_labels`, `state_byte_profiles`, `choice_byte_profiles`, and `sentence_banks`; it is data, not Python entry points. `support` and `inbox_triage` each have locale-keyed, versioned subject, body, instruction, and choice-description sentence banks committed inside that object, and `rule_generated` records combine only those banks by seed. No model-generated or copied text is allowed.

Scenarios declare `state_length_profile` and `choice_length_profile` from the closed enums `short`, `medium`, and `long`. State `subject+body` UTF-8 byte totals are `short=80..240`, `medium=400..900`, and `long=1800..2800`; choice-description totals are `short=8..40`, `medium=41..96`, and `long=97..180`. Before importing any ML library, the validator materializes all records and enforces those inclusive byte ranges, field counts, locale, NFC, and the absence of its reserved framing labels in source strings. After the verified tokenizer loads but before model construction, the adapter computes token lengths and either accepts them under its frozen cap or records `unsupported_capacity`; the plan does not pretend tokenizer-independent token bounds exist.

Every scenario contains exactly `id`, `locale`, `domain`, `novelty`, `N`, `Q`, `K`, `input_mode`, `input_seed`, `state_length_profile`, and `choice_length_profile`; `input_mode` is the closed enum `self_authored|rule_generated`. The eight tuples remain:

| # | Locale | Domain | Novelty | N | Q | K | State profile | Choice profile |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | `pt-BR` | `support` | `seen_taxonomy` | 1 | 1 | 5 | short | short |
| 2 | `pt-BR` | `support` | `seen_taxonomy` | 8 | 1 | 5 | medium | short |
| 3 | `pt-BR` | `support` | `unseen_taxonomy` | 1 | 10 | 5 | medium | medium |
| 4 | `pt-BR` | `inbox_triage` | `unseen_taxonomy` | 8 | 10 | 20 | medium | medium |
| 5 | `en` | `support` | `seen_taxonomy` | 1 | 1 | 5 | short | short |
| 6 | `pt-BR` | `support` | `seen_taxonomy` | 32 | 1 | 2 | long | medium |
| 7 | `pt-BR` | `inbox_triage` | `unseen_taxonomy` | 1 | 50 | 20 | long | long |
| 8 | `pt-BR` | `support` | `seen_taxonomy` | 1 | 1 | 77 | medium | long |

This is a systems and capacity matrix, not a quality set. The content is intentionally not labeled with correct answers. The expected Laya coverage is scenarios 1 through 7, with explicit state or choice truncation where required; scenario 8 must return `unsupported_capacity` because 77 markers plus the four-description-token minimum cannot fit its head budget. The expected NLI coverage is all eight scenarios, subject only to its declared pair-capacity checks. Any departure is recorded rather than silently dropping a cell. Phase 3B synthetic holdout, Phase 4B human records, private data, and external datasets remain unreachable.

`report_policy` contains exactly `status=real_local_candidate`, `metric_authority=measured_systems_only`, `conclusion_authority=systems_compatibility_only`, and `allowed_metrics`. The exact metric allowlist is: `candidate_id`, `model_id`, `model_revision`, `acquisition_contract_digest`, `conformance_vector_sha256`, `conformance_state`, `architecture_attribution`, `plan_sha256`, `python_version`, `torch_version`, `transformers_version`, `tokenizers_version`, `safetensors_version`, `os_version`, `architecture`, `device_type`, `execution_dtype`, `source_weight_dtypes`, `process_threads`, `attention_implementation`, `mps_fallback_enabled`, `microbatch_size`, `load_duration_ms`, `warmup_duration_ms`, `latency_samples_ms`, `latency_p50_ms`, `latency_p95_ms`, `decisions_per_second`, `peak_rss_bytes`, `mps_current_allocated_bytes`, `mps_driver_allocated_bytes`, `input_token_count_min`, `input_token_count_max`, `state_tokens_original`, `state_tokens_retained`, `truncated_state_tokens`, `choice_tokens_original`, `choice_tokens_retained`, `truncated_choice_tokens`, `supported_count`, `supported_with_state_truncation_count`, `supported_with_choice_truncation_count`, `supported_with_state_and_choice_truncation_count`, `unsupported_count`, `failed_count`, `deterministic_repeat_match`, `comparison_eligible`, and `error_code`.

The report emits no answer or ranking digest. It may compare systems performance and output stability only after checking identical hardware and dependency axes. It may not emit raw states, prompts, token IDs, raw scores, model tensors, quality metrics, accuracy, F1, ECE, Brier score, risk coverage, confidence thresholds, teacher agreement, winner, recommendation, or calibrated probabilities.

The run command is `python -m benchmarks.universal_local run`. It accepts only a candidate ID, explicit device, warm-up count from 1 through 3, iteration count from 1 through 10, and a new child output name under `.artifacts/universal-bakeoff/`. The canonical comparable MPS run is exactly one warm-up and three measured iterations across all eight scenarios. Other permitted MPS counts are diagnostic and set `comparison_eligible=false`. CPU accepts only one warm-up and one measured iteration, executes scenarios 1 and 5 as a compatibility smoke, and is always excluded from cross-candidate performance comparison. Output follows the Phase 4C.1 immutable three-file contract and adds `conclusion_authority=systems_compatibility_only` plus the conformance and attribution state. A verified candidate that cannot load or execute writes a sealed `blocked` result with one bounded error code and no exception text; it never disappears from the intended matrix.

## Implementation phases

### Phase 4C.2a: registry and acquisition

Implement registry v2 with pending conformance, strict validation, candidate lookup, the new recursive safe-acquisition path, verification receipts, `validate-registry`, `acquire`, and `verify`, malicious-fixture tests, default-environment isolation, and package-exclusion checks. Add the `universal-local` optional extra without changing the default environment. Update the repository benchmark invariant so the explicitly checkout-only Phase 4C.2 candidate adapters may execute learned models under `benchmarks/`, while keeping all installed runtime backends restricted to the compiled specialized contract. No weight download occurs in tests or CI. After merge, the operator acquires and verifies both candidates locally.

### Phase 4C.2b: loaders and source-contract vectors

Implement direct verified-byte tokenizers and built-in model construction, strict Safetensors loading, the reviewed CPU/MPS dtype paths, exact input rendering, `freeze-conformance`, the frozen source-contract fixture and final registry linkage, plan v2 plus its textual generator, `validate-plan` and `run`, fake-tensor tests, and bounded redacted errors. The operator freezes and reviews both real-tokenizer entries before merge; CI validates but does not regenerate them. No real benchmark executes in CI.

### Phase 4C.2c: local MacBook smoke

After 4C.2a and 4C.2b merge, explicitly acquire one candidate at a time, execute all eight frozen scenarios on MPS and the two compatibility scenarios on CPU, validate immutable reports, and record only systems-compatibility conclusions. A candidate that is unsupported or blocked still produces its sealed bounded result. Acquired bytes and reports remain ignored and local.

## Acceptance criteria

1. Registry v2 validates its exact root, the v1 predecessor digest and rename map, shared candidate truth, exact checkpoint/base/tokenizer/training provenance, research-only disposition, model files, sizes, SHA-256 values, roles, source-weight dtype counts, acquisition states, candidate acquisition-contract digest, and attribution limits; all mutations fail closed.
2. Plan v2 binds the exact v1 plan and v2 registry, preserves all eight dimension tuples, and deterministically creates only self-authored PT-BR or English textual fields from the frozen generator and profile ranges; it contains no answer labels or model-generated text.
3. Default install, import, manifest validation, and CI perform no network access and do not import Torch, Transformers, Tokenizers, Safetensors, Platformdirs, Psutil, or hub clients. The optional extra is absent from the default-environment proof.
4. Acquisition requires the exact candidate ID and `--allow-network`, uses no credential or proxy environment, admits only the exact HTTPS host families and derived immutable URLs, enforces TLS, content-length, disk, per-file, and cumulative bounds, verifies every byte, and never clobbers an existing snapshot.
5. Recursive verification rejects symlinks, hardlinks, unexpected or missing nested files and directories, unsafe ownership or permissions, changed directory identity, changed bytes, partial snapshots, and hand-built or stale receipts.
6. Loaders construct only the reviewed Transformers built-in `ModernBertModel` or `DebertaV2ForSequenceClassification` and fast tokenizer from verified local bytes, enforce the two exact exceptional tensor contracts, and reject `auto_map`, pickle, `.bin`, remote code, cache discovery, unknown architecture metadata, and every non-allowlisted state-dict mismatch.
7. CPU and MPS both perform the exact FP16-parameter-to-FP32 execution load path, with Laya temperature retained as F32 and mDeBERTa position IDs validated as I64 auxiliary data. MPS fallback is disabled and reports record the full source dtype map, FP32 execution, locked library versions, logical device, threads, batches, attention implementation, RSS, and available MPS memory fields.
8. Pending conformance permits acquisition and verification but blocks execution. The final digest-bound two-entry source-contract vector deterministically covers PT-BR and English rendering, tokenization, marker or pair indexing, scoring rule, tensor and output shape; reports preserve `source_contract_reviewed` and positive-only attribution rather than claiming upstream numerical parity.
9. The eight MPS scenarios and two CPU compatibility scenarios report only the measured systems allowlist, make capacity/truncation states explicit, seal bounded failures, and never emit raw text, scores, answers, ranking digests, or quality/calibration claims.
10. The wheel and sdist contain no registry-downloaded bytes, weights, reports, third-party source trees, raw inputs, credentials, local hardware identifiers, or cache paths.
11. Full repository validation passes on Python 3.11 through 3.13; optional local-candidate tests use synthetic configs and fake tensors in CI and never contact a model host.
12. The real MacBook smoke is a separately invoked operator action after code review and is not a merge prerequisite.

## Verification commands

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.universal_bakeoff validate
uv run python -m benchmarks.universal_local validate-registry
uv run python -m benchmarks.universal_local validate-plan
uv run python benchmarks/validate_default_environment.py
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
git diff --check
```

The optional implementation lane also runs `uv sync --locked --dev --extra universal-local` in a disposable environment, then repeats the local-candidate unit suite and the two offline validation commands. `acquire`, real-snapshot `verify`, `freeze-conformance`, and `run` are operator commands rather than CI commands because verified checkpoint snapshots are intentionally absent from CI.

## Rollout and rollback

Rollout is a normal repository merge of checkout-only benchmark code followed by explicit local acquisition and smoke commands. Nothing is deployed and no provider is contacted. Rollback is a Git revert; local immutable snapshots and reports remain operator-managed and are never deleted by rollback.

## Risks and mitigations

- **Model-card license mistaken for complete provenance:** record source, checkpoint, base, tokenizer, and training-data provenance separately; uncertainty limits the lane to non-redistributed local research and blocks production or commercial claims.
- **Poorjev attribution despite a different checkpoint:** use the `mdeberta-nli` ID and prohibit product-level attribution.
- **PT-BR claims from multilingual marketing:** label Portuguese as an evaluated scenario, not a guaranteed upstream language, and reserve quality claims for future human evidence.
- **Unsafe hub behavior:** never call `from_pretrained`, `snapshot_download`, or cache discovery from loaders; acquisition alone owns reviewed URLs and network access.
- **Nested snapshot traversal:** create and verify every directory and file relative to trusted descriptors, reject links and extras, and bind completion to the candidate acquisition contract.
- **Tokenizer executable surface:** accept only verified tokenizer JSON plus closed metadata and instantiate the built-in fast tokenizer directly.
- **CPU/MPS numerical-path drift:** freeze FP32 execution and SDPA on both devices, forbid fallback, record the full execution axis, and compare performance only inside identical axes.
- **Synthetic fixture bias:** preserve the parent matrix dimensions while publishing the generator, sentence banks, profile bounds, and plan digest; draw no quality conclusion from the unlabeled text.
- **Mac memory pressure:** load one candidate per process, preflight available memory and disk, bound batches, and fail without automatic device fallback.
- **Silent option truncation:** report every accepted state or choice truncation and return unsupported capacity when the declared minimum cannot fit the frozen policy.
- **Thermal and first-run distortion:** separate load, warm-up, and measured samples; synchronize MPS and require the frozen iteration counts before comparison.
- **Negative architecture claims without upstream parity:** keep attribution positive-only until a separately reviewed isolated parity run exists.
