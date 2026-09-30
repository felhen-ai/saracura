---
title: v0.2 native JSON generation recovery
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/v02-corpus-native-json.md
globalRef: qmd://saracura/docs/action/specs/v02-corpus-native-json.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-30
sourceRefs:
  - github:#55
related:
  - docs/action/specs/v02-distillation-safe-corpus.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Native JSON generation recovery

## Outcome and measured limitation

The first prompt-only training-author pilot stopped with terminal NO_GO after 40 attempts: 26 parsed author responses and 14 invalid_output failures. None were accepted training rows because independent review had not run. There were no transport/model errors. Rejected provider content was not retained; the failure code does not distinguish syntax, schema or semantic-contract rejection. These are corpus-generation observations, not Saracura model-quality results.

The request supplied response_schema as prompt text but did not send a native response_format. Add native constrained JSON decoding as a bounded hypothesis-driven correction. Do not claim the 14 failures were all malformed JSON or guarantee that this fixes acceptance. The next pilot measures it.

## Frozen scope

Change only benchmarks/v02_corpus.py, benchmarks/validate_manifests.py, benchmarks/manifests/v02-distillation-safe-corpus.v1.json, tests/test_v02_corpus.py, this spec and a small supersession note in docs/action/specs/v02-distillation-safe-corpus.md. No dependencies, runtime imports, serving backend, teachers, licenses, seed, budgets, thresholds, renderer, training, public marketing or #44/#45 changes.

## Request contract

Derive response_format locally from the SAME validated frozen role identity and adjudicator labels used by role_request. Send the documented OpenAI-compatible form type=json_schema, json_schema={name:<stable ASCII role name>, schema:<decoder schema>}. Keep existing messages, prompt schemas, max_tokens=2048, temperature=0, seeds, no-thinking and single POST/no retry unchanged. No caller-supplied arbitrary schema or alternate network route.

A small deterministic native_decoder_schema helper starts from a fresh _role_response_schema object. For vLLM 0.30 xgrammar compatibility, remove ONLY semantic.criterion_roles.uniqueItems from the decoder projection; the full prompt schema and final uniqueness validator remain unchanged. For author options, use prefixItems of exactly N closed objects, with id const option_i at index i, retain descriptions nonblank and exact array length, and disallow extra items. For training_author only, semantic is const equal to the already disclosed planned scenario_code and ordered criterion_roles; this is required metadata copying, not independent judgment. Never constrain author/reviewer/annotator answer/target/label to the planned gold, never constrain any Boolean judgment to true. Blind roles retain full option enums and vocabulary; adjudication enum contains only the two actual committed disagreeing labels. No hidden gold or training semantic assignment in blind schemas.

Native projection version and deterministic per-role/cardinality/domain decoder schema digests enter prompt_contract so schema/decoder policy changes invalidate grants and reservations. A generic template identity for digest enumeration must not fabricate a gold or leak it into blind roles; for the author template metadata-const specialization bind an explicit versioned policy and test it alongside actual frozen identities. Full source SHA remains part of environment grant. Reservation request_digest remains the existing exact-message digest because it is used by the context guard; native decoder changes are bound by the augmented prompt_contract_digest. No digest-cycle or second reservation system.

Do not weaken parse_role_response, validate_case, local_case_gates, _validate_semantic/_validate_inferred_semantic, reducers, feasibility stops or duplicates/privacy gates. Schema-shaped output is not evidence of semantic truth.

## Fresh protocol, not retry

Freeze namespace saracura-v02-native-json-v1 and protocol_digest c5-native-json-r1 consistently in source, closed manifest validator and manifest. Include this namespace in BOTH training and sealed opaque identity derivation (sealed currently uses the constant "sealed" only and must change). Seeds, split/locale/domain/cardinality/gold-position counts, bilingual pairs, pilot membership/order and thresholds remain unchanged. Explicit decision: accept the existing namespace-dependent semantic re-derivation of singleton scenario/criterion-role targets. Do not change _assign_semantic_targets to preserve old metadata: fresh opaque singleton IDs naturally reseed their scenarios/distractors. This is a fresh feasibility experiment, NOT a paired/single-variable decoder comparison with the old pilot; no causal effect-size claim. Pair semantics remain derived from existing pair IDs. Validate the new semantic values against the same vocabulary and required role structure. Prove training and sealed IDs have zero overlap with the prior namespace plan by constructing the old deterministic derivation in tests without raw content. Reject old plan bytes/grants under the new source. Do not migrate, overwrite or reopen consumed old identities or HALT. Historical pilot remains NO_GO and its denominator 140 remains historical, not merged with the fresh experiment.

Later operator rollout must preserve old lane roots immutably and initialize new distinct lane roots and staged grant/runtime metadata before a fresh pilot. No cloud operation or inference belongs to this code phase. Root must verify unchanged models/runtime and offline decoder grammar support before any new pilot; publish a new source/hash/plan/prompt-bound grant and obtain exact independent review before execution. If native schema is incompatible, stop before consuming any new corpus identity.

## Acceptance tests

1. All six roles send native schemas that are closed and deterministic; author positional IDs, copied planned metadata, full judgment Boolean domains and free label choice tested. No hidden targets in blind roles; adjudicator cannot choose third label.
2. Final parser and local gates still reject duplicates, wrong semantic metadata, malformed/extra fields, false judgments and bad privacy/length as before. Decoder projection drops only uniqueItems where unsupported and the final semantic validator still enforces it.
3. prompt_contract binds native decoder policy/digests; changed schema changes digest; existing message reservation digest and one POST semantics preserved. No ML imports/default downloads.
4. Both plans use new namespace, zero overlap old IDs, exact original split/locale/domain/cardinality/gold-position counts, bilingual/pilot membership and seeds; singleton semantic targets may be re-derived as explicitly decided above and remain within frozen vocabulary/role structure. Old plans fail validation and old environment grant mismatches before reservation/network.
5. Focused tests, Ruff, mypy and entire default suite pass. Aggregate documentation states limitation and new experiment without claiming model accuracy, READY or training complete.

## Rollout / rollback

Independent critique, executor, root conformance, independent read-only review, full local checks and GitHub CI, then PR/merge/update clean main. Only afterward rebind private cloud source and new lane roots under a separately frozen private rollout contract. On failure retain this worktree and historical artifacts; no reset or cleanup of private attempts, no fallback to hosted training content.

## Primary runtime references

- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [vLLM 0.30 xgrammar backend](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/structured_output/backend_xgrammar.py)
- [vLLM 0.30 guidance backend](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/structured_output/backend_guidance.py)

## C7 supersession note

C7 changes only the fresh training namespace and author decoder to carry private construction;
the sealed namespace, plan, non-author schemas and native projections stay source86a67f0-identical.
