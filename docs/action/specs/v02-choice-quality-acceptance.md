---
title: Decision-quality acceptance for the distillation corpus
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/v02-choice-quality-acceptance.md
globalRef: qmd://saracura/docs/action/specs/v02-choice-quality-acceptance.md
reviewCadenceDays: 30
lastReviewedAt: 2026-10-01
sourceRefs:
  - github:#55
  - github:#62
related:
  - docs/action/specs/v02-distillation-safe-corpus.md
  - docs/action/specs/v02-corpus-native-json.md
  - docs/decisions/0004-model-first-product-direction.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Decision-quality acceptance

For the approved private checkpoint only, the executable ordering and gates in
[`v02-model-delivery-execution.md`](v02-model-delivery-execution.md) take precedence
over this document's sealed-before-training ordering. Final #44 gates remain.


## Measured problem

The native-JSON training pilot produced 131 parser-valid author outputs in 140 attempts. Its first six blind reviews agreed with the author and planned correct choice in all six cases, with all gates true for those reviewed cases. Exact auxiliary taxonomy agreement was zero: all six differed in distractor-role naming, and one differed in scenario naming. The current feasibility reducer rejected all six and halted. This is a finite training-pipeline diagnostic, not a Saracura checkpoint accuracy claim. The old protocol outcome remains historical `NO_GO`.

The eight distractor-role names are not mutually exclusive semantic classes. An action may contradict a rule and also be unsafe, premature or overbroad; the chosen answer may itself be an unsafe action when the question asks which action to avoid. Exact taxonomy equality is not a reliable independent choice-quality test. This amendment separates auxiliary metadata from the supervised choice label.

## C6a public source

Introduce an explicit versioned training acceptance policy `choice-agreement-and-all-gates.v2`. A training row is accepted only if author answer, blind reviewer answer and frozen planned gold position agree, and every existing author/reviewer local and model judgment gate is true. Numeric acceptance floors, domain/locale/split/cardinality allocations, bilingual constraints, privacy/fictionality/length checks, teacher identities, budgets and one-attempt rule do not change.

Scenario and distractor-role metadata remain closed, domain/vocabulary constrained, and correctly sized. Author semantic assignment remains exactly planned and its roles remain unique. Reviewer taxonomy is auxiliary: allow repeated valid role names, since real distractors can share a category; remove only the reviewer's `uniqueItems` prompt constraint and corresponding uniqueness rejection. Unknown fields, domains, role names, wrong cardinality and malformed types still fail. Do not require inferred taxonomy, or an inferred `matches_rule` category at the chosen index, to equal the planned metadata.

The reducer reports a policy ID and aggregate diagnostic counts: reviewed rows, full auxiliary agreement/disagreement, scenario disagreement, role-vector disagreement. Diagnostics do not alter denominators or acceptance. Planned taxonomy coverage must not be advertised as independently verified realized semantic coverage. No forced gold/true values or revealing author metadata to the reviewer.

Frozen author and sealed plans/IDs/seeds remain byte-identical; no new corpus namespace, target rebalance or row replacement. Sealed author/annotation/adjudication schemas and reducer are unchanged. Training author prompt/native schema are unchanged. Reviewer native decoder schema is byte-identical because its `uniqueItems` was already removed in the C5 decoder projection; reviewer response-schema digest and message body change only for the auxiliary uniqueness clause. The explicit policy binding belongs only in `prompt_contract()`, not in the reviewer template. Assert byte-identical training-author response-schema digest against source17b6883. Existing stricter-schema reviewer outputs are a valid subset of the new auxiliary schema. Existing first-shot observations can be carried forward; their original request hashes/provenance are retained, not represented as having used the revised prompt.

Allowed public files: benchmarks/v02_corpus.py; tests/test_v02_corpus.py; the original corpus spec for a small acceptance amendment; this orchestrator-owned spec. No dependencies, manifest/planner/namespace, renderer, serialization, adapter, calibrator, training or packaging changes. Default install remains lightweight. No cloud or actual model calls by executor. Executor is final, not alone; preserve others, no nested Codex.

Tests must prove: correct three-way choice with taxonomy disagreement can be accepted; repeated known reviewer categories are valid diagnostic metadata; wrong choice, false gate, malformed/unknown auxiliary metadata still reject; same floors/denominators/order/one-attempt/zero blind leakage; aggregate diagnostics accurate; training/sealed plans identical to source17b6883; all other role schemas/native schemas unchanged except reviewer prompt uniqueness. Update the previous semantic-disagreement rejection test deliberately, not by deleting coverage. Focused tests/lint/type checks and one default full suite, no redundant full-suite runs.

## C6b separately reviewed private policy carry-forward

No model may resume under the old grant or HALT. After C6a reviewed/merged, root may create a separately reviewed policy view with exact new source/code/grant and an explicit lineage receipt. Preserve the old source, original 481-file private backup, old root and old HALT untouched under root-only historical access.

Copy the complete immutable authored cases, committed events and completed reservations byte-for-byte into the new own-UID root, with the same frozen plan and IDs. Copy every consumed record, including all failures, not selected successful rows. Do not copy the old inode-bound ledger binding: the existing constructor generates the new root's binding with the same plan digest. Record the old binding digest/ancestry and new binding digest/ancestry in the lineage receipt; preserve the old binding in the historical root. The six reviewed observations remain original first-shot commitments; remaining reviewer identities get at most one inference. Prove no unresolved reservation, hash equality/counts, correct own UID/modes and negative cross-lane/history access. Do not copy any private-control content or old approval/HALT/control reports into the new policy control area; original control/HALT remains archived intact. This is an explicit policy fork, not an old-policy reset. No case re-authoring, answer editing, commitment fabrication, dropped rejection, filtered denominator or repeated model call.

Bind the carry-forward receipt (parent event/case/reservation inventory, old source/grant/prompt identity, new policy/source/grant) in the runtime lock. Report that annotation protocol provenance is mixed: the first six used the previous stricter auxiliary-schema prompt; later unused IDs use the revised prompt. This change is developed against training-pilot data only; the sealed evaluation has zero consumed model attempts and remains untouched. No retroactive inference-quality or causal improvement claim.

Existing reviewed caller resume behavior skips all committed roles without tokenization/chat; no replay/import framework or new caller override. Private root rebinding updates exact source hash and fresh observations/grammar receipt literals only, with truthful historical throughput evidence; models/runtime/native backend unchanged. Recompile the 5300 native schemas under current source into a fresh receipt, collect the new lock/grant, and require independent combined diff/deployment/lineage/exact-grant PASS before any further inference. Only then finish unconsumed training reviewer IDs. Both pilots must still pass before full corpus, full sealed before full training. Full-run capability and final sealing remain separate increments.

## Evidence and rollback

The training policy is one module constant, not a user-selectable fallback. Bind it as training_acceptance_policy in prompt_contract(); the global contract/grant digest changes. Author/sealed templates, per-role response schemas and message bodies remain unchanged; the sole reviewer exception is the response-schema uniqueness clause and its embedded message body. All native schemas remain unchanged. Training aggregate receipts use a new TRAINING_RECEIPT_SCHEMA v02-aggregate-receipt.v2 with training_acceptance_policy and auxiliary_semantic_diagnostics fields; keep sealed aggregate receipts exactly on the existing v1 schema. Verification checks training receipt version/policy before metric re-evaluation and rejects v1 or a mismatched policy with an explicit historical-policy/source-required error, not an accusation of tampering. No silent v1 fallback. Historical reducers/validation are reproduced only with archived source17b6883 and their original ancestry context; prior NO_GO remains verified historical evidence after archival.

Diagnostics cover the whole training lane and only successfully committed reviewer envelopes (not failure/local_gate_failure records): reviewed_rows, full_agreement, full_disagreement, scenario_disagreement and role_vector_disagreement. Validate these five keys as a closed shape with nonnegative integers (not booleans), including within receipt verification; full_agreement plus full_disagreement equals reviewed_rows. Include these exact diagnostics in training aggregate receipts and verify them against events; no raw text, labels or paths. A successful reviewer envelope can count in diagnostics even if a primary gate is false, but that row remains rejected. Explicitly test old/new training policy receipt rejection and unchanged sealed receipts.

Before the separate C6b combined review, mechanically enumerate private consumers of reduce_training, create_aggregate_receipt, verify_receipt and aggregate-receipt.v1 and account for their compatibility in the private diff. The authoritative original spec must distinguish terminal old-policy NO_GO from a separately reviewed, versioned policy fork; this does not authorize reset or retry.

Old `NO_GO` is never erased or rebranded as PASS. New policy results are separately named and traceable. If the revised primary-quality pilot cannot meet unchanged floors, stop and retain all evidence. Rollback does not authorize inference retries. Model Preview remains uncalibrated and automation-disabled; no checkpoint quality or superiority claim follows from data acceptance.

## C7 supersession note

C7 keeps `choice-agreement-and-all-gates.v2` and every floor unchanged. Its 28-slot
grounding micro is an earlier terminal gate, not a replacement acceptance policy.
