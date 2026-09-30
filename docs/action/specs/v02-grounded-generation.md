---
title: Grounded training-author generation
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/v02-grounded-generation.md
globalRef: qmd://saracura/docs/action/specs/v02-grounded-generation.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-30
sourceRefs:
  - github:#55
  - github:#63
related:
  - docs/action/specs/v02-distillation-safe-corpus.md
  - docs/action/specs/v02-choice-quality-acceptance.md
  - docs/decisions/0004-model-first-product-direction.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Grounded author generation

## Decision and measured problem

The C6 training pilot ended NO_GO after 49 independent reviews: 34 primary choice agreements and 15 disagreements. Three PT-BR author failures make 18 rejected PT-BR slots, so at most 52 of 70 can pass the unchanged minimum of 53. There were nine invalid author attempts overall; no sealed calls or owned-model training occurred. A bounded diagnostic sample found targets contradicting stated constraints and semantically equivalent alternatives; a reviewer error is also plausible in one sample. Do not generalize those three samples to all failures.

The operator authorized a focused generation-protocol revision after this terminal result, on condition that it changes the failure mechanism instead of repeating the same experiment. Historical results and denominators stay immutable. This is the single C7 mechanism: require a visible governing rule and an author-only option-by-option construction check before committing a case. It is not proof of truth; the unchanged blind reviewer remains necessary.

## C7a public source: bounded executable scope

Allowed files: benchmarks/v02_corpus.py, benchmarks/validate_manifests.py, benchmarks/manifests/v02-distillation-safe-corpus.v1.json, tests/test_v02_corpus.py, tests/test_manifests.py only if needed for the manifest schema, this root-owned spec, and a short supersession/authorization note in v02-distillation-safe-corpus.md, v02-choice-quality-acceptance.md and v02-corpus-native-json.md. No dependencies, adapters, general semantic-verifier framework, model substitutions, renderer/tokenizer/configuration changes, training or calibration. Executor is final, not alone, preserves other edits, no nested workers or actual models/network/cloud calls. Use one focused test run after implementation and one default suite, not redundant suites.

### Visible rule and private construction response

Revise only the training_author role template. Its state must include the explicit governing policy plus the relevant observed facts and boundaries, including what happens at equality when a numerical threshold matters. It must ask a positive decision question: which option satisfies the stated policy. It must not invent missing contracts, authority, approvals, stock, alternative locations or consent. Every distractor must be a distinct action that conflicts with a stated condition; do not offer equivalent descriptions of the same action. Do not label the correct option or expose semantic role names in the case text. Bilingual authors preserve the source facts/policy and translate the same decision, without receiving the source label or construction check.

Add a training-author-only required response field construction with a closed object:
- rule_quote: NFC nonblank text, at most 240 characters, occurring verbatim as a contiguous substring in state;
- option_checks: exactly N closed objects in option order, each with option_id, supported (a real Boolean), and reason (NFC nonblank text, at most 160 characters).
- option IDs must equal option_i at index i. Exactly one supported value must be true, and its option_id must equal the returned answer. Reasons describe why that option satisfies or violates the visible rule. The validator verifies the closed shape, lengths, ordered IDs, quote occurrence, one-support and answer consistency, not semantic truth. A short internally inconsistent construction becomes invalid_output under the existing single-attempt failure mechanism.
- Never constrain supported values, answer or any judgment to a planned gold/true constant in the native decoder. Boolean domains remain free; author planned semantic metadata remains the pre-existing copied constant, not an independently verified assertion.

Retain construction in the immutable training-author envelope so event digests bind it. Envelope base schema_version for this author becomes v02-envelope.v2; every other role stays v1. Extend only the author's closed keys/validator. Revalidate construction structure and answer consistency on resume; rule-to-state occurrence is checked when parsing the author response against its case. If complete event-history validation has the private case available, recheck the binding there too; do not pretend an aggregate-only event verifier has case text. Proof text is untrusted private metadata, never input to the student, renderer, reviewer, bilingual context or sealed lane. case_from_author still returns exactly state/question/options. Keep all gates and response/option validators. All accepted rows still require author == reviewer == planned gold and all local/model gates under choice-agreement-and-all-gates.v2.

Full prompt schema and native decoder must both describe construction, with minLength=1 and maxLength=240 for rule_quote and maxLength=160 for reasons. Lengths count Unicode code points after NFC normalization, not JSON-escaped bytes; test accented text and escaped characters at the bounds. The native projection preserves these bounds, uses exact positional option_check IDs, just as existing author option IDs, and preserves free Boolean support judgments. Bind a versioned author-grounding policy and its schema in prompt_contract. Full state/branch/packed/token ceilings remain unchanged; proof is not rendered into a student input. max_output_tokens remains 2048. If that ceiling causes failures, retain them and stop; do not silently raise budgets. C7b must compile the new bounded author construction schemas for every option count under the actual pinned xgrammar before any inference; no unsupported-keyword fallback. The substring check deliberately is not a semantic guarantee: the prompt asks for a meaningful governing-rule quote, but even a long quote could be irrelevant. Keep the independent reviewer, rather than adding another semantic verifier or changing quote minima in this increment.

Only successful training-author envelopes use v2 and carry construction. Training-author failure envelopes stay v1 with the existing closed error codes and no construction; test this explicitly. All ledger-events are private and content-bearing after this change, even when their other fields are only labels/digests. Diagnostics and public receipts must explicitly select aggregate fields, never dump event objects.

### Fresh training identity, unchanged untouched sealed plan

Use corpus namespace saracura-v02-grounded-author-v1 and protocol_digest c7-grounded-author-r1 in source, manifest and closed manifest validation. New training IDs have zero overlap with C5/C6; retain exact seeds/counts/splits/domains/cardinalities/gold allocations/bilingual/pilot quotas. New opaque IDs may naturally rederive singleton semantic assignments; retain the planned vocabulary/uniqueness invariant and disclose this is not a paired causal comparison. Add a separate sealed namespace constant equal to saracura-v02-native-json-v1 and use it for both sealed plan namespace and sealed opaque ID derivation. Sealed plan bytes, all five non-author role templates/schemas/native projections and teacher identities must remain byte-identical to source86a67f0. The global prompt contract necessarily changes. This explicitly supersedes the old one-attempt-per-issue prohibition for one newly reviewed training protocol only; no old root/ID/HALT is reopened or imported. The sealed lane has zero calls and remains frozen.

### Smaller first evidence, not another generic harness

Add two pure, offline helpers inside the existing corpus module:
1. grounding_micro_pilot(plan): deterministic 28 training pilot slot IDs, two unpaired slots per locale x option-count cell (PT-BR/English x 2..8). Select the first two eligible unpaired slots in the existing frozen pilot order, fail closed unless each cell has two and IDs are unique. Only pilot members; no bilingual dependencies; no new corpus namespace or separate throwaway examples. Those same rows are included exactly once in later 140-slot reduction.
2. reduce_grounding_micro_pilot(plan, events): revalidate the same immutable envelopes and report aggregate-only resolved/accepted/denominator=28, accepted locale counts, accepted option-count counts, and status PENDING/PASS/NO_GO. Require at least 24/28 accepted, at least 12/14 per locale and at least 3/4 per option-count bucket. Failures count in the denominator and remain consumed. Use the same three-way choice/all-gates acceptance as reduce_training, not a second acceptance policy. PASS only after every selected identity is resolved; NO_GO when accepted + unresolved cannot meet any minimum. No raw IDs/labels/proof in the reduced report.
The 140-slot pilot and full corpus acceptance criteria remain unchanged. Micro PASS authorizes finishing the original fresh 140-slot pilot, not full generation/training. Enforce this in the two existing public admission points, _reject_inadmissible and OfflineLedger.commit, not only the private caller: while micro is PENDING, only selected micro identities may be dispatched/committed; micro NO_GO blocks every training identity; non-micro identities require micro PASS. Check before reservation/network and again before commit. Keep the existing 140/full/sealed admission gates; there is no user-disable flag or second admission framework. Test direct public execute/commit paths, pending and failed micro, and rejection before any reservation/POST. A failed micro is terminal for C7; do not tweak prompt, seed, model, threshold or run another batch. Diagnose and report instead.

Freeze the micro dimensions/floors in one new closed manifest block grounding_micro_pilot (training_slots=28, slots_per_locale_cardinality_cell=2, acceptance_floor_total=24, acceptance_floor_per_locale=12, acceptance_floor_per_option_count=3), validated against the protocol constants. Expose no runtime threshold override. With the frozen order, selected micro rows are all train split; declare and test this. Tests that need paired/non-micro/full rows should explicitly establish a valid micro PASS using a reusable self-authored fixture helper, never disable or monkeypatch the admission safeguard globally.

The micro is part of the canonical training reduction, not a separate terminal record. Factor the existing event-derived accepted/resolved sets into a shared private primitive, without recursive reducer calls or a second acceptance policy. Both pure reducers use that primitive. reduce_training adds grounding_micro_pilot with exactly the helper's aggregate metrics; its canonical status is NO_GO if either micro or 140-slot pilot is NO_GO, otherwise the unchanged full-corpus status. Micro PASS alone never yields READY. Existing admission points read this canonical micro/status, with selection membership enforced as above. The receipt for training becomes v02-aggregate-receipt.v3, adding the required closed grounding_micro_pilot_metrics field equal to reduce_training's grounding_micro_pilot. Creation and verification bind it to the same plan/event digests and recompute it from verified events; reject unknown fields, malformed counts/buckets/status, or any altered micro metric/status. Historical training v1/v2 receipts require their historical source/policy, rather than being rewritten or called tampered. Sealed receipts remain byte-identical v1. No additional receipt type, control service or receipt-side source of truth. Test an early impossible micro whose 140-slot pilot is still PENDING: canonical training status and the created/verified receipt must both be NO_GO, while later dispatch/commit are denied. Also test incomplete micro/PENDING, completed micro/PASS with canonical PENDING, and tampering with each added receipt field.

## C7a deterministic acceptance tests

- Valid closed construction accepted; missing/extra fields, wrong types, duplicate/missing/reordered option IDs, out-of-state quote, too-long/control/NFC violations, zero/multiple support or support/answer mismatch rejected. Self-authored fictional fixtures only; do not copy private diagnostic cases.
- Author envelope v2 binds construction; old author-v1 envelopes fail under this source with a historical-protocol-required error. Other role envelopes remain v1.
- Proof does not occur in the label-free case, blind reviewer request, bilingual source, rendered input or sealed role outputs; false gates and wrong answer still reject. Do not force any supported/answer/Boolean values by decoder.
- New training IDs disjoint from old; exact allocation/quota invariants; sealed plan/templates/prompt schemas/native schemas unchanged vs Git86a67f0. Generic template-digest identities must not fabricate hidden gold for blind roles.
- Micro 28 selection exact cell quotas/unpaired/frozen determinism; unchanged 140-slot totals; PASS at24/28 only when all locale/cardinality floors pass, PENDING incomplete, NO_GO impossible; failures never replaced, nonselected rows don't improve micro metrics, no raw output. Selected micro commitments counted once in later training reduction. Canonical reducer and closed training receipt v3 preserve micro terminal/result and reject tampering/historical receipts; sealed receipt v1 unchanged.
- Default install stays lightweight; focused tests/lint/types and one default full suite; CI/review/merge main before cloud deployment.

## C7b private rollout, separate contract before any paid GPU

Keep the pod off until C7a merged and the private deployment has an independently reviewed exact new source/archive/code/grant/plan binding. Preserve C6 and earlier private backups/source/HALT unchanged; initialize fresh training root with no copied cases/events/reservations. Create a fresh own-UID sealed root with the same frozen zero-consumption plan rather than overwriting any existing private-control/grant binding; archive the old unused root. No private proof/raw output in Git or hosted author/reviewer APIs. No VM151/5090. Before deployment mechanically enumerate every private consumer of author-envelope v1, ledger-events, reducer, diagnostics and final corpus export: account for successful author v2, failed author v1, and content-bearing construction. This is a bounded compatibility check, not a new export bridge.

Use existing private caller with one explicit closed execution cohort: micro (exact grounding_micro_pilot IDs) or pilot (existing140 IDs). The author/reviewer predecessor check uses exactly the selected cohort for micro; before pilot reviewers it uses the full frozen predecessor pilot. Resume skips every already committed micro role without tokenization/POST, preserving one invocation per role/identity. Bind micro selection digest and phase to the private source/runtime lock; do not accept arbitrary supplied IDs or ranges. Run author then independent reviewer for the micro. Report status and per-stratum acceptance. Only micro PASS allows remaining fresh pilot roles; both original pilots must PASS before full sealed then full training. A source change or HALT never authorizes replay.

Before inference, use self-authored concise fictional PT-BR/English eight-option fixtures with visible policy to measure unchanged state and joint-input token ceilings using the pinned admission tokenizer/renderer. Measure the complete illustrative author JSON, including construction and semantic metadata, against the pinned teacher output budget; this is a feasibility check, not a guarantee for every generated response. If these fixtures do not fit, stop before a POST rather than consume micro slots or silently raise budgets. Tell the author to use concise state/reasons. Privacy hygiene covers construction quote/reasons as well as case text; private compatibility enumeration includes canonical micro metrics and training receipt v3. Report that the deterministic micro is not balanced by gold position and is not a population accuracy estimate; do not change its selection to balance it.

No new raw-case export bridge, replay/import API, scheduler service or infrastructure wrapper framework. Existing exact archive integrity, least-privilege lane isolation and loopback-only native JSON machinery are retained mechanically. C7b caller/cohort change needs its own frozen spec and combined read-only review; public source phase does not authorize cloud inference.

## Decision rule and rollback

This is a single new measured experiment, not an automatic retry loop. A successful micro demonstrates initial feasibility of construction + independent review, not causal superiority or Saracura model quality. Historical34/49 is descriptive reference only, not a matched control. If C7 micro fails, stop and retain its complete failures/denominators and private backup; no hidden replacement or immediate C8 prompt tweak. If micro passes but 140-slot pilot fails, stop under the unchanged full pilot. Checkpoint training waits for complete publication-capable corpus and sealed capsule.

Report each cumulative USD10 crossing and notify before consumedUSD50; funding is not consumption. Current pod remains EXITED/currentspend0 during this code phase. Rollback is restored reviewed source plus preserved history, never retrying consumed IDs or mixing construction-aware and legacy envelopes.
