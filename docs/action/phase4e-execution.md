---
title: Phase 4E execution record
kind: report
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/phase4e-execution.md
globalRef: qmd://saracura/docs/action/phase4e-execution.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
  - benchmarks/manifests/training-data-source-policies.v1.json
  - benchmarks/manifests/training-data-source-policies.v2.json
  - benchmarks/manifests/phase4e-spend-baseline.v1.json
related:
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
  - benchmarks/manifests/phase4e-spend-baseline.v1.json
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E execution record

## Decision

Phase 4E.2c implementation is `GO`. The execution-recovery pilot resolved all 140 tasks without an operational failure, but its quality decision is `FAIL`: a systematically ambiguous fictionality instruction caused 127 rows to fail the same reviewer gate. Corpus generation and training remain `NO-GO` until that protocol defect is corrected and a complete pilot reaches `PASS`.

## Cumulative spend through 2026-09-24

The local research history contains 46 `corpus-work*` directories and 2,117 charged ledger entries: 2,099 settled and 18 open reservations. Provider-reported cost is USD 0.81263107. Conservative debit is USD 5.12153043, which already includes USD 0.01971612 from the 18 open reservations. These figures are fixed in `benchmarks/manifests/phase4e-spend-baseline.v1.json`.

The former USD 17.00 cumulative authorization and per-run caps are historical v1 controls only. The operator removed the financial ceiling for subsequent research execution. V2 therefore records actual provider cost and conservative debit without using either value to authorize, block or stop work. The operator is informed whenever run-local provider-reported spend crosses another USD 10 and at run completion.

The v1 pilot reported USD 0.00312645 and conservatively debited USD 0.03176494. The v2 pilot reported USD 0.10454764 and conservatively debited USD 0.96313398. Post-v2 cumulative totals are USD 0.92030516 provider-reported and USD 6.11642935 conservative. No corpus or training spend was incurred.

## Registry digests

| Registry | SHA-256 |
| --- | --- |
| `benchmarks/manifests/training-data-source-policies.v1.json` | `633245cde07b9f944647e615e17855d392048074b1dbb4a021f5f29718a5e43c` |
| `benchmarks/manifests/training-data-source-policies.v2.json` | `ea32197f712ed972556126482162ad02900200350a2081a98280a87a68861e78` |

Human and Phase 4B consumers stay bound to the v1 registry. The protocol pilot is the only consumer of v2. v2 keeps the same two exception IDs and adds the closed `protocol_pilot` object to the Phase 4E exception. Pilot rows are evaluation-only and cannot enter a training packet.

## V1 pilot outcome

The protocol pilot used seed `saracura-phase4e-protocol-pilot-v1` and the immutable 140-task plan covering 70 PT-BR/English pairs and option counts 2 through 8. Its complete-plan preflight was USD 0.33295360 for the author and USD 0.86601256 for the reviewers, for USD 1.19896616 total within the stage and pilot caps.

The run stopped fail-closed with decision `INCONCLUSIVE` after resolving three pairs:

- 0 accepted, 6 rejected, and 134 unresolved tasks;
- 0 complete accepted pairs, with both Wilson lower bounds at 0;
- 4 completed blind reviews, with 2 scenario disagreements and 2 criterion-role disagreements recorded only as semantic diagnostics;
- 0 local privacy violations and 0 reviewer privacy flags;
- 1 operational failure after the third author request and one of that pair's reviewer requests had settled.

The first two pairs completed normally and were rejected by the acceptance gates. The third pair was stored as a settled fallback after an exception interrupted review. The fallback reason is the generic `author_response_failure`; the settled journal sequence shows that the author request and one reviewer request had already completed, so the artifact does not support attributing the interruption specifically to the author. This is an observability limitation, not evidence that the remaining 134 tasks failed a quality gate.

The work tree was copied create-only to the durable research root at `~/Library/Application Support/saracura/phase4e/research-ledgers/pilot-work-v1`. The sealed local evidence has these bindings:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan | `1fff45552e6a6f0b262a473492b30c857e7d0732d2f95a3ead9efecff6de3a63` |
| post-copy research inventory | `e9d4cc7f5afd61fd150385d1217f8a61aff225a6bf819c68ff7cd8d9237b0c47` |
| report-declared ledger | `08eeac18ef7856e217e4701c844984bb5c9f62378185317fa4ccf3a66f78381b` |
| sealed report file | `7b97255e1a2a47447303ee91c7b1fbb0a837f56d7ccac5b60be9e297ed4aebc1` |
| sealed ledger file | `242361ec1fe5745fe154862ac701a2fcd8c69103798c92e0f9914819b9d7f74f` |

The v1 evidence remains immutable. It was superseded operationally by the v2 execution-recovery protocol below.

## V2 execution-recovery outcome

The v2 recovery kept the same answer-blind seed and 140 task identities, added report-only cost telemetry, stage-specific bounded response recovery, settled-call replay protection and continuation across locally rejected responses. It completed all 70 PT-BR/English pairs:

- 0 accepted, 140 rejected and 0 unresolved tasks;
- 0 complete accepted pairs, with both Wilson lower bounds at 0;
- 133 completed blind reviews, with 86 scenario disagreements and 32 criterion-role disagreements recorded as non-blocking semantic diagnostics;
- 0 author failures, 7 exhausted reviewer responses, 0 orphaned settled calls and 0 operational failures;
- 0 local privacy violations and 0 reviewer privacy flags;
- USD 0.10454764 provider-reported cost, below the first USD 10 reporting milestone.

The rejection distribution is decisive: 127 rows failed `review_quality_fictional`, 7 failed after exhausted reviewer-response validation, 4 were explicitly rejected and 2 disagreed with the selected criterion. The reviewer instruction currently says that declared synthetic intent is not evidence while requiring `fictional=true`. For generic invented entities with no external provenance available to the blind reviewer, that makes the positive claim effectively unprovable. This is a protocol wording defect; it does not support a conclusion that the generated rows contain real people, organizations or private data.

The work tree was copied create-only to the durable research root at `~/Library/Application Support/saracura/phase4e/research-ledgers/pilot-work-v2`. The sealed evidence has these bindings:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan | `ff056cf6f128d73865a6e66711b99c98fa655f07a42e1703561154bf555c2359` |
| post-copy research inventory | `4c2d2a89dc2bb701d6d77894e2950748c7bdad4651e5aeed1cb3c07c3efa44ec` |
| report-declared ledger | `3263a957747eec369c89e7adcd50a45f58c2fec6a573d4c3c1d282c80538be10` |
| sealed report file | `bef202e43d769aac54b874aa68d672e3a2dd9aed0b0e238cfd20b6a2e30baf30` |
| sealed ledger file | `19e2ff03cbcf583da5a9a4d2d789dbf1cdbfb49c307a2aa1f7233bd9eef06392` |

Phase 4E.3 remains blocked on quality evidence, not financial authorization. The next increment must replace the ambiguous fictionality test with an observable-content test for generic/invented entities versus identifiable real-world or sensitive identifiers, preserve all v2 blindness and privacy invariants, and rerun the protocol in new create-only v3 evidence paths.

## V3 diagnostic outcome

V3 replaced the epistemic fictionality wording with an observable-content rule and cryptographically bound the exact reviewer instruction into the plan. The run was intentionally stopped after 12 of 70 pairs because the first 24 resolved rows reproduced the same systematic pattern: 22 `review_quality_fictional` rejections and 2 exhausted reviewer responses, with no accepted row.

The remaining ambiguity is now narrower. The answer-blind payload necessarily includes `task_id` and criterion IDs for routing and answer selection. V3 told the reviewer to reject visible identifiers but did not explicitly exclude those technical metadata fields from the content-safety judgment. The next protocol must state that fictionality and privacy quality flags are computed only from `instruction`, `state.summary` and criterion descriptions; `task_id`, criterion IDs, locale and domain are trusted routing metadata, not evaluated content.

The interrupted run has 46 settled ledger entries and one open reservation because interruption occurred while reading a provider response. Settled provider-reported cost was USD 0.02153594 and settled conservative debit was USD 0.19660007; the open reservation remains conservatively accounted for and must never be replayed. Its work tree was copied create-only to `~/Library/Application Support/saracura/phase4e/research-ledgers/pilot-work-v3`, and the post-copy inventory SHA-256 is `0fef1659be91b467e5fe59fbf04cb24e35e5ee3f8de54377b88c8e75374a7d5c`.

V3 is `INCONCLUSIVE` by interruption and is not resumable. Phase 4E.3 remains blocked until a new create-only protocol resolves all 140 tasks and reaches `PASS`.

## V4 diagnostic outcome

V4 excluded routing metadata from the quality judgment and defined observable real-world anchors. It still produced 10 `review_quality_fictional` rejections across the first five pairs, then stopped `INCONCLUSIVE` after a transport-uncertain author call. The other 130 tasks were not attempted. The run reported USD 0.00709627 in settled provider cost and USD 0.06954541 in conservative debit; cumulative totals became USD 0.94893737 provider-reported and USD 6.38257483 conservative.

This result eliminates both reviewed hypotheses about instruction wording and routing IDs. The remaining failure is the response field itself: a boolean named `fictional` asks the model for an epistemic conclusion even when the intended decision is an observable genericity/safety check. The next protocol must expose `generic_or_invented` to the provider and map that field locally into the existing internal `fictional` compatibility field. The provider schema and exact prompt remain plan-bound, while downstream acceptance remains unchanged.

The work tree was copied create-only to the durable research root at `~/Library/Application Support/saracura/phase4e/research-ledgers/pilot-work-v4`. The sealed evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan | `00f2f88f8722d09142fdae946a37a3b4951dad265739a6815b5b34b351c530b5` |
| post-copy research inventory | `b613e6f0afcdf2f7763c53ff18277c43104875251a878ac573082ef22d969c9e` |
| report-declared ledger | `d0dbca4c51cd8ebced5f104049c7ce72887812c2388eb09702ed968ea50834ca` |
| sealed report file | `b391e00f241aa78354c138eb6a7d570f188a1e1b2be9ce3282810a980d814f0a` |
| sealed ledger file | `0d2314472ee30c5627f07eea51bc41812e57a4f5d55385e1c874b33f716894f8` |

V4 is not resumable because its ledger contains one transport-uncertain call. Phase 4E.3 remains blocked pending a complete `PASS`.

## V5 diagnostic outcome

V5 renamed the provider field to `generic_or_invented` and mapped it locally to the stable internal `fictional` field. It was stopped after five resolved pairs because the provider-facing JSON Schema still inherited `title: "Fictional"` from the internal Pydantic field. That hidden semantic label meant the provider continued receiving the concept V5 intended to remove.

The 10 resolved rows included 1 accepted row, 4 `review_quality_fictional`, 4 `review_quality_exclusive_options` and 1 `semantic_duplicate`. The work ledger contains 15 settled calls and one open reservation from interruption during an author response. Settled provider-reported cost was USD 0.00729730 and settled conservative debit was USD 0.07053515; no USD 10 reporting milestone was crossed. The open reservation must never be replayed.

The work tree was copied create-only to `~/Library/Application Support/saracura/phase4e/research-ledgers/pilot-work-v5`; the post-copy inventory SHA-256 is `36fb9a6198a685ffa2c2f2b5c7267f630d75a1b88d72816b4644887a75b1c877`.

V5 is `INCONCLUSIVE` by interruption and is not resumable. The next protocol must replace the inherited schema title and add an explicit schema description for the observable genericity field before another create-only run.

## V6 diagnostic outcome

V6 removed every provider-facing `fictional` key, title and reason code, replacing them with the observable genericity contract. This fixed the systematic zero-acceptance behavior, but the run was stopped after 36 of 70 pairs because `PASS` had become mathematically impossible: 12 of 72 rows were accepted, so even accepting all 68 remaining rows would reach only 80, below the minimum of 98.

The resolved distribution was 24 genericity rejections, 16 exclusivity rejections, 11 local semantic duplicates, 6 explicit review rejections, 3 exhausted reviewer responses and 12 accepted rows. This is no longer a wire-schema defect; it is evidence that the Qwen3 30B-A3B author plus Llama 3.3 70B reviewer pair cannot meet the current protocol threshold reliably.

The ledger has 120 settled calls and one open reservation from interruption during a provider response. Settled provider-reported cost was USD 0.06145967 and settled conservative debit was USD 0.54864810; no USD 10 reporting milestone was crossed. The work tree was copied create-only to `~/Library/Application Support/saracura/phase4e/research-ledgers/pilot-work-v6`, and the post-copy inventory SHA-256 is `59afe660f0ad9b6ff6fe3e01585bed73c631f1f95c5097233eb4991ecc6ec76e`.

V6 is `INCONCLUSIVE` by deliberate early termination and is not resumable. The next protocol should change the model pair while preserving the now-validated genericity wire contract and all acceptance thresholds.

## V7 model-pair outcome

V7 changed the author to GPT-4.1 Mini and the independent reviewer to Gemini 3.8 Flash, retained ZDR/no-collection routing and versioned the report-only ledger prices. The first author call settled successfully, but the first reviewer exceeded the transport's fixed 30-second socket timeout. Its outcome is unknown, so the call was marked `uncertain` and the run correctly stopped `INCONCLUSIVE` with all 140 tasks unresolved.

Settled provider-reported cost was USD 0.000926 and conservative debit was USD 0.00923025; no USD 10 reporting milestone was crossed. Cumulative totals became USD 1.01862034 provider-reported and USD 7.01098833 conservative. The v7 ledger contains one settled call and one uncertain call and must never be resumed.

The create-only evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan | `40642b0226342204449fef7317f71141519f81665c0ccdefdb0c314af44e1b4a` |
| post-copy research inventory | `71e8c87275d150240e87985a7985492291945363b16798565119bf3cd1b2990e` |
| report-declared ledger | `cda6dad416ec3f02728bb41225268e5a5ea9afa45eff96e322d5a6fecbf6652b` |
| sealed report file | `a48c61c831c059596e2447a29e14e42fae7129b43b2c504ffa289c08660c7f9e` |
| sealed ledger file | `fd57c76de7821bd3efb9348bab4d4d88744e702e3c20bc44591e8973cd83af38` |

The next create-only run keeps the v7 model pair and raises only the fixed transport timeout from 30 to 120 seconds. This is an operational correction, not a quality-gate relaxation.

## V8 provider-compatibility outcome

V8 kept the v7 model pair and raised only the live pilot transport timeout to 120 seconds. The first author call settled, but the first reviewer returned a provider-routing error before inference and was conservatively recorded as `uncertain`; all 140 tasks therefore remained unresolved and the run stopped `INCONCLUSIVE`.

The post-run diagnostic isolated the request incompatibility. Under the required ZDR/no-collection policy, OpenRouter retained a Google Vertex endpoint for Gemini 3.8 Flash. That endpoint does not advertise `temperature`, while the shared request builder unconditionally sent `temperature: 0`; with `require_parameters: true`, OpenRouter rejected the reviewer request during parameter filtering. A separate exact-schema probe proved that the same endpoint succeeds when reviewer `temperature` is omitted and reasoning is explicitly set to minimal. Privacy routing and strict parameter enforcement do not need to be relaxed.

The v8 run reported USD 0.0006108 in provider cost and USD 0.0092205 in conservative debit; no USD 10 reporting milestone was crossed. Cumulative totals became USD 1.01923114 provider-reported and USD 7.02020883 conservative. Its one settled author call and one uncertain reviewer call make v8 non-resumable.

The create-only evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan | `906f3340ea361e7d0905972daa368ce49dcfb4b3b65f9063b0e296cb495e0d87` |
| post-copy research inventory | `b200a0cff597c25465614ffab2cc5ccb92ae69199da8800d20e51bd3e7de4cec` |
| report-declared ledger | `1552b3dc6dc15bc1211c44fa416cf80b3832ffab955e4aa7baa0f802acc78ade` |
| sealed report file | `ccfc1e23f604cd13251508eb58c25e6fd721ea6526aa33155683ac7a09bc3fba` |
| sealed ledger file | `e26680d87e9bbe7a3dd24d9fc0f05a76283d73d65607deb88d67746ac0d16602` |

The next create-only run removes `temperature` only from the Gemini reviewer request, binds its exact reasoning/output policy into the immutable plan and preserves the author request, quality gates and privacy policy.
