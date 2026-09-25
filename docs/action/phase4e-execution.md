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
lastReviewedAt: 2026-09-25
sourceRefs:
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
  - docs/action/specs/phase4e3a-v4-ptbr-transport-recovery.md
  - benchmarks/manifests/training-data-source-policies.v1.json
  - benchmarks/manifests/training-data-source-policies.v2.json
  - benchmarks/manifests/phase4e-spend-baseline.v1.json
related:
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
  - docs/action/specs/phase4e3a-v4-ptbr-transport-recovery.md
  - benchmarks/manifests/phase4e-spend-baseline.v1.json
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E execution record

## Decision

Phase 4E.2 is complete with a `PASS`. V12 resolved all 140 precommitted tasks, accepted 134, met every global, locale and option-count gate, and recorded no operational or privacy failure. This result authorizes the Phase 4E.3 training/checkpoint increment; it does not by itself register the planned runtime backend or authorize automation.

The first Phase 4E.3A corpus execution completed all 1,600 planned identities but ended `NO-GO` before packet sealing because its 848 accepted PT-BR rows represented only 58.2% of the accepted corpus. The separately reviewed v4 recovery then resolved 100 new disjoint PT-BR identities, accepted 95, rejected 5 and sealed a combined packet with 1,552 accepted rows, including 943 PT-BR rows (60.7603%). No identity attached to an uncertain or failed call was replayed. This closes Phase 4E.3A and authorizes only a separately reviewed Phase 4E.3B checkpoint increment; training, calibration, holdout evaluation and runtime registration have not started.

## Cumulative spend through 2026-09-25

The local research history contains 46 `corpus-work*` directories and 2,117 charged ledger entries: 2,099 settled and 18 open reservations. Provider-reported cost is USD 0.81263107. Conservative debit is USD 5.12153043, which already includes USD 0.01971612 from the 18 open reservations. These figures are fixed in `benchmarks/manifests/phase4e-spend-baseline.v1.json`.

The former USD 17.00 cumulative authorization and per-run caps are historical v1 controls only. The operator removed the financial ceiling for subsequent research execution. V2 therefore records actual provider cost and conservative debit without using either value to authorize, block or stop work. The operator is informed whenever run-local provider-reported spend crosses another USD 10 and at run completion.

The v1 pilot reported USD 0.00312645 and conservatively debited USD 0.03176494. The v2 pilot reported USD 0.10454764 and conservatively debited USD 0.96313398. Post-v2 cumulative totals are USD 0.92030516 provider-reported and USD 6.11642935 conservative. Corpus-v3 and its v4 recovery reported USD 6.57710960 and USD 0.4477648 respectively, for USD 7.02487440 in actual corpus-generation cost. No run crossed the USD 10 reporting milestone, and no training spend has been incurred.

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

## V9 Gemini operational outcome

V9 removed the unsupported reviewer `temperature`, requested minimal reasoning and raised reviewer output capacity to 512 tokens. This fixed the provider-parameter rejection and resolved four pairs before another Gemini reviewer call exceeded the 120-second transport timeout. The run stopped `INCONCLUSIVE` with 3 accepted, 5 rejected and 132 unresolved rows; only one pair was fully accepted.

The partial rejection distribution was four reviewer answer disagreements and one local near-duplicate. Nine reviews completed, with four scenario disagreements recorded only as diagnostics. There were no local privacy violations or reviewer privacy flags. Two malformed reviewer responses recovered through the bounded response retry; the final reviewer call remained transport-uncertain and must never be replayed.

The v9 run reported USD 0.02210190 in provider cost and USD 0.10156685 in conservative debit; no USD 10 reporting milestone was crossed. Research-ledger cumulative totals became USD 1.04133304 provider-reported and USD 7.12177568 conservative.

The create-only evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan | `c12b7d09619b9131b856d797ce3d411542cce7a2436750f4cc30da73ac3de742` |
| post-copy research inventory | `3a63900e11fbe1d4463784bfcc850e0f0d64f5b6dceb46a6bd051a3c204248b4` |
| report-declared ledger | `ea9101a3cfc97b14b22d923e24a9f599b6275ee83b781935a9cbc519fbbdb279` |
| sealed report file | `b56c4726c30a6ace335a1d6438206f08f21c924520da03d1dd9e2fce837518b4` |
| sealed ledger file | `3c12ab1b618d0cdf6921e2c67c70606e1667f747ea90dad4dc717be7b7006580` |

Provider-compatibility probes outside the research ledger reported an additional USD 0.00965475. Combined with v9, the diagnostic cycle consumed USD 0.03175665 and did not cross a USD 10 milestone. Three consecutive exact-schema probes of GPT-4.1 as reviewer then completed through Azure with valid structured output in 1.7-2.2 seconds. The next create-only run therefore keeps GPT-4.1 Mini as author and replaces only the reviewer with GPT-4.1; all prompts, privacy settings and quality gates remain unchanged.

## V10 stable-reviewer outcome

V10 kept GPT-4.1 Mini as author, moved the independent reviewer to GPT-4.1, pinned both stages to Azure under ZDR/no-collection routing, disabled provider fallback and restored the installed-runtime boundary for the still-planned Phase 4E workflow. The transport remained stable for 343 settled calls. The 344th call, a reviewer request, became transport-uncertain and stopped the run with 18 tasks unresolved.

The partial result was 65 accepted, 57 rejected and 18 unresolved rows, with 18 complete accepted pairs. PT-BR accepted 39 rows and English accepted 26. There were no local privacy violations or reviewer privacy flags. The rejection distribution was 35 exhausted reviewer responses, 16 local semantic duplicates, 5 reviewer answer disagreements and 1 exclusivity rejection. Response diagnostics recorded no author failure, 195 reviewer-validation failures, 12 retry recoveries and no orphaned settled call.

The reviewer failures were not output truncation. A controlled 512-versus-1024-token probe across option counts 2 through 8 completed with `finish_reason=stop` at both limits. A repeated six-option probe then reproduced the precise issue five times: every response contained six roles and one `matches_rule`, but only two unique roles. The provider JSON Schema cannot express the internal Pydantic distinct-role validator through its supported strict-schema subset; a probe adding `uniqueItems` was rejected before inference. Because semantic disagreement is diagnostic and explicitly non-blocking in this pilot, treating repeated inferred roles as a malformed provider response is a protocol-layer defect rather than a quality-gate failure.

The v10 run reported USD 0.6513888 in provider cost and USD 5.0675728 in conservative debit; no USD 10 reporting milestone was crossed. Research-ledger cumulative totals became USD 1.69272184 provider-reported and USD 12.18934848 conservative. V10-related provider probes outside the research ledger added USD 0.0540308, so the full v10 diagnostic cycle consumed USD 0.7054196.

The create-only evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan file | `0d8a4425c4cda7bd6e34a4bfcb01dca1fefcd7f3600da84694b0dc1223c91bca` |
| post-copy research inventory | `e7ecd54c457d2f438dd3e3e19a3021cd7bcedfd7565d74e4ec2d1708cdd6de18` |
| report-declared ledger | `ac990c4bc1c8efcde532d39df7f8cff855b5e09c2d26910dcfc48d5a641151d8` |
| sealed report file | `4a2f8f2c2296932aa4ff84fa4691d5b16c9dae58928f4746b468dab83297ad5c` |
| sealed ledger file | `2ad191c02396c4c6e6d82adcf6cdc87698c858b4831620b4376cf837eb5e60ba` |

V10 is immutable and non-resumable because its ledger contains one uncertain call. The next create-only run must preserve every quality threshold and answer-blind field, treat repeated reviewer semantic roles as a valid diagnostic disagreement instead of a transport failure, and extend only the fixed transport timeout enough to finish the 140-task protocol.

## V11 non-blocking-diagnostics outcome

V11 made repeated reviewer roles a diagnostic criterion-role disagreement while keeping the strict corpus model unchanged. That correction worked completely: all 134 attempted reviews were structurally valid, with zero reviewer failure or retry. The run reached 134 resolved tasks before the next author request exceeded 240 seconds and became uncertain, leaving its pair and the two subsequent pairs unresolved.

The partial result was 96 accepted, 38 rejected and 6 unresolved rows, with 37 complete accepted pairs. PT-BR accepted 57 rows and English accepted 39. The English result could reach at most 42 even if all three unresolved English rows were accepted, below the protocol minimum of 45; a transport-only rerun with the same deterministic model pair would therefore not fix quality. Rejections were concentrated in 16 English semantic duplicates and 11 English answer disagreements; PT-BR had 8 answer disagreements and 2 explicit review rejections. There were no privacy flags, reviewer-schema failures or orphaned settled calls.

The v11 run reported USD 0.4641744 in provider cost and USD 2.5762804 in conservative debit; no USD 10 milestone was crossed. Research-ledger cumulative totals became USD 2.15689624 provider-reported and USD 14.76562888 conservative. V11-related model and endpoint probes outside the ledger added USD 0.1877528, so the full v11 diagnostic cycle consumed USD 0.6519272.

The create-only evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan file | `f2ce28182acadf97e46087295d0f0f8e7a3087ca9ffa15d5c6c4d68fa8304695` |
| post-copy research inventory | `c19ad65bcf0054955deddc25de00628d2a8562f8cadefe66f3a1c9eadc4af1c8` |
| report-declared ledger | `9d4ef6ed6d164ef5a01715d27b0bf96da0caf05845ee96abcc382fb0fe7f5d10` |
| sealed report file | `a9f3e3c4900705264d7c0f18a5d224002199121f5d60c5dba8e59b393ce0e453` |
| sealed ledger file | `1ad79bd3f1ee6446164169255269c926014a369937e58d22706c7d63a0bdfa5a` |

Two bounded quality probes selected the next model pair. GPT-4.1 authored 20/20 locally valid sampled tasks with zero exact duplicate and one near-duplicate, costing USD 0.06416. GPT-4.1 Mini then independently reviewed 20 previously accepted rows with 20/20 acceptance, no transport or schema failure, median latency 1.984 seconds and USD 0.0115272 cost. V12 will use GPT-4.1 as author and GPT-4.1 Mini as reviewer, preserve Azure/ZDR/no-fallback routing and every quality threshold, and conservatively reject only the tasks attached to any transport-uncertain call while continuing the rest of the precommitted plan.

## V12 completion outcome

V12 promoted GPT-4.1 to author, used GPT-4.1 Mini as the independent answer-blind reviewer and applied the reviewed conservative transport policy. The complete 140-task plan finished without any uncertain call, retry recovery, schema failure, orphaned settlement or operational failure.

The final result was `PASS`:

- 134 accepted, 6 rejected and 0 unresolved tasks;
- 64 accepted English rows and all 70 PT-BR rows accepted;
- 64 complete accepted PT-BR/English pairs out of 70;
- accepted option-count cells ranged from 18 to 20 across cardinalities 2 through 8;
- global Wilson lower bound `0.9192728634` and complete-pair Wilson lower bound `0.8427095599`;
- 0 local privacy violations and 0 reviewer privacy flags.

The reviewer recorded 94 scenario disagreements and 41 criterion-role disagreements as non-blocking semantic diagnostics. Those values describe disagreement over the protocol's abstract metadata labels, not answer disagreement or an acceptance bypass: all six rejected rows remained in the 140-task denominator and every answer/content gate remained unchanged.

V12 reported USD 0.5680124 in provider cost and USD 2.0644700 in conservative debit. Research-ledger cumulative totals became USD 2.72490864 provider-reported and USD 16.83009888 conservative. No run-local USD 10 reporting milestone was crossed. The durable spend index contains 3,340 ledger entries across 58 research directories: 3,313 settled, 6 uncertain and 21 historical open reservations, with zero overspent entry.

The create-only evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| pilot plan file | `e7069aad474ce32379499bcaf331f9d25383db42b32c97cdf2e3e13dfcd11d78` |
| report-declared plan | `3356ec071525fae6585e7e68303b321944d3d9cc68dd2a726858a7b90a3a8618` |
| post-copy research inventory | `f30907ec0170059d41db50bc9c7ec5b7817518f15ff4b7af589eb9ab5a993271` |
| report-declared ledger | `5573cb185a0bb67a068df2960cc5b8049dba76c70e6c20468c8cbd6eea4eeffe` |
| sealed report file | `a5414b88d38ccd502402843932c97a641433ef434d5d93c5120ccaf936d9d2a9` |
| sealed ledger file | `db210edc536a3bfdbb891d09747f2bad48ad51ef169cc2e3f75f4905e0b671d9` |

The durable work evidence is sealed create-only at `~/Library/Application Support/saracura/phase4e/research-ledgers/pilot-work-v12`. Phase 4E.3 may now build the owned training packet and checkpoint under a separately reviewed spec. Runtime registration remains fail-closed until that checkpoint passes its holdout, calibration, latency and artifact-integrity gates.

## Phase 4E.3A corpus-v3 outcome

The reviewed corpus-v3 plan resolved all 1,600 precommitted identities and accepted 1,457. The accepted rows otherwise met the packet topology: 996 train, 227 dev and 234 sealed holdout rows; all 42 split/locale/cardinality cells passed; all 12 domains were represented globally and in every split; the split family counts were 801 train, 185 dev and 189 holdout; and 282 complete PT-BR/English pairs exceeded the minimum of 120.

The packet still failed its fixed locale gate. It accepted 848 PT-BR and 609 English rows, leaving PT-BR at 58.2% instead of the required 60%. The rejection distribution was:

- PT-BR: 80 `author_transport_uncertain` and 32 `review_disagreement`;
- English: 2 `author_transport_uncertain`, 16 `review_disagreement`, 12 `semantic_duplicate` and 1 `near_duplicate`.

This concentration supports an operational transport diagnosis, not a general corpus-quality failure. The ledger contains 2,737 settled calls and 81 uncertain calls, all in the author stage. One uncertain call resulted from the interrupted foreground process and was recovered without replay through the deterministic reservation match. Two later provider-transport outage bursts produced the other 80 uncertain author calls. Every linked identity was conservatively rejected; no uncertain call was retried or reassigned.

The run reported USD 6.57710960 in provider cost and did not cross the USD 10 reporting milestone. The immutable plan SHA-256 is `13ca8304cbf18c8bd5871b135b5c609740c952805d656003959155f13d7a3adc`. The final ledger snapshot is `ledger-8454.json`, SHA-256 `a257032653e8078ab5d03c62f68a77c943d39912400dd2384fb4c09d1b4429d2`. The implementation revision used for the run is `edc5af4`.

Packet sealing failed closed on `locale_minimum`. Therefore no accepted packet and no corpus report were created at their planned paths, and holdout content remained unopened. Training, calibration, checkpoint creation and runtime registration were not attempted.

The 11,055-file work tree occupies 8.9 GB and remains intact at `.artifacts/phase4e/corpus-work-v47` on the external development SSD. A create-only copy to the previous internal research root was byte/count verified, but filled the internal volume and was removed after verification; the source evidence was preserved. The existing catalog validator also loaded every cumulative ledger snapshot simultaneously and consumed approximately 15 GB of memory. Its chain validation now streams one transition at a time, but corpus-v3 is intentionally not cataloged until the recovery increment establishes a durable external or configurable private-artifact root and a compact retention policy.

The recovery increment must keep corpus-v3 immutable, never replay the 81 uncertain calls, generate only new disjoint PT-BR identities, stop cleanly behind a consecutive-uncertainty circuit breaker, preserve balanced ordering, emit a failure report even when packet sealing fails, and combine only accepted v3 plus accepted supplemental rows into a new create-only packet. The estimated deficit is 66 net accepted PT-BR rows; the recovery plan should overprovision approximately 100 new PT-BR tasks while preserving all existing split, cardinality, domain, privacy and blindness gates.

## Phase 4E.3A v4 PT-BR recovery outcome

The reviewed v4 recovery migrated the immutable corpus-v3 evidence into a configured private external state root, retained a compact research capsule and generated exactly 100 new disjoint PT-BR identities. The live execution accepted 95 supplemental rows and rejected 5, producing the combined sealed result:

- 1,552 accepted, 148 rejected and 0 unresolved rows;
- 943 accepted PT-BR and 609 accepted English rows, placing PT-BR at 60.7603%;
- 95 supplemental acceptances and 5 supplemental rejections;
- USD 0.4477648 supplemental provider-reported cost and USD 2.1465188 supplemental conservative debit;
- 0 transport-uncertain supplemental calls and no open recovery error.

The first paid author response settled before a missing local ML dependency caused local validation to fail. Recovery recorded that identity as `author_validation_failure`, did not replay it and continued only after the implementation added a complete tokenizer preflight and reusable verified receipt. The remaining four supplemental rejections were ordinary content-quality or deduplication outcomes. Existing corpus-v3 identities, including all 81 identities linked to uncertain calls, remained immutable and were never retried.

The packet passed the sealing gates without opening holdout content. Its public-safe evidence bindings are:

| Evidence | SHA-256 |
| --- | --- |
| recovery plan file | `f78cff36f9d9818022feefc4a8c508f1267223bdd2cd641c65fafa9a3500412b` |
| report-bound recovery plan | `a302b51d9eb9095de77b66f0da9ee0422e290687bd99e9b6abf1a5d2f3b551f1` |
| supplemental resolution | `32a78f39a7f746f7b5d2f9fc2315ef2d9ef216a607c2337856017c3fadd1ab75` |
| supplemental ledger | `b29b53f21ed4e3add28301bc16d15f1c1eb045cfa2c0df6c5019340a6b01a076` |
| final research inventory | `5cd3c66ec9163867e7b5d4a715182496d3df6bde05348906b1c6d13229091a29` |
| sealed report file | `4f20120c5bd3f74fbc014cf85a9aef0cc1b2dcec1dcdd66fe804ae50ff03973c` |
| sealed packet manifest | `3e5dccc8bb551cf4840046b20712a52c9409c9424f20333185f3072e8dd6c239` |

Phase 4E.3A is complete. The sealed packet is eligible as input to a separately reviewed Phase 4E.3B training and checkpoint increment. It does not itself prove checkpoint quality, calibration, latency or runtime fitness, and it does not authorize automation or runtime registration.
