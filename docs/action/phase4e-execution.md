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

Phase 4E.2b implementation is `GO`. The first reviewed protocol pilot is `INCONCLUSIVE`, so corpus generation and training remain `NO-GO`. The pilot seed is burned and this record does not authorize another provider call, a retry, corpus generation, or training.

## Cumulative spend through 2026-09-24

The local research history contains 46 `corpus-work*` directories and 2,117 charged ledger entries: 2,099 settled and 18 open reservations. Provider-reported cost is USD 0.81263107. Conservative debit is USD 5.12153043, which already includes USD 0.01971612 from the 18 open reservations. These figures are fixed in `benchmarks/manifests/phase4e-spend-baseline.v1.json`.

The operator's cumulative Phase 4E authorization is USD 17.00. Adding the pilot's USD 1.50 complete-plan bound to the historical debit gave a pre-run maximum of USD 6.62153043, which remained inside that authorization. Per-directory caps of USD 5.00 / USD 10.00 / USD 0.75 / USD 1.25 do not reset this cumulative total. The pilot's own caps were USD 0.50 author and USD 1.00 reviewer.

The pilot reported USD 0.00312645 and conservatively debited USD 0.03176494. The post-pilot cumulative totals are USD 0.81575752 provider-reported and USD 5.15329537 conservative. No corpus or training spend was incurred.

## Registry digests

| Registry | SHA-256 |
| --- | --- |
| `benchmarks/manifests/training-data-source-policies.v1.json` | `633245cde07b9f944647e615e17855d392048074b1dbb4a021f5f29718a5e43c` |
| `benchmarks/manifests/training-data-source-policies.v2.json` | `ea32197f712ed972556126482162ad02900200350a2081a98280a87a68861e78` |

Human and Phase 4B consumers stay bound to the v1 registry. The protocol pilot is the only consumer of v2. v2 keeps the same two exception IDs and adds the closed `protocol_pilot` object to the Phase 4E exception. Pilot rows are evaluation-only and cannot enter a training packet.

## Pilot outcome

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

There is no automatic retry for this seed. Any successor pilot requires a new reviewed protocol and manifest, a distinct seed, and renewed cumulative authorization. Phase 4E.3 remains blocked until such a pilot reaches a reviewed `PASS`.
