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

Phase 4E.2b implementation is `GO`. Corpus generation and training remain `NO-GO` until a reviewed protocol pilot passes and a later spec revision replaces the corpus acceptance gate. This record does not authorize a provider call.

## Cumulative spend through 2026-09-24

The local research history contains 46 `corpus-work*` directories and 2,117 charged ledger entries: 2,099 settled and 18 open reservations. Provider-reported cost is USD 0.81263107. Conservative debit is USD 5.12153043, which already includes USD 0.01971612 from the 18 open reservations. These figures are fixed in `benchmarks/manifests/phase4e-spend-baseline.v1.json`.

The operator's cumulative Phase 4E authorization is USD 17.00. Adding the pilot's USD 1.50 complete-plan bound to the historical debit gives USD 6.62153043, which remains inside that authorization. Per-directory caps of USD 5.00 / USD 10.00 / USD 0.75 / USD 1.25 do not reset this cumulative total. The pilot's own caps are USD 0.50 author and USD 1.00 reviewer.

## Registry digests

| Registry | SHA-256 |
| --- | --- |
| `benchmarks/manifests/training-data-source-policies.v1.json` | `633245cde07b9f944647e615e17855d392048074b1dbb4a021f5f29718a5e43c` |
| `benchmarks/manifests/training-data-source-policies.v2.json` | `ea32197f712ed972556126482162ad02900200350a2081a98280a87a68861e78` |

Human and Phase 4B consumers stay bound to the v1 registry. The protocol pilot is the only consumer of v2. v2 keeps the same two exception IDs and adds the closed `protocol_pilot` object to the Phase 4E exception. Pilot rows are evaluation-only and cannot enter a training packet.

## Pilot outcome

No protocol pilot has been executed. A later run must use seed `saracura-phase4e-protocol-pilot-v1`, import the historical ledgers into the durable research root, and record its decision and cumulative spend here before any successor pilot or corpus revision is authorized.
