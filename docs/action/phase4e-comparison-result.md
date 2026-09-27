---
title: Phase 4E.4 blind comparison result
kind: report
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/phase4e-comparison-result.md
globalRef: qmd://saracura/docs/action/phase4e-comparison-result.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-26
sourceRefs:
  - benchmarks/results/phase4e-comparison-v1.json
related:
  - docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md
  - README.md
  - docs/README.pt-BR.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 4E.4 blind comparison

Status: `scored`. Primary cell: CPU, 1 thread(s).

Saracura: 187/198 correct (accuracy 0.944444; 95% Wilson [0.903272, 0.968699]).

Laya: 139/198 correct (accuracy 0.70202; 95% Wilson [0.63496, 0.761391]).

Paired outcomes: both correct 137; candidate only 50; control only 2; both not correct 9.

- The author/reviewer model family and prompt family are both shared with training generation; this synthetic comparison is in-distribution for Saracura but not necessarily for Laya.
- The common workload is restricted to Saracura's 128-context and 96-criterion-token envelope, not Laya's larger envelope.
- This is descriptive research evidence, not a superiority, calibration, production, or automation claim.

Canonical JSON SHA-256: `080e9264b3d7fbaf2c95ff51d2599f0522daf27bf94861a144c1463e94ac5596`.
