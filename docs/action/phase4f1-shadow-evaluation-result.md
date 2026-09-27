---
title: Phase 4F.1 synthetic shadow operational smoke result
kind: report
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/phase4f1-shadow-evaluation-result.md
globalRef: qmd://saracura/docs/action/phase4f1-shadow-evaluation-result.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-27
sourceRefs:
  - "github:felhen-ai/saracura@b6a50c51a38db49d09a310f115dbb31ecc5b675b"
related:
  - docs/action/specs/phase4f1-shadow-evaluation-and-email-reference.md
  - README.md
  - docs/README.pt-BR.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 4F.1 synthetic shadow operational smoke

## Scope and interpretation

This was a local CPU smoke of the synthetic reference workflow. It verifies
that the sealed runtime can execute a bounded batch and produce a descriptive
evaluation summary. It is not a model-quality evaluation, a calibration result,
or evidence of production readiness. No per-item content or labels are
published here.

## Bound execution record

- Source commit: `b6a50c51a38db49d09a310f115dbb31ecc5b675b`.
- Public model revision: `phase4e-saracura-ranker.v1.f1d72c34cc535ddefbf7e24cf45d1be6e0aee40ba881228b4edd092d78640d5e`.
- Policy SHA-256: `fefe634cb4da9aa2b8afa91de84e1d284a4996f26c62e09c562c8cefc2131110`.
- Decision output SHA-256: `dded5e4bea3c97a4dc3d7e86945a78b81df1bb4675ef3a0a3e9e158a1575c678`.
- Evaluation summary SHA-256: `a7497280d6faaaa6209c80bf61049992c607aa0bd0cb5c2a1164088d164be50b`.
- Aggregate: 8 predictions and 8 feedback records; 7 labeled, 1 skipped, and 0 unreviewed.
- Agreement: 1 of 7 labeled records (14.2857%).
- Runtime: CPU; one runner/backend lifecycle; 0 errors; canary text absent from captured stdout and stderr.
- Empty-state policy context: 73 tokens, leaving 55 tokens under the 128-token context limit. This is a measurement with empty `subject` and `preview`, not a content budget or truncation allowance.

## Safety conclusion

The observed 14.2857% agreement is low and must not be interpreted as
readiness, confidence, or authorization for automation. Keep any Phase 4F.2
adapter strictly read-only, with human review of each proposed classification.
Policy and model iteration must remain separate, explicitly versioned work;
feedback from this smoke is not training, retrieval, prompt-optimization, or
calibration data. Any future quality claim requires a separately designed
evaluation on appropriate held-out data.
