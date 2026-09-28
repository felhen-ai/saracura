---
title: Open-model runtime and public readiness
kind: decision
area: platform
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/decisions/0002-open-model-runtime-and-readiness.md
globalRef: qmd://saracura/docs/decisions/0002-open-model-runtime-and-readiness.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-27
sourceRefs:
  - https://huggingface.co/jaredpalmer/kev-4b
  - https://huggingface.co/shisa-ai/shisa-de-1
related:
  - README.md
  - docs/README.pt-BR.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md
supersedes: []
supersededBy: []
sensitivity: public
---

# ADR 0002: Open-model runtime and public readiness

## Status

Accepted for Phase 5 research direction on 2026-09-27. This ADR has no runtime effect.

## Context

The installed Saracura-owned `saracura-universal-ranker.v0` is a MiniLM bi-encoder projection ranker. It is historical research evidence and may remain a narrow fast-path hypothesis, but it is not the presumed foundation for new workflows. Its private quality observation is insufficient for automation or a public comparative claim, and Phase 5 makes no parameter-count claim for it.

Saracura needs a model-pluggable route for evaluating open decision backends while preserving the typed API and the compiled specialization described by ADR 0001. Evaluation must remain offline by default, weight-free in ordinary tests, PT-BR-first, bilingual where evidence permits, and separated from acquisition, training, calibration, and runtime registration.

## Decision

Saracura retains one typed decision API with three evidence-bound roles:

- `universal` is a pluggable question-conditioned backend for new schemas. Kev-4B is the first planned acquisition, pending a pinned revision and license review; it is not yet an installed dependency.
- `compiled` remains an optional specialized encoder/head path for stable, repeated high-volume workflows where measured throughput and adequate quality justify it.
- `control` contains remote or heavyweight comparison references. TypeSafe/Jev is an independent external product control, and heavyweight open candidates remain comparison-only until their deployment envelope is evidenced.

The candidate and readiness manifests are the closed public contracts for this direction. They keep acquisition review, evidence authority, allowed claims, and Gate A/Gate B requirements explicit. Both gates are `not_met` in Phase 5A. Neither gate permits production readiness or automation authorization.

## Consequences

Phase 5A adds no backend registration, model download, dataset acquisition, training, calibration, remote fallback, or private evidence. The default install remains local, lightweight, and offline.

Gate A can support a third-party-checkpoint developer preview only after clean-machine installation, pinned license-reviewed acquisition, reproducible public development and local-systems reports, offline CI, public PT-BR/English examples, and a non-automation safety contract. Gate B additionally requires a Saracura-owned checkpoint, sealed PT-BR and bilingual tests, disjoint calibration evaluation, and a clean external submission bundle. Any future comparative statement remains limited to the declared task and its measured evidence.

## Reversal conditions

Reject or replace a candidate if its reviewed license, immutable acquisition, local systems evidence, PT-BR/bilingual protocol, or safety boundary fails. If no candidate satisfies Gate A, Saracura remains a research alpha; the gate is not weakened. A runtime integration or a change to either closed manifest requires a separately reviewed increment.
