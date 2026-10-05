---
title: Model-first product direction
kind: decision
area: product
project: saracura
collection: saracura
owner: saracura-maintainers
status: historical
canonical: docs/decisions/0004-model-first-product-direction.md
globalRef: qmd://saracura/docs/decisions/0004-model-first-product-direction.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-29
sourceRefs:
  - github:#43
related:
  - README.md
  - docs/README.pt-BR.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/decisions/0002-open-model-runtime-and-readiness.md
  - docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md
supersedes: []
supersededBy:
  - docs/decisions/0005-open-base-benchmark-first.md
sensitivity: public
---

# ADR 0004: Model-first product direction

## Status

Superseded by [ADR 0005](0005-open-base-benchmark-first.md) on 2026-10-02. Accepted on 2026-09-29 as the product-direction constraint for the first public Saracura model release. It changes roadmap and public-positioning priority; it does not claim that an unpublished checkpoint already exists or authorize automation.

## Context

Saracura has accumulated a rigorous typed runtime, evaluation contracts, benchmark runners, external controls, and integrations with third-party checkpoints. Those components reduced technical uncertainty, but the public artifact and documentation began to present the project as a model laboratory and integration harness. A third-party checkpoint became the recommended preview path and the center of the release narrative.

That direction is inconsistent with the intended product. Saracura must become a downloadable local decision model with its own weights, model card, measurements, and user-facing quickstart. Runtime and evaluation infrastructure are supporting capabilities, not the primary artifact.

The previous readiness plan also coupled two different milestones: publishing a useful model and authorizing calibrated automation. Requiring every automation-safety property before the first model release delays the artifact users need in order to evaluate and adopt the project.

## Decision

Saracura adopts a model-first product direction.

The critical path for the next release is:

1. select a practical open base architecture;
2. train or adapt a Saracura-owned checkpoint;
3. publish immutable weights and a model card;
4. provide a one-command local inference path;
5. publish directly comparable PT-BR quality and hardware evidence;
6. state only the narrow advantage demonstrated by that evidence.

The first checkpoint targets the `universal` role in ADR 0001. The `compiled` role remains an optional later optimization for stable high-volume workflows and may proceed only when it improves a measured quality-throughput frontier.

The release ladder is split into two independent milestones:

- **Model Preview:** downloadable Saracura-owned checkpoint, local quickstart, model card, public PT-BR evaluation, hardware measurements, limitations, and a scoped evidence-backed claim. It may remain uncalibrated and must not authorize automation.
- **Calibrated Automation:** calibration fitted and evaluated on disjoint records, selective-risk evidence, thresholds, workflow-specific safety policy, and explicit automation authorization where appropriate.

Model Preview is the immediate milestone. Calibrated Automation is not a prerequisite for publishing Model Preview.

## Roadmap filter

A new adapter, harness, candidate integration, dataset pipeline, or benchmark facility belongs on the model-release critical path only if it directly contributes to at least one of these outcomes:

1. checkpoint selection or training;
2. model quality, footprint, latency, or throughput measurement;
3. local model packaging and execution;
4. correction of a measured weakness in the released model.

If none applies, the work is deferred. Architectural completeness, support for another third-party checkpoint, or a more elaborate harness is not sufficient by itself.

## Public positioning

The public homepage leads with the model being built, the intended deployment envelope, and the next concrete artifact. It must remain explicit that the first public Saracura checkpoint is not available until it is actually released.

Third-party checkpoints may appear in detailed research records as controls under a common protocol. They must not be presented as the recommended Saracura model, the headline of a Saracura release, or the organizing principle of the public roadmap.

Comparative claims require a Saracura-owned checkpoint and directly comparable evidence. Claims are bounded by model size, task, language, hardware, protocol, and metric. The project does not claim to be the best local decision model overall.

## Consequences

- The README becomes a product front door rather than a chronological phase log.
- Detailed experimental instructions and negative results remain in versioned research records.
- The next release must center on a Saracura-owned checkpoint rather than another runtime adapter.
- PT-BR is the opening evaluation advantage, not a permanent language restriction.
- High-volume typed decisions remain the architectural differentiator to measure.
- Calibration and automation safety continue, but in a subsequent evidence gate.

## Reversal conditions

Revisit this decision only if measured evidence shows that publishing and maintaining a first-party checkpoint is infeasible or provides no user value relative to a model-independent runtime. A difficult training run, a stronger external release, or an opportunity to integrate another checkpoint is not sufficient evidence to reverse the direction.
