---
title: Two-tier typed decision architecture
kind: decision
area: platform
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/decisions/0001-two-tier-decision-architecture.md
globalRef: qmd://saracura/docs/decisions/0001-two-tier-decision-architecture.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-22
sourceRefs:
  - https://docs.typesafe.ai/concepts/how-to-build-with-system-one
  - https://docs.typesafe.ai/primitives/choice
  - https://github.com/NandhaKishorM/laya
related:
  - README.md
  - docs/README.pt-BR.md
  - docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
  - docs/action/specs/phase3d-typesafe-native-control.md
  - docs/action/specs/phase4c-universal-backend-bakeoff.md
  - docs/action/specs/phase4c2-local-candidate-adapters.md
  - docs/action/specs/phase4c3c-typesafe-response-contract-recovery.md
  - docs/action/specs/phase4c4-architecture-decision.md
supersedes: []
supersededBy: []
sensitivity: public
---

# ADR 0001: Two-tier typed decision architecture

## Status

Accepted for research architecture on 2026-09-22. The Phase 4C production selection disposition is `insufficient_evidence`.

## Context

Saracura needs to accept new typed Choice schemas without requiring a task-specific head, while preserving the measured systems advantage of a compiled head for stable, repeated, high-volume workflows. Phase 4C has systems evidence for the compiled MiniLM head, local Laya and mDeBERTa candidates, and TypeSafe Jev controls. It has no qualifying human-original PT-BR artifact, cross-locale quality result, production calibration, or selective-risk result. Therefore a research architecture decision is possible, but no backend is selected for production or automation.

The native TypeSafe continuation is a sealed terminal partial: it preserved its completed coverage and stopped at `choice_not_maximal`. The operator stopped further recovery. This closes that evidence lane administratively, not as a complete artifact; conclusions that need its missing cells are `insufficient_evidence`.

## Decision

Saracura adopts a two-tier research architecture under one typed decision API:

- The `universal` tier is for new typed Choice schemas and question-conditioned state without task-specific heads. The Laya multilingual/option-marker family is the next experimental implementation hypothesis because its exact candidate completed the broad local MacBook matrix with deterministic repeats and a better measured systems fit than the measured mDeBERTa baseline. This is neither a quality winner nor a production selection.
- The `compiled` tier is optional specialization for stable, repeated, high-volume workflows. The current MiniLM head remains limited to its frozen research workflow until human-original PT-BR evaluation and calibration are available.
- TypeSafe Jev remains an external product control and design reference. It is not a Saracura default backend, remote fallback, runtime dependency, calibration authority, or source of training labels. Its captured native response-contract inconsistency blocks confidence-gated automation for this integration on this evidence.

This decision separates research architecture, experimental implementation priority, and production selection. It does not change `DecisionEngine`, register a backend, acquire a model, call a provider, or authorize automation.

## Evidence ledger

Every factual measurement below is limited to the indicated immutable artifact-manifest SHA-256 and allowed use.

| Evidence | Artifact-manifest SHA-256 | Allowed use |
| --- | --- | --- |
| Phase 3C local compiled and OpenRouter Jev throughput | `f12e1444a41b4b6257729a504e74e60c401bfe5aa53c4f132eafc09393fbc9e5` | synthetic systems performance only |
| Phase 3D native TypeSafe control | `3cd4bf8dbc4e516baadc9effc0c0f0f8b4fa2d40ca0177e4cb1a63e8b57ca401` | synthetic systems comparison only |
| Phase 4C.2 Laya CPU smoke | `b80ec6a92c1e5879db863fa87799e4594bea31f3782aa106eff3eb85da2f8688` | local systems compatibility only |
| Phase 4C.2 Laya MPS | `f7ca4f3327202c98691bc5f6a7abbe1d222ab7ee976473d2a10fcad370b4f798` | local systems compatibility only |
| Phase 4C.2 mDeBERTa CPU smoke | `c73c1fe9fe3acbeb396624aa32def1f7b7b2ce7a10faad8044647b5775da4ea5` | local systems compatibility only |
| Phase 4C.2 mDeBERTa MPS | `57d6c0eb43d1d4881af13e1f4bdfcff17153954d8ee182979180b1d86cbc7e58` | local systems compatibility only |
| Phase 4C.3c native TypeSafe continuation | `bcc62247e37a261877e4c75f9b7a14f430ab5a8ba3231076c5a25c812a9681b3` | partial response-contract and cost evidence only |

The Phase 3C artifact measured compiled MiniLM on Apple MPS at 177.0 decisions/s for `N=1` and 991.8 decisions/s for `N=128`, with approximately 800 MB RSS after load, in its synthetic workflow. The Laya MPS artifact measured 26.9 decisions/s, approximately 4.30 GB peak RSS, a deterministic repeat match, 52 fully supported decisions, 130 decisions with declared choice truncation, and one unsupported decision. Its 24.8-second load and 6.74-second matrix-level p50 are not comparable to Phase 3C's per-request MiniLM figures. The mDeBERTa MPS artifact measured 0.91 decisions/s, approximately 3.20 GB peak RSS, and 1,028 decisions with declared state truncation; its CPU smoke does not overturn that MPS disposition.

The Phase 3D artifact measured native TypeSafe Jev at 1.39, 11.40, and 21.47 items/s for batch sizes 1, 8, and 20; the Phase 3C artifact measured OpenRouter Jev at 2.68, 20.30, and 49.41 items/s on the same synthetic workload. WAN and provider state are part of those observations. The Phase 4C.3c artifact recorded 51 completed requests, 132 completed decisions, and cumulative governed use of USD 0.06114981 before it stopped at `choice_not_maximal`; it is neither billing proof nor a quality result, and it does not establish that every TypeSafe response violates documentation.

## Comparison and unsupported cells

| Phase 4C.4 axis | Evidence-bound finding | Unsupported or incomparable cell |
| --- | --- | --- |
| Generality on unseen taxonomies and domains | `insufficient_evidence` | No authorized unseen-taxonomy or cross-domain quality evaluation. |
| PT-BR behavior and cross-locale parity | `insufficient_evidence` | Systems inputs do not establish PT-BR quality or parity. |
| Quality and selective-risk evidence | `insufficient_evidence` | No qualifying human-original PT-BR, quality, calibration, or selective-risk artifact. |
| Scaling with `N`, `Q`, `K`, state length, and choice-description length | Compiled MiniLM has synthetic `N=1` and `N=128` throughput evidence; Laya and mDeBERTa capacity behavior is recorded in their local matrices. | `insufficient_evidence` for a comparable cross-candidate scaling conclusion across every listed dimension. |
| Cold/warm latency, throughput, memory, artifact size, and MacBook usability | The ledger records measured MiniLM, Laya, mDeBERTa, and Jev systems observations; Laya completed the broad MacBook matrix. | Artifact size is `insufficient_evidence`. Benchmark shapes differ, so no universal latency or “times faster” claim is valid. |
| Calibration stability across question type and option count | `insufficient_evidence` | No disjoint calibration result across those axes. |
| Supply-chain, licensing, data, and operational complexity | Local candidates remain research-only with unresolved provenance/licensing considerations; remote Jev adds provider and WAN dependence. | `insufficient_evidence` for a production supply-chain or licensing approval. |
| Compiled specialization for stable high-volume work | The Phase 3C synthetic MiniLM measurements support retaining compiled specialization as a research systems optimization for its known workflow. | `insufficient_evidence` for general quality, production benefit, or comparison with universal protocols. |

## Candidate dispositions

These seven rows cover `decision-backend-candidates.v2`; evidence does not transfer from one implementation or family to another.

| Candidate | Research disposition | Production disposition |
| --- | --- | --- |
| Saracura compiled MiniLM | `recommend` for stable high-volume known workflows | `insufficient_evidence` |
| Laya multilingual / option-marker family | `recommend` as the next universal implementation hypothesis; systems-only evidence | `insufficient_evidence` |
| mDeBERTa NLI / poorjev-style pairwise baseline | `reject` as the primary measured MacBook MPS path; retain reference-only role; this does not reject every NLI implementation | `insufficient_evidence` |
| TypeSafe Jev | `recommend` only as an external product control; `reject` as default/local runtime or automatic fallback | `insufficient_evidence` |
| Qwen/system-one | `insufficient_evidence` | `insufficient_evidence` |
| Von option-marker | `insufficient_evidence`; Laya family evidence does not transfer | `insufficient_evidence` |
| DiffusionGemma OpenJev | `reject` for the initial MacBook lane under the documented exclusion | `insufficient_evidence` |

## Consequences

New Choice schemas belong to the universal tier; users do not need to adopt a predefined model pack or commission a compiled head to express them. Compiled specialization remains optional and is justified only for stable high-volume work. The first universal implementation is experimental. No model is production-approved, and no result from this ADR authorizes automation.

The sealed partial TypeSafe lane may inform the bounded response-contract and cost discussion above, but it cannot support a candidate-quality recommendation or be relabeled complete. No additional provider call or recovery is authorized by this ADR.

## Next gate: Phase 4D

A separately reviewed Phase 4D spec may introduce an experimental, opt-in universal Choice backend with scoped invariants. It must have no default dependency, request-triggered download, remote fallback, provider credential in the runtime, or silent state/question/choice truncation. It must expose explicit capacity and computation limits. Compiled backends must independently encode a state once and reuse it; universal backends may jointly encode state, instructions, and choices. Phase 4D begins research-only and cannot authorize automation. Human-original PT-BR evaluation and disjoint calibration remain required before any production claim.

## Reversal conditions

Revisit this architecture if reviewed evidence shows that an opt-in universal backend cannot meet the stated capacity and explicit-limit boundaries, if a compiled workflow no longer has stable high-volume value, or if new human-original PT-BR, cross-locale, calibration, selective-risk, licensing, or operational evidence materially changes these bounded dispositions. A reversal requires a new reviewed ADR or spec; it does not authorize mutation of sealed artifacts or a provider retry.
