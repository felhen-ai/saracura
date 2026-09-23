---
title: Phase 4C.4 architecture decision record
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4c4-architecture-decision.md
globalRef: qmd://saracura/docs/action/specs/phase4c4-architecture-decision.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs:
  - https://docs.typesafe.ai/concepts/how-to-build-with-system-one
  - https://docs.typesafe.ai/primitives/choice
  - https://docs.typesafe.ai/confidence
related:
  - AGENTS.md
  - README.md
  - docs/README.pt-BR.md
  - docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
  - docs/action/specs/phase3d-typesafe-native-control.md
  - docs/action/specs/phase4b-human-ptbr-calibration.md
  - docs/action/specs/phase4c-universal-backend-bakeoff.md
  - docs/action/specs/phase4c2-local-candidate-adapters.md
  - docs/action/specs/phase4c3-controlled-remote-comparison.md
  - docs/action/specs/phase4c3c-typesafe-response-contract-recovery.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4C.4 architecture decision record

## Context

Phase 4C has now produced real systems evidence for the compiled Saracura MiniLM head, the local Laya multilingual option-marker candidate, the local mDeBERTa NLI candidate, and TypeSafe Jev through both OpenRouter and the native TypeSafe endpoint. The native Phase 4C.3 comparison ended with a sealed partial artifact after 51 of 57 requests and 132 decisions because the provider returned a selected choice that was not the maximum-probability option, contrary to the reviewed live Choice documentation. A proposed tail-recovery tool was deliberately abandoned after its final review found another integrity requirement; no further paid recovery attempt is authorized or required for this decision.

The available evidence is sufficient to decide the research architecture and the next implementation direction, but it is not sufficient to select any backend for production. All measured quality values come from synthetic or self-authored data, the human PT-BR Phase 4B pipeline has tooling but no qualifying real human artifact, cross-locale quality was not established, and no candidate has a production calibration or selective-risk result.

## Objective

Publish one versioned ADR that closes Phase 4C with a bounded architectural disposition and makes the public roadmap explicit. The ADR must decide the relationship among universal decisions, compiled specialization, local candidates, and remote controls without modifying runtime code or overstating the evidence.

## Decision to encode

Saracura adopts a two-tier research architecture:

1. A `universal` tier accepts new typed Choice schemas and question-conditioned state without task-specific heads. The option-marker encoder family is the first implementation hypothesis because Laya completed the broad MacBook execution matrix with deterministic repeats and materially better measured systems fit than the mDeBERTa NLI baseline. This is a direction for the next experimental runtime increment, not a quality winner or production selection.
2. A `compiled` tier remains available for stable, repeated, high-volume workloads. The current MiniLM head demonstrates the throughput value of specialization, but remains restricted to its frozen research workflow until human-original PT-BR evaluation and calibration exist.
3. TypeSafe Jev remains an external product control and design reference. It must not become Saracura's default backend, remote fallback, runtime dependency, calibration authority, or source of training labels. Its native response-contract inconsistency blocks confidence-gated automation based on the captured Phase 4C.3c evidence.
4. mDeBERTa NLI is rejected as the primary MacBook MPS universal path for the measured configuration. It may remain a checkout-only CPU/reference baseline; this rejection is not a claim about every NLI model or hardware configuration.
5. Qwen/system-one, Von, poorjev, and DiffusionGemma receive no positive selection from this phase because the intended comparable real-candidate evidence is absent. They remain `insufficient_evidence` or documented exclusion according to the existing registry.

The Phase 4C production-selection disposition is `insufficient_evidence`. The research architecture recommendation and the production-selection disposition must be stated separately so that “recommended next implementation” cannot be read as “safe for automation.”

## Evidence-lane closure rule

Reconcile the parent Phase 4C and Phase 4C.3c specs with the already stated rule that Phase 4C.4 may begin when an intended lane is complete or explicitly declared insufficient. A sealed terminal partial may close a lane only when the operator has stopped further recovery, the failure and completed coverage are preserved, and the ADR assigns `insufficient_evidence` to every conclusion that depended on the missing cells. This administrative closure does not relabel the partial artifact as complete, authorize another provider call, or permit the incomplete lane to support a candidate-quality recommendation.

The parent spec must link the new ADR and state this closure rule next to the Phase 4C.4 gate. The Phase 4C.3c next-gate section must say that either a complete continuation or a sealed terminal partial explicitly declared insufficient may proceed to Phase 4C.4 under these restrictions.

## Evidence ledger

The ADR must bind each factual claim to these immutable artifact-manifest SHA-256 values:

| Evidence | Artifact-manifest SHA-256 | Allowed use |
| --- | --- | --- |
| Phase 3C local compiled and OpenRouter Jev throughput | `f12e1444a41b4b6257729a504e74e60c401bfe5aa53c4f132eafc09393fbc9e5` | synthetic systems performance only |
| Phase 3D native TypeSafe control | `3cd4bf8dbc4e516baadc9effc0c0f0f8b4fa2d40ca0177e4cb1a63e8b57ca401` | synthetic systems comparison only |
| Phase 4C.2 Laya CPU smoke | `b80ec6a92c1e5879db863fa87799e4594bea31f3782aa106eff3eb85da2f8688` | local systems compatibility only |
| Phase 4C.2 Laya MPS | `f7ca4f3327202c98691bc5f6a7abbe1d222ab7ee976473d2a10fcad370b4f798` | local systems compatibility only |
| Phase 4C.2 mDeBERTa CPU smoke | `c73c1fe9fe3acbeb396624aa32def1f7b7b2ce7a10faad8044647b5775da4ea5` | local systems compatibility only |
| Phase 4C.2 mDeBERTa MPS | `57d6c0eb43d1d4881af13e1f4bdfcff17153954d8ee182979180b1d86cbc7e58` | local systems compatibility only |
| Phase 4C.3c native TypeSafe continuation | `bcc62247e37a261877e4c75f9b7a14f430ab5a8ba3231076c5a25c812a9681b3` | partial response-contract and cost evidence only |

The ADR may include the following measured facts, preserving their scope:

- Compiled MiniLM on Apple MPS measured 177.0 decisions/s at `N=1` and 991.8 decisions/s at `N=128`, with approximately 800 MB RSS after load in the Phase 3C synthetic workflow.
- Laya MPS completed the real local matrix with 26.9 decisions/s, approximately 4.30 GB peak RSS, deterministic repeat match, 52 fully supported decisions, 130 decisions requiring declared choice truncation, and one unsupported decision. Its 24.8-second load and 6.74-second matrix-level p50 are not directly comparable to the Phase 3C per-request MiniLM figures.
- mDeBERTa MPS measured 0.91 decisions/s, approximately 3.20 GB peak RSS, and 1,028 decisions requiring declared state truncation. Its CPU smoke is not sufficient to overturn the MPS disposition.
- Native TypeSafe Jev measured 1.39, 11.40, and 21.47 items/s for batch sizes 1, 8, and 20 in Phase 3D, while OpenRouter Jev measured 2.68, 20.30, and 49.41 items/s on the same synthetic workload. WAN and provider state are included.
- Phase 4C.3c sealed 51 completed requests and 132 completed decisions, with cumulative governed use of USD 0.06114981, then stopped at `choice_not_maximal`. This is not billing proof, a quality result, or a general statement that every TypeSafe response violates its documentation.

## Required comparison

The ADR must include one compact table that covers every Phase 4C.4 axis:

- generality on unseen taxonomies and domains;
- PT-BR behavior and cross-locale parity;
- quality and selective-risk evidence;
- scaling with `N`, `Q`, `K`, state length, and choice-description length;
- cold/warm latency, throughput, memory, artifact size, and MacBook usability;
- calibration stability across question type and option count;
- supply-chain, licensing, data, and operational complexity;
- whether compiled specialization materially improves stable high-volume work.

For any axis without authorized evidence, the cell must say `insufficient_evidence`, not infer a result from upstream claims. Artifact size is also `insufficient_evidence` unless an existing sealed result contains the value. The table must distinguish incomparable benchmark shapes and must not calculate a universal “times faster” headline across different protocols.

## Candidate dispositions

The ADR must assign an explicit disposition and bounded rationale to every candidate class already registered:

| Candidate | Research disposition | Production disposition |
| --- | --- | --- |
| Saracura compiled MiniLM | `recommend` for stable high-volume known workflows | `insufficient_evidence` |
| Laya multilingual / option-marker family | `recommend` as the next universal implementation hypothesis | `insufficient_evidence` |
| mDeBERTa NLI / poorjev-style pairwise baseline | `reject` as the primary measured MacBook MPS path; retain reference-only role; this does not reject every NLI implementation | `insufficient_evidence` |
| TypeSafe Jev | `recommend` only as an external product control; `reject` as default/local runtime or automatic fallback | `insufficient_evidence` |
| Qwen/system-one | `insufficient_evidence` | `insufficient_evidence` |
| Von option-marker | `insufficient_evidence`; family evidence does not transfer from Laya | `insufficient_evidence` |
| DiffusionGemma OpenJev | `reject` for the initial MacBook lane under the existing documented exclusion | `insufficient_evidence` |

## Public narrative updates

Update `README.md` and `docs/README.pt-BR.md` with a short architecture section that says:

- the project is pursuing one typed decision API with universal and compiled execution tiers;
- users do not need to adopt predefined model packs to express a new Choice schema;
- compiled specialization is an optional optimization for stable, high-volume decisions;
- the first universal implementation remains experimental and no model is production-approved;
- TypeSafe/Jev is a benchmark and design reference, not a Saracura dependency.

The English and PT-BR sections must be semantically equivalent without literal translation requirements. Existing research-only warnings remain unchanged or stronger.

## Next implementation gate

The ADR must name a separate Phase 4D spec as the next step. Phase 4D may introduce an experimental opt-in universal Choice backend and scoped invariants, but only after review. It must preserve these boundaries:

- no default dependency or request-triggered download;
- no remote fallback or provider credential in the runtime;
- no silent state, question, or choice truncation;
- explicit capacity and computation limits;
- compiled backends encode an independent state once and reuse it;
- universal backends may jointly encode state, instructions, and choices;
- the implementation begins as research-only and cannot authorize automation;
- human-original PT-BR evaluation and disjoint calibration remain the gate for any production claim.

The ADR itself does not implement Phase 4D, change `DecisionEngine`, register a backend, add dependencies, download weights, call providers, or create a release.

## Files and scope

Allowed files:

- `docs/action/specs/phase4c4-architecture-decision.md`;
- `docs/action/specs/phase4c-universal-backend-bakeoff.md`, limited to recording that an explicitly terminated lane may close as `insufficient_evidence` and linking the ADR;
- `docs/action/specs/phase4c3c-typesafe-response-contract-recovery.md`, limited to reconciling its next-gate text with its existing explicit-insufficiency rule and linking the ADR;
- new `docs/decisions/0001-two-tier-decision-architecture.md`;
- `README.md`;
- `docs/README.pt-BR.md`.

No code, test, manifest, lockfile, dependency, runtime API, artifact, benchmark result, or operator runbook may change. The local `.artifacts/` directories remain ignored and are never copied into Git.

## Acceptance criteria

1. The ADR has valid structural frontmatter, a stable decision ID, status, context, decision, evidence ledger, candidate dispositions, unsupported cells, consequences, next gate, reversal conditions, and explicit production `insufficient_evidence` disposition.
2. Every quantitative claim is traceable to one of the seven exact artifact-manifest digests and preserves the original evidence authority.
3. The ADR distinguishes research architecture, experimental implementation priority, and production selection.
4. All Phase 4C.4 comparison axes are represented, with missing evidence explicitly identified.
5. All seven candidate rows in `decision-backend-candidates.v2` receive bounded research and production dispositions without transferring evidence across implementations or families.
6. TypeSafe/Jev is retained as an external control but excluded as a runtime dependency, automatic fallback, calibration authority, or training-label source.
7. The public README sections in English and PT-BR describe the two-tier direction and high-volume compiled advantage without claiming a production-ready model.
8. No provider call, model acquisition, artifact mutation, dependency change, runtime change, or generated benchmark occurs.
9. Repository validation passes with no new dependency, runtime module, model, dataset, artifact, or executable content in the wheel or sdist; ordinary README/package-metadata changes are allowed.

## Validation

```bash
git diff --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python benchmarks/validate_default_environment.py
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
```

The final QA must also verify that the diff is limited to the six allowed files and that no `.artifacts`, model, cache, secret, or generated benchmark file is tracked.

## Rollout and rollback

This is a documentation-only architecture decision. Rollout is merge to `main`; there is no deploy or provider action. Rollback is a Git revert of the ADR and README edits. Future Phase 4D code requires its own reviewed spec and does not inherit authorization to change runtime behavior from this ADR.

## Risks

- The public may read “recommend Laya family” as an endorsement of Laya quality. Mitigation: always pair it with “implementation hypothesis,” systems-only evidence, and production `insufficient_evidence`.
- The compiled benchmark can produce attractive speed ratios that are invalid across different workloads. Mitigation: publish absolute measurements and protocol boundaries, not a universal speedup headline.
- The incomplete TypeSafe lane can be overgeneralized into a provider-quality claim. Mitigation: describe the exact captured inconsistency and its consequence for this integration only.
- A two-tier design can become pack-centric if every workflow requires a compiled head. Mitigation: the universal tier owns new schemas; compiled heads are optional optimizations for proven stable volume.
- PT-BR-first positioning can be mistaken for current PT-BR superiority. Mitigation: state that PT-BR is the first evidence and governance lane, while quality and cross-locale parity remain unproven.
