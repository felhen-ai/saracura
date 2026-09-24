---
title: Phase 4E.2i reviewer timeout recovery
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2i-reviewer-timeout-recovery.md
globalRef: qmd://saracura/docs/action/specs/phase4e2i-reviewer-timeout-recovery.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
related:
  - docs/action/specs/phase4e2h-model-pair-upgrade.md
  - benchmarks/phase4e_pipeline.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2i reviewer timeout recovery

## Decision

Keep the v7 model pair, protocol, routing, prices and thresholds unchanged. Raise only the live pilot HTTPS socket timeout from 30 to 120 seconds in a create-only v8 run. Historical corpus execution retains its 30-second default.

## Verified context

- The first v7 GPT author request settled successfully.
- The first Gemini reviewer request exceeded the fixed 30-second `HTTPSConnection` timeout and was recorded as uncertain.
- V7 stopped before any row resolution and cannot be resumed.
- `benchmarks/phase4e_pipeline.py:271-295` owns the only live HTTPS transport and previously fixed its socket timeout at 30 seconds.
- `benchmarks/saracura_universal_pilot.py:1335-1383` validates the pilot plan and paths before constructing that transport.
- `benchmarks/saracura_universal_pilot.py:670-697` seals every plan field by exact canonical equality, so the timeout can be bound into the immutable plan.

## Scope, invariants and acceptance

- Add v8 manifest, plan, report and exact v8 paths with `transport_timeout_seconds=120` bound into the plan.
- Parameterize the transport timeout with a validated positive integer; default remains 30 seconds.
- V8 passes 120 explicitly only when constructing the pilot transport.
- Do not replay v7, relax ZDR/privacy, retry uncertain calls or change quality gates.
- Tests prove default 30 and pilot 120 are wired without opening a socket.
- Full QA and read-only review pass before execution; only complete `PASS` advances Phase 4E.3.

## Review resolution

Cross-model critique remains degraded after repeated operational unavailability. GPT/Terra read-only review and complete local QA remain required before v8 execution.

The first read-only review found that the repository-wide manifest dispatcher did not route any of the historical v2-v7 recovery manifests or the new v8 manifest. The implementation now validates v8 semantically with `validate_pilot_recovery_policy`, binds v2-v7 to their exact canonical bytes, and proves all seven revisions plus tamper rejection. `python -m benchmarks.validate_manifests` now validates all 22 routed manifests.

The second GPT/Terra read-only review returned `PASS` with no blockers.

## Validation evidence

- Focused: `116 passed` across pilot, pipeline, corpus and documentation tests.
- Full QA after review fix: Ruff lint and format pass; mypy passes on 73 source files; `879 passed, 29 skipped`; all 22 routed manifests validate; source and wheel builds pass; `git diff --check` passes.
