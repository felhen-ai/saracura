---
title: Phase 4C.3c operator runbook
kind: runbook
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/phase4c3c-operator.md
globalRef: qmd://saracura/docs/action/phase4c3c-operator.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs: []
related: [docs/action/specs/phase4c3c-typesafe-response-contract-recovery.md]
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4C.3c operator runbook

Run only after implementation review and explicit live authorization. Use the same Git clone and Git common directory as the sealed Phase 4C.3 artifact: its original claim is mandatory validation evidence. Validate the recovery plan first:

```bash
uv run python -m benchmarks.universal_remote_resume validate-plan
```

Before a live command, inspect the provider account UI for unexplained use without recording account or billing identifiers. Provide `TYPESAFE_API_KEY` only through the approved secret wrapper; never create `.env`, pass a secret in argv, or copy it into an artifact. The run requires `--allow-network --budget-usd 0.25`; it dispatches ordinals 10 through 56 once, in order, and stops on the first failure. Finalization and validation are credential-free and cannot dispatch.

The artifact is computed systems-compatibility evidence only. It preserves predecessor limits, distinguishes the ten reused measurements from 47 continuation measurements, and does not treat a completed continuation as a quality, calibration, or billing result.
