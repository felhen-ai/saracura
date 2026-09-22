---
title: Phase 4C.3 TypeSafe operator runbook
kind: runbook
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/phase4c3-operator.md
globalRef: qmd://saracura/docs/action/phase4c3-operator.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs: []
related:
  - docs/action/specs/phase4c3-controlled-remote-comparison.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4C.3 TypeSafe operator runbook

This checkout-only tool is not a Saracura runtime backend. It makes no request
unless the operator passes `--allow-network` and the inherited budget is exactly
`0.25`. The only credential contract is `TYPESAFE_API_KEY`, injected into the
child process by the approved secret manager. Do not create `.env` files or put
keys, artifact paths, prompts, or provider responses in tickets or Git.

After independently locating the seven sealed predecessor inputs, run the
command in the Phase 4C.3 spec. It uses only the literal TypeSafe endpoint and
the pinned `jev-1.13.0` model, sequentially, once per request, with no proxy,
redirect, retry, fallback, or SDK. A preflight failure makes no network call.

If the process stops, do not invoke `run` again. Use `finalize-partial` with the
same inputs to create a credential-free, no-network partial seal, then use
`validate-artifact` with those same inputs. Existing claims are intentionally
single-use across all worktrees sharing the Git common directory.

The result is systems-compatibility evidence only. It cannot establish quality,
calibration, Portuguese parity, a model winner, or authorization for automation.
