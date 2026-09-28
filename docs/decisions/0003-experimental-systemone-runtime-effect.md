---
title: ADR 0003: Experimental open decision runtime effect
kind: decision
area: platform
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/decisions/0003-experimental-systemone-runtime-effect.md
globalRef: qmd://saracura/docs/decisions/0003-experimental-systemone-runtime-effect.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-28
sourceRefs:
  - github:#40
  - docs/action/specs/phase5c-fast-public-preview.md
related:
  - README.md
  - docs/README.pt-BR.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/decisions/0002-open-model-runtime-and-readiness.md
  - docs/action/specs/phase5c-fast-public-preview.md
supersedes: []
supersededBy: []
sensitivity: public
---

# ADR 0003: Experimental open decision runtime effect

## Status

Accepted for the Phase 5C experimental GitHub prerelease on 2026-09-28. This ADR has no effect on Gate A, the historical candidate disposition, or the readiness manifest.

## Context

Phase 5C ships the smallest honest public experimental preview of Saracura around existing open checkpoints. It preserves the generic System One wire adapter and adds a direct CPU path for the pinned Apache-2.0 Julia-1 checkpoint. This ADR records the installed experimental runtime effect without promoting any checkpoint to production readiness or weakening the Phase 5B historical disposition.

## Decision

Saracura installs one optional loopback-only `systemone` backend for exactly `universal-choice@phase5c-systemone.v1`. The backend:

- accepts only literal `http://127.0.0.1:<port>` endpoints, disables environment proxies, and never follows redirects;
- requires explicit endpoint, model alias, and checkpoint identity supplied by the operator via `--endpoint`, `--model-id`, `--model-revision`, and `--checkpoint-sha256`;
- accepts authentication only through the `SARACURA_SYSTEMONE_API_KEY` environment variable; a bearer value is never accepted through argv, included in errors, diagnostics, or reports;
- preserves and validates the raw serialized values from the wire;
- requires exact question and option identity and finite values in `[0, 1]`;
- accepts only a documented serialization residual bounded by `max(1e-6, option_count * 0.00005 + 1e-9)`;
- rejects a larger residual;
- renormalizes accepted values to an exact internal distribution using `math.fsum`, accepted when the final sum is within `1e-12` of one;
- retains the raw serialized values in the existing `raw_scores` response field and records residual/normalization evidence only in the sanitized live-smoke report, without changing the public response schema;
- never presents the result as calibrated by Saracura or automation-ready.

The installed engine contract gains optional `normalized_probabilities` and `selected_choice` fields on `ScoredChoice`. Existing backends leave them absent and preserve the current logits-to-softmax behavior. The System One adapter supplies both the raw serialized mapping and the exact normalized mapping; the engine validates both, uses the normalized mapping directly, and never applies softmax to probabilities.

Saracura also installs one opt-in `julia` backend for exactly
`universal-choice@phase5c-julia.v1`. It verifies the pinned Julia-1 weight
digest, imports the separately installed public runtime lazily, runs on CPU,
accepts only Choice requests in PT-BR or English, and validates Julia's complete
softmax distribution before returning the same uncalibrated, abstained Saracura
response. The default Saracura installation remains lightweight.

The wire contract follows the public TypeSafe OpenAPI shape as instantiated by the pinned Kev source revision: questions are keyed by ID; every Choice question contains exactly `type`, `instructions`, and `criteria`; request fields are exactly `model`, `questions`, and `state`; a success response requires `model`, `answers`, and `usage`, with only the known optional `latency_ms` extension. Each call has a finite 30-second timeout and sends exactly one question; no together-call or request cache is claimed.

## Consequences

- The `systemone` backend is opt-in and isolated. Default installation stays lightweight and offline.
- ADR 0002 and candidate manifest v3 remain immutable historical inputs. This increment adds ADR 0003 but does not create a candidate-manifest successor.
- Kev remains `reject_local` for the frozen managed-systems protocol with no candidate claims.
- The README and release notes may say that the direct Julia-1 backend completed an experimental local CPU smoke with the pinned public source and checkpoint. This is not calibration, production readiness, or a comparative superiority claim.
- The terms `usable_developer_preview`, `third_party_checkpoint_supported`, `production_ready`, calibrated, and automation-ready remain forbidden.
- The Phase 5B `reject_local` result remains unchanged and is not reclassified. The published result does not establish that Kev-4B is unusable; four-decimal serialization is a plausible, unverified explanation that motivates the corrected adapter boundary. The historical run is preserved as-is.

## Reversal conditions

Revert the adapter release commit and documentation without touching historical evidence. After a GitHub prerelease is published, mark it withdrawn in the release notes, delete no immutable tag or artifact silently, and publish a corrective `0.1.0a3` when needed. No PyPI publication is part of this increment.
