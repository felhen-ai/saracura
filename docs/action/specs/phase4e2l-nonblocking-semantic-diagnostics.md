---
title: Phase 4E.2l non-blocking semantic diagnostics
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2l-nonblocking-semantic-diagnostics.md
globalRef: qmd://saracura/docs/action/specs/phase4e2l-nonblocking-semantic-diagnostics.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
related:
  - docs/action/specs/phase4e2k-stable-independent-reviewer.md
  - benchmarks/saracura_universal_pilot.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2l non-blocking semantic diagnostics

## Decision

Preserve every v10 model, prompt, provider, privacy and quality gate while making the reviewer's independent semantic attestation genuinely diagnostic. Repeated inferred criterion roles remain recorded as criterion-role disagreement and no longer make an otherwise valid quality review a malformed response. Run the corrected protocol only in new create-only v11 paths.

## Verified context

- V10 settled 343 calls and resolved 122 tasks before one reviewer call became transport-uncertain.
- It accepted 65 rows and rejected 57, including 35 exhausted reviewer responses after 195 failed validations.
- A 512-versus-1024-token probe completed with `finish_reason=stop` at both capacities, excluding output truncation.
- Five repeated six-option probes each returned six role positions, one `matches_rule`, and only two unique roles. The strict internal semantic model rejected all five.
- The pilot already declares semantic mismatch non-blocking; answer agreement, genericity, exclusivity, natural-language and privacy gates do not depend on exact semantic-role agreement.
- Azure rejected a strict-schema probe that added `uniqueItems`, so the provider-supported schema subset cannot enforce the internal distinct-role invariant.

## Scope and invariants

- Add create-only v11 manifest, plan, work and report paths. V10 remains immutable and non-resumable.
- Add a pilot-only reviewer record whose semantic attestation preserves the closed scenario and role vocabularies, exact role-list capacity and `selected_role=matches_rule`, while permitting repeated inferred roles.
- Keep the corpus reviewer model and all production/training packet validators byte-for-byte unchanged. Pilot rows remain evaluation-only and cannot enter a training packet.
- Make the shared review resolver accept an explicitly injected Pydantic review model; its default remains the strict corpus `ReviewerRecord`.
- Keep GPT-4.1 Mini author, GPT-4.1 reviewer, Azure-only ZDR/no-collection routing, fallback disabled, temperature zero, no reasoning parameter and 512 reviewer output tokens.
- Raise only the fixed transport timeout from 120 to 240 seconds to cover the one long-tail call observed among 344 v10 calls.
- Preserve all acceptance thresholds, answer blindness, retry count, artifact provenance and report-only cost telemetry.

## Acceptance

- Tests prove duplicate pilot semantic roles bind successfully, are counted as criterion-role disagreement and do not bypass any quality or answer gate.
- Tests prove the default corpus resolver still rejects the same duplicate-role record.
- V2-v11 manifests validate; pre-v11 paths and manifest tampering fail closed before ledger or transport access.
- Full QA and GPT/Terra read-only review pass before the v11 network run.
- Only a complete v11 `PASS` authorizes Phase 4E.3. `FAIL` or `INCONCLUSIVE` remains fail-closed.

## Validation commands

```bash
uv run --isolated --locked ruff check .
uv run --isolated --locked ruff format --check .
uv run --isolated --locked mypy src benchmarks
uv run --isolated --locked pytest -q
uv run --isolated --locked python -m benchmarks.validate_manifests
uv build
git diff --check
```

## Review resolution

Cross-model critique remains explicitly degraded after repeated operational unavailability in the preceding Phase 4E recovery increments. GPT/Terra read-only review and full local QA are required before v11 execution.

Local pre-execution QA passed on 2026-09-24: Ruff lint and format checks, mypy, all 880 tests with 29 expected skips, all 25 routed manifests, package build and `git diff --check`. GPT/Terra then returned `PASS` with no blockers, confirming that the strict corpus resolver remains the default and the relaxed semantic observation model is injected only by the evaluation-only pilot.
