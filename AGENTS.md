---
title: Saracura contributor instructions
kind: policy
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: AGENTS.md
globalRef: qmd://saracura/AGENTS.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-28
sourceRefs:
  - github:#40
  - docs/action/specs/phase5c-fast-public-preview.md
related:
  - README.md
  - SECURITY.md
  - CONTRIBUTING.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/decisions/0002-open-model-runtime-and-readiness.md
  - docs/decisions/0003-experimental-systemone-runtime-effect.md
  - docs/action/specs/phase4d-experimental-universal-choice.md
  - docs/action/specs/phase4f1-shadow-evaluation-and-email-reference.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura contributor instructions

Saracura is a standalone, local-first typed decision engine. This repository must remain usable without Felhen, AIOS, private services, private data, or private configuration.

## Current product boundary

- `v1alpha1` is research-only and supports `choice` for known, versioned workflows plus reviewed dynamic workflows. The optional loopback-only `systemone` backend serves `universal-choice@phase5c-systemone.v1`. The direct CPU `julia` backend serves only `universal-choice@phase5c-julia.v1`, verifies the pinned public Julia-1 checkpoint digest, loads only an explicit local snapshot, and returns uncalibrated abstained results with automation disabled.
- The public Phase 4F.1 `ShadowRunner` is a local, read-only evaluator over caller-selected minimized state and an editable `ShadowPolicy`. It has no Gmail, Outlook, IMAP, browser, provider, or network adapter and performs no mailbox reads or mutations. Decisions are content-free, uncalibrated, abstained rankings with `automation_allowed=false`; ranking weights are not confidence. Feedback is descriptive evaluation only and must not be reused for training, retrieval, prompt optimization, or calibration. A provider-specific mailbox adapter remains outside the public increment and requires a separately reviewed private Phase 4F.2.
- Boolean, ordinal, other dynamic-label product paths, HTTP serving, training, model downloads, public checkpoint publication, calibration, confidence thresholds, and private adapters are out of scope until their own reviewed increments.
- The deterministic fixture backend is test infrastructure, not a model and not evidence of decision quality.
- The checkout-only Phase 5D FAQ benchmark is an aggregate external control over the pinned `MTEB-BR/faq-bacen` derivative. It may read only the explicit caller cache, must not redistribute rows or emit raw text, and cannot authorize training, calibration, automation, or a general superiority claim.
- No result authorizes automation. Calibration status `verified_for_research` only describes compatibility with the declared research protocol.

## Development

Use Python 3.11+ and `uv`:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

Keep optional model backends in isolated extras. Default development installation must not download weights, datasets, or heavy ML runtimes. Use self-authored fixtures only unless a later data gate is explicitly approved.

Every new canonical Markdown file must include structural frontmatter with `title`, `kind`, `area`, `project`, `collection`, `owner`, `status`, `canonical`, `globalRef`, `reviewCadenceDays`, `lastReviewedAt`, `sourceRefs`, `related`, `supersedes`, `supersededBy`, and `sensitivity`.

## Safety invariants

- Keep Pydantic request, response, error, and calibration models closed to unknown fields.
- Normalize strings and keys with the versioned NFC transform before RFC 8785 canonicalization; reject post-normalization key collisions.
- Preserve string values as data. Never reinterpret a string as JSON.
- Frame semantic segments by byte length, never by ambiguous delimiters.
- Compiled state encoding must not depend on questions or criteria and must execute once per request. The exact Phase 4D `laya-universal` backend is the installed exception for `universal-choice@phase4d-laya.v1`: it may jointly encode state, instruction, and ordered choices for that workflow, remains opt-in and research-only, rejects truncation, returns uncalibrated abstained rankings, and never authorizes automation. Phase 4E.3c installed `saracura-universal` as the explicit `universal-choice@phase4e-saracura-ranker.v1` backend; it is a question-conditioned universal backend that does not claim the compiled tier's encode-state-once optimization, requires private verified artifacts, and remains opt-in, synthetic-only, and research-only. The Phase 5C systemone backend is installed for `universal-choice@phase5c-systemone.v1`, preserves raw serialized values, applies bounded normalization with `math.fsum`, and records residual/normalization evidence only in the sanitized live-smoke report. Other question-conditioned candidate adapters remain checkout-only until a separately reviewed increment updates this policy.
- Calibration compatibility is fail-closed across every declared axis. Immutable artifacts must use a sibling temporary file plus an atomic create-if-absent operation; an existing revision is never overwritten.
- Never add telemetry, remote fallback, arbitrary model paths, or request-triggered downloads.
