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
lastReviewedAt: 2026-09-23
sourceRefs: []
related:
  - README.md
  - SECURITY.md
  - CONTRIBUTING.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/action/specs/phase4d-experimental-universal-choice.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura contributor instructions

Saracura is a standalone, local-first typed decision engine. This repository must remain usable without Felhen, AIOS, private services, private data, or private configuration.

## Current product boundary

- `v1alpha1` is research-only and supports `choice` for known, versioned workflows plus the exact experimental `universal-choice@phase4d-laya.v1` dynamic workflow. The planned `universal-choice@phase4e-saracura-ranker.v1` contract is unsupported until Phase 4E.3 seals a real checkpoint that passes its holdout gate.
- Boolean, ordinal, other dynamic-label product paths, HTTP serving, training, model downloads, private adapters, and the planned `saracura-universal` backend are out of scope until their own reviewed increments.
- The deterministic fixture backend is test infrastructure, not a model and not evidence of decision quality.
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
- Compiled state encoding must not depend on questions or criteria and must execute once per request. The exact Phase 4D `laya-universal` backend is the only installed exception: it may jointly encode state, instruction, and ordered choices for `universal-choice@phase4d-laya.v1`, remains opt-in and research-only, rejects truncation, returns uncalibrated abstained rankings, and never authorizes automation. Phase 4E.1 may contain planned-only policy, renderer, ranker, checkpoint, and workflow contracts, but it must not register `saracura-universal` in the runtime or CLI; that remains unsupported until Phase 4E.3. Other question-conditioned candidate adapters remain checkout-only until a separately reviewed increment updates this policy.
- Calibration compatibility is fail-closed across every declared axis. Immutable artifacts must use a sibling temporary file plus an atomic create-if-absent operation; an existing revision is never overwritten.
- Never add telemetry, remote fallback, arbitrary model paths, or request-triggered downloads.
