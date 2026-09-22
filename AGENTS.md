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
lastReviewedAt: 2026-09-21
sourceRefs: []
related:
  - README.md
  - SECURITY.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura contributor instructions

Saracura is a standalone, local-first typed decision engine. This repository must remain usable without Felhen, AIOS, private services, private data, or private configuration.

## Current product boundary

- `v1alpha1` is research-only and supports `choice` for known, versioned workflows.
- Boolean, ordinal, dynamic-label product paths, HTTP serving, training, model downloads, and private adapters are out of scope until their own reviewed increments.
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
- State encoding must not depend on questions or criteria and must execute once per request.
  This invariant governs runtime decision backends registered with
  `DecisionEngine`; installed runtime backends remain restricted to the
  compiled specialized contract. Benchmark-only reference adapters under
  `benchmarks/` may model question-conditioned computation, and the explicitly
  checkout-only Phase 4C.2 candidate adapters may execute reviewed learned
  models there, but neither class is registered as a runtime backend.
- Calibration compatibility is fail-closed across every declared axis. Immutable artifacts must use a sibling temporary file plus an atomic create-if-absent operation; an existing revision is never overwritten.
- Never add telemetry, remote fallback, arbitrary model paths, or request-triggered downloads.
