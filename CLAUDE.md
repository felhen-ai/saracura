---
title: Saracura contributor instructions
kind: policy
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: CLAUDE.md
globalRef: qmd://saracura/CLAUDE.md
reviewCadenceDays: 30
lastReviewedAt: 2026-10-05
sourceRefs:
  - docs/decisions/0005-open-base-benchmark-first.md
related:
  - README.md
  - SECURITY.md
  - CONTRIBUTING.md
  - docs/decisions/0005-open-base-benchmark-first.md
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

- The direction is ADR 0005: open base models, benchmark first. The public artifacts are the PT-BR typed-decisions benchmark (`felhen-ai/ptbr-typed-decisions-bench`) and the fine-tuned checkpoints (`felhen-ai/saracura-ptbr-v0` and successors), both on the Hugging Face Hub. The method is the set of plain scripts in `research/`.
- Checkpoints derive from open base models under permissive licenses (Laya multilingual, the Kev recipe on Qwen 3.5 4B) with explicit attribution. Do not reimplement training loops that upstream projects publish; do not train from scratch.
- Training may use internal, non-redistributed data (marketplace listings, Felhen documents) with labels from open-weight models. Model cards must declare it. No output of a hosted model with usage restrictions enters training or published data.
- Every result is reported on the benchmark with balanced accuracy, option order shuffled per item, against the baselines published alongside it. Teacher agreement is reported as agreement, not accuracy. Comparative claims stay bounded by the benchmark.
- `src/saracura` and `benchmarks/` are the research runtime that preceded ADR 0005 (ADRs 0001 to 0004, Phases 2 to 5). They remain as the typed decision API and a historical record; they are not the critical path. Their safety invariants below still apply to any change there.
- Public issues from the earlier roadmap (#45 to #50, #55) are closed as historical. The live roadmap is in the open issues created after 2026-10-05.

## Development

Use Python 3.11+ and `uv`:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv build
```

The research scripts live in `research/` and are covered by `ruff` only; see `research/README.md` for how they run.

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
