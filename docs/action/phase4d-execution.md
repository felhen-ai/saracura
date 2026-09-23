---
title: Phase 4D execution report
kind: report
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/phase4d-execution.md
globalRef: qmd://saracura/docs/action/phase4d-execution.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-23
sourceRefs: []
related:
  - AGENTS.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/action/specs/phase4d-experimental-universal-choice.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4D execution report

## Result

Phase 4D implemented the exact opt-in `laya-universal` runtime path described by the reviewed spec. It preserves the compiled state-reuse contract, accepts only `universal-choice@phase4d-laya.v1` as a dynamic workflow, returns uncalibrated abstained rankings with automation disabled, and loads the exact local candidate only after request gates pass.

The real local MPS smoke passed against the previously acquired Laya checkpoint after creating a private runtime snapshot containing only the five allowlisted files. The original research snapshot remained unchanged. The smoke used the synthetic public PT-BR example, processed 101 input tokens and four criteria, and completed its cold path in approximately 26.1 seconds. This establishes local execution compatibility only; it is not quality, calibration, licensing, or production evidence.

## SDD handoff

| Field | Value |
| --- | --- |
| orchestrator family | `gpt` |
| spec | `docs/action/specs/phase4d-experimental-universal-choice.md` |
| branch | `codex/phase4d-universal-choice` |
| executor provider | `gpt` |
| executor model | `gpt-5.6-terra`, high reasoning |
| phases | 4D.1 contracts; 4D.2 verified backend; 4D.3 CLI/docs/smoke |
| reviewer | `gpt-5.6-terra`, high reasoning, read-only |
| final reviewer verdict | `PASS` |
| provider fallback | none |

The required cross-family critic was dispatched to Claude before implementation but could not run because the account reported its weekly limit. The spec therefore received an explicitly degraded local architecture review, not a cross-model approval. Implementation still proceeded in bounded phases with independent read-only Terra reviews, active-session fixes, full QA, and a final `PASS` after the repository policy was reconciled with ADR 0001.

## Scope delivered

- separate compiled and universal backend protocols under one closed request API;
- exact dynamic workflow, locale, question, and cardinality gates;
- response invariants for ranking-only abstention and disabled automation;
- package-owned candidate ledger and descriptor-bound snapshot verification;
- delayed optional ML imports, tokenizer-before-model capacity gates, explicit CPU/MPS selection, no fallback, and synchronized single-model residency;
- fail-closed handling for NFC, special tokens, total token envelopes, snapshot identity, permissions, hardlinks, symlinks, extras, mutation, tensor keys, shapes, dtypes, and non-finite values;
- exclusive CLI argument forms, backend description, synthetic PT-BR example, and bilingual public documentation;
- narrow `AGENTS.md` policy exception for this exact backend and workflow only.

No provider, mailbox, model host, acquisition endpoint, remote fallback, calibration fit, training run, external action, deploy, or release occurred.

## Validation

The final active-session QA runs:

```bash
git diff --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python benchmarks/validate_default_environment.py
uv build
uv run python benchmarks/inspect_wheel.py <wheel> <sdist>
```

The real smoke uses the exact model revision `052592a15d198d9ad47da779604259b10b47b7aa`, checkpoint SHA-256 `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`, Apple MPS, disabled MPS fallback, and `examples/ptbr-universal-request.json`. The response contract is `uncalibrated`, `ranking_weights`, `abstained=true`, `calibration=null`, and `automation_allowed=false`.

## Rollback and next gate

Rollback is a Git revert of the Phase 4D merge; local snapshots remain operator-managed and are not deleted. The next separately reviewed increment is Phase 4E: read-only shadow e-mail triage with minimized retention and operator agree/correct feedback. No mailbox mutation is authorized.
