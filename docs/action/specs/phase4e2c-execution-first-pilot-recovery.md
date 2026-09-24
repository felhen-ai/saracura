---
title: Phase 4E.2c execution-first pilot recovery
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4e2c-execution-first-pilot-recovery.md
globalRef: qmd://saracura/docs/action/specs/phase4e2c-execution-first-pilot-recovery.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
related:
  - benchmarks/saracura_universal_pilot.py
  - benchmarks/saracura_universal_corpus.py
  - benchmarks/phase4e_pipeline.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.2c execution-first pilot recovery

## Decision

Replace the pilot's hard financial gates and single-error terminal behavior with report-only cost telemetry and bounded technical recovery. Run a fresh attempt over the same answer-blind 140-task evaluation after implementation and review. Financial spend has no ceiling; the operator is informed whenever run-local provider-reported spend crosses another USD 10 and once at completion.

This increment removes process that blocked execution. It does not relax credential handling, reviewer blindness, privacy checks, task identity, immutable local evidence, or the acceptance thresholds that measure whether the generated data is useful.

## Verified context

- `benchmarks/saracura_universal_pilot.py:124-153` turns the historical spend baseline into an authorization gate.
- `benchmarks/saracura_universal_pilot.py:157-241` freezes USD 0.50, USD 1.00, USD 1.50 and USD 17.00 caps plus zero retries in the pilot policy.
- `benchmarks/saracura_universal_pilot.py:525-554` rejects a plan whose conservative estimate exceeds those caps.
- `benchmarks/saracura_universal_pilot.py:921-944` rejects execution based on conservative cumulative headroom instead of merely validating research-history integrity.
- `benchmarks/saracura_universal_pilot.py:1184-1332` stops the entire 70-pair run after the first batch exception and marks all remaining tasks unresolved.
- `benchmarks/saracura_universal_pilot.py:1335-1459` places author generation, reviewer parsing, review resolution and persistence under one exception boundary; its fallback always says `author_response_failure`.
- `benchmarks/phase4e_pipeline.py:236-260` rejects an otherwise usable structured response whenever `reasoning` or `reasoning_details` is non-empty.
- `benchmarks/saracura_universal_corpus.py:168-214` defines zero-retry, hard-cap ledger policies.
- `benchmarks/saracura_universal_corpus.py:2395-2550` enforces stage and total caps before and after every request.
- `benchmarks/saracura_universal_corpus.py:2900-3105` forbids non-zero retry policy and derives the same reservation ID for repeated identical request bodies.
- The sealed v1 pilot ledger shows the third author call and one reviewer call settled before the run stopped. Therefore the interruption occurred after provider settlement and cannot correctly be attributed generically to the author.

## Scope

### 1. Versioned report-only pilot policy

Add a v2 pilot policy and v2 pilot ledger identity. Preserve v1 manifests and the ability to validate historical v1 ledgers. The v2 policy keeps the same seed and task identities because no gold answer is sent to the author or blind reviewer; it explicitly records that this is an operational recovery attempt of the v1 evaluation.

The v2 policy contains:

- `cost_mode=report_only`;
- `cost_report_interval_usd=10.00`;
- `maximum_attempts_per_request=5`;
- the existing provider prices for estimation and reporting, but no stage, total or cumulative authorization ceiling;
- unchanged task counts, reviewer blindness, quality thresholds and evaluation-only authorization.

The plan retains exact cost estimates as telemetry. Estimates never authorize, block or stop execution.

### 2. Report-only ledger behavior

Extend `LedgerPolicy` with explicit enforcement mode. Existing v1/v2 historical policies remain hard-cap policies. The new pilot policy records reservations, provider cost, conservative debit and provider journal exactly as today but never raises because a financial amount crossed a stage or total cap.

Repeated identical calls under a retry-enabled policy receive deterministic attempt-specific reservation IDs. Existing zero-retry policy reservation IDs remain byte-compatible.

### 3. Recoverable response handling

`_response_content` ignores provider reasoning metadata and parses only `message.content`; reasoning is neither interpreted nor persisted.

For v2 pilot requests, retry only settled responses that fail local content/schema validation. Transport-uncertain calls remain non-retriable because their provider outcome is unknown. Each retry is a new charged journal entry. At most five attempts are made for one author or reviewer request.

After attempts are exhausted:

- an author failure resolves that pair as `author_response_failure` and execution continues with the next pair;
- a reviewer failure resolves only the affected generated row as `reviewer_response_failure`; the sibling can still be evaluated;
- diagnostic counts record stage, attempts and whether a retry recovered;
- no raw provider response, reasoning, prompt, credential or rejected generated text is persisted.

An unexpected persistence or invariant failure still stops the run because continuing could corrupt evidence.

### 4. Cost reporting

The CLI emits one concise progress line when run-local provider-reported cost crosses USD 10, USD 20, USD 30 and so on. It emits a final provider-reported total even when no milestone was crossed. These messages are telemetry only.

The historical catalog remains integrity-checked and append-only. Its conservative debit is reported but is not compared with an authorization ceiling.

### 5. Real recovery run

Create new `pilot-plan-v2`, `pilot-work-v2` and `pilot-report-v2` paths. Do not overwrite v1 evidence. Run with the existing OpenRouter credential through `op-felhen-projects` and the verified local MiniLM tokenizer snapshot.

The result may be `PASS`, `FAIL` or `INCONCLUSIVE` based on the complete evidence. No corpus or training runs in this phase. A `PASS` authorizes writing the next execution increment; it does not silently start it.

## Non-scope

- Changing the 140 evaluation tasks or their answer positions.
- Sending answers, pair identity, siblings, split or semantic targets to the reviewer.
- Weakening privacy or quality acceptance thresholds.
- Persisting raw provider payloads.
- Corpus generation, embedding extraction, checkpoint training, calibration, runtime registration or publication.
- Adding any new financial ceiling.

## Acceptance criteria

1. V1 ledgers and evidence still validate; v2 uses report-only cost accounting.
2. Reasoning metadata does not invalidate valid structured `content` and is not retained.
3. Retry-enabled identical requests have distinct deterministic reservations and all settled calls have journal bindings.
4. A malformed settled author or reviewer response is retried and can recover.
5. Exhausted author and reviewer retries have correct stage-specific reasons and do not stop unrelated pairs.
6. Transport-uncertain calls are not automatically retried.
7. No financial amount can stop a v2 pilot; USD 10 milestones and final actual cost are emitted.
8. The pilot still resolves exactly 140 task identities or reports the exact unresolved identities as `INCONCLUSIVE`.
9. Reviewer blindness, privacy gates, immutable create-only artifacts and secret handling remain covered by tests.
10. The real v2 report, ledger hashes, actual cost and decision are appended to `docs/action/phase4e-execution.md`.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src benchmarks
uv run pytest -q tests/test_phase4e_pilot.py tests/test_phase4e_pipeline.py tests/test_phase4e_corpus.py tests/test_docs.py
uv run pytest -q
uv build
git diff --check
```

## Rollout and rollback

The rollout is additive: v1 artifacts remain untouched and the real run uses v2 directories. Before network use, run the complete offline suite and a credential-absent preflight that reaches only the expected missing-credential boundary. If implementation or review fails, discard only uncommitted v2 code. If the live v2 run becomes transport-uncertain, preserve its journal, report `INCONCLUSIVE` and do not replay that uncertain request automatically.

## Risks

- Retrying malformed settled responses increases cost. This is intentional and visible through actual-cost telemetry.
- A provider may repeatedly violate the schema. Five attempts prevent an infinite loop; exhaustion is stage-specific and the remaining pairs continue.
- Ignoring reasoning metadata could accidentally persist it if the parser returns the full message. The parser must return only decoded `content`, and tests must search work/report artifacts for a marker placed only in reasoning.
- Continuing after a recoverable response failure could mix partial lineage. Tests must prove every settled reservation has a journal entry and every persisted outcome points to the correct final author/reviewer attempt.

## Review resolution

The cross-model critic was operationally unavailable and made no changes. Three read-only reviewer rounds then identified three consequential defects introduced by this increment: retries initially included settled HTTP errors; a crash after settlement but before local resolution could replay paid task IDs; and locally rejected author fields did not enter the retry path. All three were corrected and receive adversarial tests.

The reviewer also repeatedly raised the pre-existing Phase 4E workflow contract in `src/saracura/runtime/workflows.py`. That contract comes from commit `2a07d6e`, is deliberately exercised as a planned-contract path in `tests/test_phase4e.py`, and has no installed backend or CLI capability. It is outside this diff and this recovery increment. No unresolved blocker remains that is caused or exposed by the current change, so implementation proceeds to final QA and the isolated v2 run.
