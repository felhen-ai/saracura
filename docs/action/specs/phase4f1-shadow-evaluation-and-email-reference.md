---
title: Phase 4F.1 generic shadow evaluation and e-mail reference workflow
kind: specification
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4f1-shadow-evaluation-and-email-reference.md
globalRef: qmd://saracura/docs/action/specs/phase4f1-shadow-evaluation-and-email-reference.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-27
sourceRefs:
  - github:felhen-ai/saracura#31
related:
  - AGENTS.md
  - README.md
  - docs/README.pt-BR.md
  - docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md
  - docs/action/phase4e-comparison-result.md
  - docs/decisions/0001-two-tier-decision-architecture.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 4F.1 generic shadow evaluation and e-mail reference workflow

## Objective

Add the first reusable shadow-evaluation primitive around Saracura's existing
universal decision runtime. The primitive must let an external adapter stream
private items through one resident local model, return content-free decision
records, collect content-free operator labels, and calculate descriptive
agreement. A synthetic PT-BR e-mail policy is the first reference workflow, not
a product-specific pack or a mailbox connector.

This phase starts the read-only Phase 4F pilot released by the completed Phase
4E.4 comparison. It does not connect to Gmail or another provider, mutate a
mailbox, train or calibrate a model, persist message content, or authorize an
automated action.

## Verified context

- `src/saracura/contracts/models.py` lines 13-145 defines closed, frozen request
  and response models. `DecisionResponse.automation_allowed` is the literal
  `False`; universal answers are uncalibrated, abstained ranking weights.
- `src/saracura/contracts/models.py` lines 152-220 parses one request with
  duplicate-key rejection, stable public errors, and `extra="forbid"`.
- `src/saracura/runtime/engine.py` lines 221-336 executes a universal request,
  retains only ranking semantics, and always emits
  `automation_allowed=false`.
- `src/saracura/runtime/workflows.py` lines 40-148 validates dynamic
  `universal-choice` requests for `pt-BR` and `en`, with backend-owned workflow
  revision and capacity limits.
- `src/saracura/backends/saracura_universal.py` lines 536-595 applies the
  artifact-free locale, domain, cardinality, state, instruction, criterion, and
  NFC gates before model work. Lines 618-815 keep one prepared backend resident
  until `close()`.
- `src/saracura/cli.py` lines 122-163 owns the current command surface; lines
  604-641 execute one Saracura request only after request and capsule gates;
  lines 675-713 own top-level dispatch and public error rendering.
- `examples/ptbr-universal-request.json` is synthetic and already demonstrates
  e-mail triage without mailbox access, but it is tied to the Phase 4D Laya
  revision and is not a batch or feedback contract.
- `docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md` lines
  238-240 releases Phase 4F only as a separate read-only e-mail shadow pilot.
  It expressly forbids delete, archive, move, reply, forward, mark-read, and
  training on private mail without another reviewed policy and explicit
  authorization.
- `AGENTS.md` keeps Saracura standalone and local-first. Optional model
  dependencies may not enter the default import/help path, and request-triggered
  downloads, telemetry, remote fallback, and arbitrary model paths remain
  forbidden.

## Architectural decision

Phase 4F.1 is generic at its core and concrete only in its example:

1. A versioned `ShadowPolicy` contains the model revision, locale, domain,
   universal workflow revision, one instruction, and two through eight ordered
   choices. It also contains the exact ordered state keys accepted by that
   policy. It is data, not executable configuration.
2. A streamed `ShadowItem` contains only an opaque caller-created `item_ref` and
   the state passed to the existing universal renderer. The public reference
   e-mail adapter contract uses exactly `subject` and `preview`; full message
   bodies, attachments, addresses, provider IDs, thread IDs, labels, and mailbox
   metadata are outside this phase.
3. `ShadowRunner` constructs an ordinary closed `DecisionRequest` per item and
   calls the existing `DecisionEngine`. It does not fork scoring, rendering,
   workflow, capacity, or model-identity logic.
4. One context-managed runner owns one backend for the whole batch so the model
   is prepared once and closed once. The batch is bounded to 500 items, 4,096
   bytes per JSON line, and 2,048,000 input bytes in total.
5. A `ShadowDecisionRecord` contains the opaque reference, policy digest, model
   reference, suggested label, ordered ranking weights, uncalibrated/abstained
   semantics, and `automation_allowed=false`. It contains no input state,
   instruction, criterion description, raw model score, timing, provider ID, or
   mailbox action.
6. A separate content-free `ShadowFeedbackRecord` records an operator label or
   `skipped`. A deterministic evaluator joins decisions and feedback by exact
   `item_ref` and emits counts, coverage, agreement, and a closed confusion
   matrix. It never reads the original item state.

The primitives belong under `src/saracura/shadow/`. The CLI is a thin adapter;
library callers can use the same closed models and runner without subprocesses.
The first command is `saracura shadow-decide`; the second is
`saracura shadow-evaluate`.

## Public contracts

### Policy

`ShadowPolicy` is a frozen closed model with exactly:

- `schema_version = "saracura-shadow-policy.v1"`;
- `id` and immutable `revision` using the repository identifier/revision rules;
- `model`, `locale`, `domain`, and `workflow` copied into every generated
  `DecisionRequest`;
- `state_keys`: an ordered tuple of 1 through 16 unique identifiers; every
  declared state value is an NFC string in this schema version;
- `question_id`, `instruction`, and an ordered tuple of two through eight
  existing `ChoiceCriterion` objects.

Its binding is the SHA-256 of canonical NFC-normalized RFC 8785-compatible JSON
bytes using the repository serialization path. Duplicate keys, non-NFC input,
unknown fields, aliases such as `latest`, unsupported locales/domains/workflow
revisions, and label collisions fail before backend construction.
The item state key set must equal `state_keys`; extra keys, missing keys, and
non-string values fail without echoing a received key or value.

The policy also enforces the installed Saracura limits before backend
construction: instruction and criterion description are each at most 120
codepoints and 480 UTF-8 bytes. Token-dependent limits remain part of the real
backend validation: rendered context is at most 128 tokens and each rendered
criterion at most 96 tokens. The reference policy must pass both the byte and
codepoint tests in CI and the token gate in the operator smoke.

### Stream input

Each JSONL line is one closed `ShadowItem`:

- `schema_version = "saracura-shadow-item.v1"`;
- `item_ref`: 1-128 lowercase identifier characters, caller-created and opaque;
- `state`: a non-empty JSON object consumed only in memory.

The e-mail reference policy requires the state key set to be exactly
`{"subject", "preview"}` and both values to be NFC strings. Saracura does not
truncate, summarize, strip replies, or select message bodies. A private adapter
must choose a minimized provider preview before this boundary; an over-capacity
item fails closed instead of being silently shortened.

The canonical state is limited to 200 codepoints and 800 UTF-8 bytes, including
keys and JSON framing. Before backend construction, the command reads the
bounded batch and runs every assembled request through
`validate_saracura_request_structure`. Token-dependent capacity is checked only
after the verified tokenizer is available. Any line failure is rendered with a
zero-based `line_index` in public error `details`, while the path is normalized
to the item or `/state` and never repeats a caller-provided state key.

`item_ref` must not be an e-mail address or a raw provider message/thread ID.
The public contract cannot prove provenance, so the docs require a random or
keyed local reference and tests reject obvious addresses, path-like values, and
duplicates inside one batch. Pure hexadecimal provider IDs cannot be
distinguished mechanically from safe opaque references; preventing those is an
explicit Phase 4F.2 adapter obligation.

### Decision output

Each output line is one closed `ShadowDecisionRecord` with exactly:

- `schema_version = "saracura-shadow-decision.v1"`;
- `item_ref`, `policy_sha256`, and the existing `ModelReference`;
- `suggested_label` and ordered `(label, ranking_weight)` entries;
- literal `status="uncalibrated"`, `score_semantics="ranking_weights"`,
  `abstained=true`, `reason="uncalibrated_research"`, and
  `automation_allowed=false`.

The runner derives this from the existing `DecisionResponse`, validates that all
policy labels appear exactly once, orders descending weights with policy order as
the deterministic tie-break, and discards state, descriptions, raw scores,
timing, and token usage. Ranking weights are not confidence.

### Feedback and summary

`ShadowFeedbackRecord` contains exactly schema version, `item_ref`,
`policy_sha256`, `disposition` (`labeled` or `skipped`), and `operator_label`.
A labeled record requires one policy label; a skipped record requires
`operator_label=null`. Duplicate references, a mismatched policy digest, and
labels outside the bound policy fail closed.

`ShadowEvaluationSummary` contains the policy digest, prediction count, feedback
count, labeled count, skipped count, unreviewed-prediction count,
`feedback_coverage`, agreement count/rate, and one closed matrix containing every
policy label on both axes. Feedback whose reference has no prediction fails
closed rather than becoming an ambiguous orphan counter. Predictions without
feedback remain `unreviewed`. `feedback_coverage` is
`(labeled + skipped) / prediction_count`; `agreement_rate` is
`agreement_count / labeled`, or `null` when labeled is zero. The matrix includes
only labeled records. No threshold is interpreted as approval, and the summary
is not a calibration artifact.

## CLI behavior

### `shadow-decide`

The command requires the same explicit `--encoder-snapshot`,
`--training-capsule`, and `--device` arguments as `saracura-universal`, plus a
public `--policy` path. Items are read only from stdin as UTF-8 JSONL; there is
no mailbox/network connector and no raw-input output path. The command buffers
only validated decision records and writes canonical JSONL to stdout after the
entire batch succeeds. It validates the entire bounded structural batch before
constructing the backend. On any invalid line or runtime error it emits the
normal public error envelope with the zero-based line index to stderr, emits no
decision stdout, closes the backend, and exits 2.

Stdin is read from `sys.stdin.buffer` and decoded as strict UTF-8. Shadow-layer
errors that may originate in private input are raised without a chained cause or
context containing validation input. Tests inspect `__cause__` and
`__context__`, not only rendered error text.

### `shadow-evaluate`

The command takes explicit local paths to decision and feedback JSONL plus the
same policy file, verifies the policy digest and closed records, and writes one
canonical summary JSON object to stdout. It performs no model import or model
work. Inputs are content-free by contract; the command still rejects symlinks,
files above 8 MiB, duplicate JSON keys, duplicate decision references, duplicate
feedback references, unknown fields, and non-NFC text. Every decision must use
the policy model revision, and all decisions in one file must carry one identical
`ModelReference`.

Both commands reject calibration arguments, Laya-only paths, network access,
and undeclared backend choices. Default import, `saracura --help`, and invalid
argument paths remain free of optional ML imports.

The public `ShadowRunner` is a context manager with an idempotent `close()`.
Calling it before entering or after close fails deterministically. It requests
`include_timing=false`; timing is neither computed for the shadow contract nor
projected. The runner receives a backend factory and constructs and owns exactly
one backend instance. The CLI verifies the capsule and checks
`policy.model == candidate.model_revision` before invoking that factory. A
second open runner in the same process fails with `BACKEND_UNAVAILABLE` through
the existing resident-model guard.

## Reference e-mail policy

Add a synthetic `examples/ptbr-email-shadow-policy.json` bound to the installed
Phase 4E Saracura model and workflow. It contains seven editable decision labels
for the pilot:

1. `action_required`;
2. `financial_or_accounting`;
3. `legal_or_security`;
4. `project_or_operations`;
5. `marketing_or_newsletter`;
6. `low_value_or_spam`;
7. `manual_review`.

The descriptions must be mutually distinguishable, written in natural PT-BR,
and explicitly place ambiguity in `manual_review`. The example input contains
only fictional messages. These labels are a reference policy that users may
replace through data; they are not hard-coded into the runner and are not a
claim that Saracura is optimized or calibrated for e-mail.

The domain is exactly `email_triage`, one of the closed `UNIVERSAL_DOMAINS`; the
hyphenated Phase 4D Laya example is not copied. Fictional input and feedback
JSONL examples remain repository/documentation files and are excluded from the
wheel and sdist by the existing artifact gates. The JSON policy remains outside
the wheel; it may remain in the sdist like the repository's existing JSON
examples.

## Implementation phases

### 4F.1A — contracts and deterministic evaluator

- Add the closed policy, item, decision, feedback, and summary models.
- Add canonical policy binding, decision projection, feedback join, matrix, and
  invariants without importing optional ML packages.
- Add exhaustive unit tests with deterministic fixture responses.

### 4F.1B — resident local runner and CLI

- Add `ShadowRunner` over the existing `DecisionEngine`.
- Add `shadow-decide` and `shadow-evaluate` dispatch, input bounds, all-or-none
  stdout, close behavior, and stable public errors.
- Update `.github/workflows/ci.yml` explicitly so `tests/test_shadow.py` runs in
  the applicable Python 3.11, 3.12, and 3.13 matrices rather than only the
  default job.
- Exercise the runner with the fake backend in default CI and with synthetic ML
  tensors in the existing `universal-local` optional job. CI has no private
  model host, checkpoint, capsule, or operator cache and must not pretend to
  execute the real sealed backend.
- After CI-equivalent tests pass, run a local operator smoke outside CI with the
  private verified encoder/capsule and fictional repository input. A private
  receipt binds source commit, policy digest, output digest, item count, one
  runner/backend lifecycle, zero errors, and the absence of canary text in
  stdout/stderr. It also records the reference policy's context token count with
  empty state values so Phase 4F.2 can size previews from measured headroom. The
  public execution record may publish only those aggregate bindings, never
  artifact paths or message content.

### 4F.1C — public reference and documentation

- Add the PT-BR policy and fictional input/feedback examples.
- Update both READMEs and `AGENTS.md` with identical authority boundaries.
- Keep mailbox integration as Phase 4F.2 outside this public increment.

## Non-scope

- Gmail, Outlook, IMAP, browser automation, OAuth, API keys, mailbox reads, and
  provider-specific IDs.
- Delete, archive, move, label, mark read/unread, reply, forward, unsubscribe, or
  any other mailbox mutation.
- Background monitoring, UI overlays, notifications, queues, HTTP serving, or
  agent tool registration.
- Persistence of subject, preview, message body, sender, recipients, attachment,
  or headers in decisions, feedback, summaries, logs, telemetry, or tests.
- Training, fine-tuning, retrieval, prompt optimization, calibration,
  confidence thresholds, or reuse of feedback as fit/evaluation data.
- Public or repository storage of internal pilot decisions or corrections.
- Laya control execution in the first implementation increment.

## Acceptance criteria

1. Every new model is frozen and closed; duplicate keys, extras, invalid NFC,
   invalid identifiers, aliases, unknown labels, and malformed disposition
   combinations fail with deterministic public errors.
2. The policy digest is canonical and stable; changing policy id/revision,
   `state_keys`, question id, criterion id or order, description, instruction,
   locale, domain, workflow, or model changes it.
3. The runner uses the existing `DecisionRequest`, workflow registry,
   `DecisionEngine`, and Saracura backend. It does not duplicate rendering or
   scoring logic and prepares one backend at most once for a successful batch.
4. A 100-item fake-backend batch preserves input order and opaque references,
   produces 100 decisions, and constructs/prepares/closes one backend once. A
   synthetic-ML optional test proves one heavy-load initialization. The local
   operator smoke proves the sealed runtime handles multiple fictional items in
   one runner/backend lifecycle.
5. Every decision remains uncalibrated, abstained, and
   `automation_allowed=false`; ranking weights are never named confidence.
6. Decision, feedback, summary, stdout, stderr, and exception tests prove that
   subject, preview, instruction, criterion descriptions, and raw scores do not
   leave the decision boundary or survive as exception cause/context. Tests
   reject references containing `@`, `/`, or duplicates; a hexadecimal-looking
   reference stays accepted with an explicit test documenting that provider-ID
   provenance remains a Phase 4F.2 obligation. Shadow wrapping also removes
   content-bearing causes/contexts from errors returned by request construction,
   engine, and backend paths.
7. Any invalid line, duplicate reference, capacity failure, model error, or
   interrupted batch emits zero decision stdout, closes the backend, and exits
   nonzero. A `KeyboardInterrupt` regression covers the `BaseException` path.
8. Feedback evaluation is deterministic and tests exact feedback coverage,
   agreement, skipped, unreviewed, orphan rejection, zero-labeled behavior, and
   every confusion-matrix cell. Decision and feedback files reject duplicate
   references, and feedback is bound to the exact policy digest. Evaluation
   rejects mixed `ModelReference` values and any model revision different from
   the policy. It cannot create a calibration or training artifact.
9. `shadow-evaluate` imports no optional ML module. Default package import,
   `saracura --help`, invalid-input paths, and the ordinary test suite remain
   offline and optional-ML-free.
10. The PT-BR policy and examples contain fictional data only, pass repository
    privacy/secret gates, use `state_keys=["subject", "preview"]`, fit byte,
    codepoint, and measured token limits, and are replaceable without code
    changes.
11. The wheel contains the generic Python shadow primitives but no JSONL,
    repository example, internal mail, pilot output, model snapshot, checkpoint,
    capsule, or private adapter. The existing `.jsonl` prohibition is not
    relaxed. The JSON reference policy stays outside the wheel but may remain in
    the sdist; fictional JSONL stays outside both archives.
12. English and PT-BR documentation state the same boundary: read-only shadow
    classification, operator correction, no mailbox action, no confidence or
    automation claim, and no feedback reuse without a later policy.

## Validation

```bash
uv run pytest -q tests/test_shadow.py tests/test_cli.py tests/test_docs.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
gitleaks detect --source . --no-git --redact --exit-code 1
```

The optional CI matrix must also run the fake batch contract on Python 3.11,
3.12, and 3.13 and synthetic-ML residency tests through the existing
`universal-local` dependency boundary. The sealed-model smoke is an explicit
local operator step after CI-equivalent validation because CI intentionally has
no private artifacts. No live mailbox or network smoke belongs in the public
repository.

## Phase 4F.2 gate

Only after 4F.1 passes review and QA may a private integration increment connect
one explicitly selected Google account in read-only mode. That spec must define
account identity, least-privilege scopes, preview selection, HMAC reference
creation, retention, operator UI, correction capture, revocation, deletion, and
an audit proving zero mailbox mutations. It may consume the public shadow API;
it must not add Google/Felhen coupling to the Saracura core.

## Rollback

Rollback is a Git revert of the 4F.1 merge. No migration or server deployment is
introduced. The public commands and models disappear; externally managed model
artifacts and private adapter state are untouched. Because this phase writes no
mailbox data and performs no mailbox action, rollback does not require mailbox
repair.

## Risks and mitigations

- **Private content leaks through output or error text:** project only closed,
  content-free records; buffer stdout until the batch succeeds; redact all
  caught exceptions to existing public errors; normalize state-error paths; test
  canary strings across every output surface.
- **A ranking is mistaken for confidence:** keep the existing uncalibrated,
  abstained semantics and explicit naming; no threshold or action field exists.
- **A reference workflow becomes a restrictive product pack:** keep policy and
  labels as versioned input data; the runner is domain-agnostic and the e-mail
  file is only one example.
- **Silent truncation changes meaning:** accept only caller-selected previews
  and fail closed on capacity; never truncate inside Saracura.
- **Feedback contaminates later evaluation:** emit content-free corrections but
  prohibit training/calibration reuse until a new provenance and split policy.
- **Batch processing reloads the model or loses ordering:** one runner owns one
  backend, preserves input order, and has explicit 100-item residency tests plus
  one sealed-runtime fictional smoke.
- **The public core becomes tied to a provider:** connectors and provider IDs
  remain out of scope; Phase 4F.2 is a private adapter against this generic API.

## Critique record

Round 1 returned `NO-GO` with three blockers. Repository inspection confirmed
all three: the `universal-local` CI job has synthetic fixtures and no private
capsule (`.github/workflows/ci.yml` around lines 85-102); installed state and
renderer limits are materially below the first draft's line limit
(`saracura_universal.py` around lines 79-87 and `rendering.py` around lines
15-97); and packaging intentionally excludes JSONL while the wheel allowlists
only Python and named resources (`pyproject.toml` around lines 64-90 and
`benchmarks/inspect_wheel.py` around lines 10-87). The spec now separates CI
proof from the local sealed smoke, publishes capacity and line-index behavior,
and keeps examples outside the wheel without relaxing artifact gates.

Round 2 returned `NO-GO` with one blocker. Inspection confirmed that
`DecisionRequest.state` is a generic dictionary and
`validate_saracura_request_structure` validates capacity but not application
keys. `state_keys` is now closed policy data, enters the digest, and is enforced
before backend construction. The round's related low-cost findings were also
incorporated: feedback binds the policy digest; shadow exceptions discard
content-bearing causes; evaluator files have explicit bounds and duplicate
checks; CI file ownership is explicit; and archive wording matches the existing
wheel/sdist rules.

Round 3 returned `GO` with no blockers. Its low-cost findings were incorporated
where they made the implementation contract more falsifiable: the runner owns a
factory-created backend after capsule/model preflight; evaluation rejects mixed
models; the resident-model limitation and interruption behavior are explicit;
the private smoke measures reference-policy token headroom; and shadow wrapping
removes content-bearing causes from engine/backend errors as well as local model
validation errors.
