---
title: Phase 4D experimental universal Choice runtime
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase4d-experimental-universal-choice.md
globalRef: qmd://saracura/docs/action/specs/phase4d-experimental-universal-choice.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-23
sourceRefs:
  - https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af
  - https://huggingface.co/convaiinnovations/laya-multilingual/tree/052592a15d198d9ad47da779604259b10b47b7aa
related:
  - AGENTS.md
  - README.md
  - docs/README.pt-BR.md
  - docs/decisions/0001-two-tier-decision-architecture.md
  - docs/action/specs/phase4c2-local-candidate-adapters.md
  - docs/action/specs/phase4c4-architecture-decision.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4D experimental universal Choice runtime

## Context

ADR 0001 selected a two-tier research architecture: a universal tier accepts new typed Choice schemas, while the compiled tier remains an optional optimization for stable high-volume work. Phase 4C.2 proved that the exact Laya multilingual option-marker candidate can load and execute locally on the target MacBook, but that benchmark code is checkout-only, permits explicitly reported truncation for measurement, and does not expose a runtime backend. Phase 4D turns that candidate into an opt-in installed runtime path without converting systems evidence into a quality, calibration, licensing, or production claim.

The downstream internal Portuguese e-mail triage pilot is deferred to Phase 4F, after the independently trained Phase 4E checkpoint and comparison gates. It needs dynamic labels and instructions without a task-specific training run, but it must begin in shadow mode: Saracura may rank choices and collect the operator's agreement or correction, while deletion, archive, movement, reply, forwarding, or any other mailbox mutation remains outside this phase and the pilot's initial authority.

## Objective

Add one experimental local `laya-universal` backend to the existing typed `v1alpha1` Choice API. It must accept a reviewed dynamic-workflow reference, run only against an explicit verified local snapshot, return uncalibrated ranking output with automation disabled, fail closed instead of truncating, keep all ML dependencies optional, and preserve the compiled runtime's one-state-encoding invariant.

## Non-goals

This phase does not train or fine-tune weights, fit a temperature map, claim calibrated confidence, approve automation, read a mailbox, add an e-mail connector, mutate an e-mail, add a remote provider or fallback, use TypeSafe or OpenRouter, download weights in response to a decision, discover a Hugging Face cache, execute upstream Laya code, copy benchmark modules into the installed package, publish weights, resolve the candidate's tokenizer or training-data licensing provenance, select a production model, create a release, or deploy a service.

## Architecture and contracts

### Two explicit backend tiers

`BackendCapabilities` gains `execution_tier: Literal["compiled", "universal"]`. Every existing fixture and MiniLM backend declares `compiled`. The compiled `Backend` protocol remains `encode_state` once per request followed by `score_choice` for each question. A separate `UniversalBackend` protocol exposes a joint question-conditioned scoring operation and may tokenize state, instruction, and ordered criteria together for each question. `DecisionEngine` accepts either protocol and dispatches only by the closed capability value; it must not identify tiers by concrete class, attribute probing, or import availability.

The universal operation returns `ScoredChoice` with an `input_tokens` count for that question. It receives the already validated request, the selected question, and the canonical serialized state bytes. The engine still serializes and byte-bounds state once, but it does not claim that a universal model embeds that state once. Compiled backends continue to encode the independent state exactly once and reuse it for every question; regression tests preserve that invariant.

### Dynamic workflow gate

The only dynamic workflow reference in this phase is:

| Field | Exact value |
| --- | --- |
| `workflow.id` | `universal-choice` |
| `workflow.revision` | `phase4d-laya.v1` |
| locales | `pt-BR`, `en` |
| domain | any valid existing `Identifier` |
| question type | `choice` only |
| question count | 1 through 10 |
| criteria per question | 2 through 20 |
| criterion order | semantic and preserved |

`WorkflowRegistry.validate` receives the declared execution tier. Known immutable workflows retain their exact current validation for both tiers. The dynamic reference is accepted only for a `universal` backend and only under the table above. A compiled backend presented with that reference returns `WORKFLOW_UNSUPPORTED`. A universal backend presented with an unregistered dynamic revision also returns `WORKFLOW_UNSUPPORTED`. This is a scoped runtime contract, not an open-ended bypass of workflow validation.

The public request model retains its current global maximum of 50 questions so existing compiled workflows remain compatible. The dynamic workflow's lower maximum of 10 is enforced before any optional ML import, snapshot inspection, tokenizer work, or device allocation.

### Research-only response semantics

`Answer.status` adds `uncalibrated`. `Answer.calibration` becomes nullable, with a model-level invariant: `uncalibrated` requires `calibration=null`; `calibrated` and `fixture_only` require the existing reference. `Answer.score_semantics` is `calibrated_confidence` for `calibrated`, `fixture_distribution` for `fixture_only`, and `ranking_weights` for `uncalibrated`.

`DecisionResponse` adds `automation_allowed: Literal[False]`, and the engine sets it to false for every research response, including existing compiled responses. A universal answer preserves the top-ranked criterion ID in `value`, raw marker logits in `raw_scores`, and temperature-1 softmax normalization in the existing `probabilities` field for API compatibility. Those normalized values are ranking weights, not calibrated probabilities. It sets `status=uncalibrated`, `score_semantics=ranking_weights`, `abstained=true`, `reason=uncalibrated_research`, and `calibration=null`. No risk threshold or action gate may consume them as confidence.

The universal path accepts no calibration artifact. The compiled path retains its current required compatibility validation. CLI argument validation rejects `--calibration` with `laya-universal` and requires it for `fixture` and `minilm-routing`.

## Exact local candidate contract

The installed package contains a closed, reviewed `laya-candidate.v1.json` ledger with only the exact runtime facts required for verification and construction. It is independently validated and does not import or parse the benchmark registry at runtime.

| Identity | Exact value |
| --- | --- |
| candidate | `laya-multilingual` |
| model ID | `convaiinnovations/laya-multilingual` |
| model revision | `052592a15d198d9ad47da779604259b10b47b7aa` |
| upstream source revision | `573e5b62696ba441230cd6be71d593331b5d23af` |
| acquisition contract digest | `be1356e4d2a771830e44b899d1ddd8fc2f30b36943d6becb321afdfecc5b638d` |
| conformance vector digest | `14f2eba88e45d9b9138c3c4ceabc5413a5bdb069f838a6480b4071a9d94fa3b7` |
| runtime disposition | `research_only_unresolved_provenance` |

The ledger allowlists exactly these files and no others:

| Relative path | Bytes | SHA-256 |
| --- | ---: | --- |
| `encoder/config.json` | 1,938 | `83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4` |
| `rl_agent_config.json` | 472 | `25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6` |
| `tokenizer/tokenizer_config.json` | 502 | `424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7` |
| `tokenizer/tokenizer.json` | 34,363,188 | `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f` |
| `model.safetensors` | 643,835,514 | `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204` |

The public `ModelReference` uses ID `convaiinnovations/laya-multilingual`, the exact model revision above, and the `model.safetensors` SHA-256 as `checkpoint_sha256`.

### Snapshot verification

The caller supplies `--model-snapshot <directory>`. The runtime never infers a cache root, reads model-related environment variables, calls a hub API, follows a URL, or searches a home directory. A package-owned verifier opens the caller path and every nested component descriptor-relatively, requiring current-UID ownership, mode 0700 for directories, mode 0600 for files, regular files, one hard link, no symlink traversal, no unexpected directory or file, exact byte length, and exact SHA-256. It rejects special files, permission drift, ownership drift, hardlinks, symlinks, extras, missing files, digest mismatch, size mismatch, and descriptor identity changes before, during, or after hashing/loading.

The verifier returns a descriptor-bound receipt. Metadata, tokenizer, and Safetensors readers may read only through that receipt. The receipt rechecks live descriptor identity around each read and remains open for the backend lifetime. The implementation may extract a package-owned general verifier shared by the installed backend and checkout-only benchmark lane, but installed code must never import `benchmarks`, and the benchmark suite must keep its existing security cases passing.

### Loader and execution isolation

The optional dependency group remains `universal-local`; it is not added to default dependencies. Importing `saracura`, request parsing, schema validation, workflow validation, help, and invalid CLI invocations must not import `torch`, `transformers`, `tokenizers`, `safetensors`, `psutil`, or `platformdirs`.

The backend directly constructs the reviewed ModernBERT encoder and Laya decision head from closed configuration. It rejects `auto_map`, `trust_remote_code`, unknown loader families, changed structural dimensions, changed special tokens, changed source tensor keysets, shapes, or dtype counts, and any non-F16 parameter except the exact F32 temperature tensor. The stored upstream temperature tensor is checked for shape and dtype but is not used as a calibration authority. No `from_pretrained`, `snapshot_download`, dynamic import, cache discovery, `trust_remote_code`, remote code, or upstream package execution is permitted.

Execution device is explicitly `cpu` or `mps`; no automatic device selection is allowed. MPS requires availability and `PYTORCH_ENABLE_MPS_FALLBACK` absent or exactly `0`; CPU fallback is forbidden. The model is built and run in FP32, resident-model count is limited to one per process, CPU thread count is bounded to 1 through 8, and local resource preflight retains the reviewed Phase 4C.2 minimum free-memory and free-disk checks. A backend construction failure maps to `BACKEND_UNAVAILABLE` without exposing local paths, exception strings, environment values, or tensor names in the public envelope.

## Input rendering and fail-closed capacity

Universal state text uses the existing RFC 8785 and NFC canonical state representation. The Laya subject is the NFC string `locale=<locale>;domain=<domain>`; the body is the UTF-8 decoding of `canonical_json_bytes(request.state)`. The renderer preserves declared criterion order and uses the reviewed option-marker framing `choice question: <instruction>`, one mask marker per criterion, length-framed subject/body, and the reviewed token structural IDs.

Input strings and object keys must already be NFC at the runtime boundary; unlike the compiled serializer, the universal renderer does not silently normalize user-visible content. It rejects any tokenizer special token occurring in the subject, state JSON text, instruction, criterion ID, or criterion description. It also rejects non-finite scores, duplicate or missing result IDs, and any mismatch between score keys and declared criterion IDs.

The following limits are checked before model execution:

| Limit | Value |
| --- | ---: |
| serialized state bytes | 1,000,000 maximum, preserving the existing global limit |
| universal questions | 10 maximum |
| criteria per question | 20 maximum |
| criterion description source characters | 4,096 maximum from the request contract |
| total Laya input tokens | 1,024 maximum |
| Laya decision-head prefix tokens | 256 maximum |
| microbatch size | 1 question in Phase 4D |

The benchmark renderer's truncation behavior is not permitted in the installed runtime. Each complete state, instruction, and criterion description must fit the reviewed 1,024/256-token envelope. Any condition that would produce `supported_with_state_truncation`, `supported_with_choice_truncation`, `supported_with_state_and_choice_truncation`, `unsupported_input`, or `unsupported_capacity` fails with `CAPACITY_EXCEEDED` before inference. No prefix, state, question, criterion, choice, or batch is silently dropped or shortened.

Criterion order is semantic for this dynamic workflow. The result mapping must preserve criterion IDs even when descriptions are identical. The backend returns a single score for each declared criterion in the same order; the engine validates exact key equality and finite values before normalization.

## CLI and public examples

`saracura decide` adds `--backend laya-universal`, `--model-snapshot`, and the existing explicit `--device`. It accepts exactly one of these argument shapes:

- `fixture`: request plus calibration, with no model paths or device;
- `minilm-routing`: request, calibration, its existing explicit artifacts, and device;
- `laya-universal`: request, model snapshot, and device, with no calibration or MiniLM artifacts.

The CLI parses and validates the request, model alias, dynamic workflow, question count, serialized state byte limit, NFC policy, and obvious string/special-token-independent capacity constraints before importing optional ML modules or opening the snapshot. Token capacity is then checked after the verified tokenizer loads but before model inference. `describe-backend` adds the same Laya snapshot/device path and reports exact model identity, execution tier, capacity, uncalibrated ranking semantics, research-only provenance disposition, and dynamic workflow reference.

Add a compact PT-BR universal request example using synthetic e-mail-like content and labels such as `action_required`, `billing_or_accounting`, `marketing_or_newsletter`, and `low_value_or_spam`. It must contain no real address, message, company correspondence, personal data, secret, or mailbox identifier. English and PT-BR README sections explain the opt-in command, the uncalibrated/abstained response, optional dependency, explicit local snapshot requirement, and prohibition on automation.

Reconcile `AGENTS.md` with ADR 0001 and this reviewed increment by replacing its earlier blanket compiled-only runtime restriction with one narrow exception for the exact Phase 4D backend and workflow. The policy must keep every other question-conditioned candidate checkout-only and bind the exception to opt-in local execution, no truncation, uncalibrated abstention, and no automation. This update does not authorize another universal backend or product path.

## Implementation phases

### Phase 4D.1: tier and response contracts

Implement the closed execution-tier field, separate universal protocol, dynamic-workflow policy, response semantics, branching engine path, error code, and deterministic fake-universal tests. Preserve all compiled invariants and current compiled CLI behavior. No Laya model code, optional dependency import, or snapshot read occurs in this phase.

### Phase 4D.2: verified Laya backend

Implement the package-owned candidate ledger, descriptor-bound snapshot verifier, closed tokenizer/model loader, fail-closed no-truncation renderer, Laya backend, and malicious-fixture/unit tests. Reuse or extract reviewed primitives where safe, but the installed package must remain independent of `benchmarks`.

### Phase 4D.3: CLI, example, documentation, and real local smoke

Wire the opt-in CLI and backend description, add the synthetic PT-BR example and bilingual documentation, then run one real local smoke against the already acquired immutable Laya snapshot if it is present and verifies. The smoke uses no network or provider, records no input or weight artifact in Git, and is reported as execution compatibility only. Absence of the local snapshot blocks only the real smoke, not unit/contract validation or merge, and must be reported explicitly rather than triggering acquisition.

## Acceptance criteria

1. Existing compiled backends declare `compiled`, still encode state once per request, require compatible calibration, preserve immutable workflow validation, and pass all existing tests.
2. Only a backend declaring `universal` may accept the exact dynamic workflow revision, and all request/capacity validation that does not require a tokenizer precedes optional ML import, snapshot inspection, or device work.
3. Universal responses are explicitly uncalibrated, abstained, ranking-only, calibration-null, and automation-disabled; model validators reject inconsistent response combinations.
4. The Laya backend uses only the exact package ledger, explicit verified snapshot, direct closed construction, explicit CPU/MPS device, FP32 execution, and the reviewed source key/shape/dtype contract.
5. Runtime execution has no network, cache discovery, environment-derived model path, upstream package execution, remote fallback, request-triggered download, silent truncation, or fallback device.
6. The verifier rejects unexpected paths, symlinks, hardlinks, wrong modes, wrong owner when testable, special files, wrong size/digest, and before/after identity mutation; verified readers remain descriptor-bound.
7. Complete state, instruction, and all criteria either fit unchanged or fail with `CAPACITY_EXCEEDED` before inference. Tests cover state overflow, prefix overflow, choice overflow, special-token injection, non-NFC text/key, unsupported locale, question count, and criteria count.
8. Default installation and invalid/help CLI paths do not import optional ML packages. Wheel and sdist contain the small candidate ledger and runtime modules but no benchmark package, model, cache, report, private input, or generated artifact.
9. A deterministic fake universal backend proves dynamic PT-BR and English Choice schemas, declared criterion-order preservation, exact score-key validation, finite-score validation, usage accounting, and response semantics without ML dependencies.
10. The bilingual documentation and synthetic example make the research-only, uncalibrated, no-automation boundary visible and identify e-mail triage only as a future Phase 4F shadow-mode pilot.
11. If the existing local Laya snapshot is available, a no-network CPU or MPS smoke returns one structurally valid universal response. The result supports only local execution compatibility and is not committed as quality evidence.
12. Repository policy and both public READMEs no longer claim that every installed backend is compiled or that all dynamic Choice labels are absent; they document only the exact reviewed Phase 4D exception without widening the product boundary.

## Validation

```bash
git diff --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python benchmarks/validate_default_environment.py
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
```

Phase-specific tests may be selected during implementation, but final QA runs the full list. Final QA also searches the installed runtime for forbidden hub/cache/network APIs, confirms the package boundary, verifies no `.artifacts`, model, tokenizer, report, mailbox content, secret, or generated benchmark file is tracked, and checks that `git diff origin/main...HEAD` contains only Phase 4D code, tests, public synthetic examples, documentation, and dependency-lock changes attributable to the already declared optional group.

## Rollout and rollback

Rollout is a normal merge to `main`. The backend is opt-in and inert unless a caller installs the optional group and supplies both `--backend laya-universal` and an explicit verified snapshot. There is no service deploy, mailbox connection, provider call, model acquisition, or release in this phase.

Rollback is a Git revert of the Phase 4D merge. Existing immutable local snapshots remain outside Git and operator-managed; rollback does not delete them. Existing compiled behavior remains available throughout because the universal path is additive and selected explicitly.

## Next gate: Phase 4E Saracura-owned universal checkpoint

The next reviewed increment is the planned Phase 4E Saracura-owned universal
checkpoint and blind Laya control comparison. Phase 4E.1 contains only
offline deterministic contracts and explicitly does not call a provider,
acquire a model, train a checkpoint, or register `saracura-universal`. Until
Phase 4E.3 seals a real synthetic-only checkpoint that passes its holdout gate,
`universal-choice@phase4e-saracura-ranker.v1` and `saracura-universal` remain
planned and unsupported.

A read-only internal e-mail shadow pilot is Phase 4F, after the checkpoint and
comparison gates. It must minimize retained content, separate message access
from Saracura scoring, show category and abstention without mutating the
mailbox, capture the operator's agree/correct decision with provenance, and
produce disjoint fit/evaluation inputs. Archive, delete, move, reply, forward,
send, unsubscribe, or any other external action requires a later calibrated
risk policy and separate authorization. Corrections are private first-party
evidence only under a new reviewed policy; they are not public training data
and never serve as evaluation after fitting.

## Risks

- **Ranking weights may be mistaken for confidence:** keep `uncalibrated`, `ranking_weights`, `abstained`, `calibration=null`, and `automation_allowed=false` together in the validated contract and documentation.
- **Dynamic workflow may become an unbounded schema bypass:** accept one immutable workflow reference with closed locale, question, cardinality, and execution-tier rules.
- **Installed runtime may weaken benchmark acquisition safeguards:** retain descriptor-bound verification and exact bytes, and require all malicious-fixture tests across both lanes.
- **No-truncation can reject realistic long e-mails:** expose the capacity error and measure its frequency in the shadow pilot; do not silently shorten content in order to improve apparent success.
- **Large local model can make the pilot feel slow or memory-heavy:** keep the backend opt-in, report current systems limits honestly, and preserve the compiled tier as the future high-volume path.
- **Research-only candidate provenance remains unresolved:** keep the runtime disposition in the model description, avoid release/production claims, and revisit provenance before any production or redistributed-weight plan.
- **Mailbox pilot can create privacy or action risk:** Phase 4D contains no connector. A read-only, minimized-retention pilot is deferred to Phase 4F after the Phase 4E checkpoint and comparison gates.
