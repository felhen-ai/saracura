---
title: Phase 4E.3c Saracura-owned runtime registration
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4e3c-owned-runtime-registration.md
globalRef: qmd://saracura/docs/action/specs/phase4e3c-owned-runtime-registration.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-26
sourceRefs:
  - docs/action/phase4e-execution.md
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
  - docs/action/specs/phase4e3b-first-owned-checkpoint.md
related:
  - AGENTS.md
  - README.md
  - docs/README.pt-BR.md
  - src/saracura/cli.py
  - src/saracura/runtime/engine.py
  - src/saracura/runtime/workflows.py
  - src/saracura/universal/checkpoint.py
  - src/saracura/universal/rendering.py
  - benchmarks/inspect_wheel.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.3c Saracura-owned runtime registration

## Decision and task identity

`task_id=saracura-phase4e3c-owned-runtime-registration-v1`.

Register the first approved Saracura-owned universal checkpoint as the explicit local backend `saracura-universal`. The backend uses the verified multilingual MiniLM foundation encoder and the sealed Phase 4E.3B projection checkpoint, supports only the exact `universal-choice@phase4e-saracura-ranker.v1` workflow, and returns uncalibrated abstained rankings with automation disabled.

This increment performs no training, provider call, holdout access, calibration, model publication, mailbox access, remote inference or external action. It consumes the already sealed checkpoint only as a private operator artifact. Laya remains an independent external control and is neither loaded nor required by the Saracura backend.

## Approved artifact identity

The runtime candidate is exactly the Phase 4E.3B result recorded on `main`:

| Field | Exact value |
| --- | --- |
| model ID | `saracura/universal-ranker` |
| model revision | `phase4e-saracura-ranker.v1.f1d72c34cc535ddefbf7e24cf45d1be6e0aee40ba881228b4edd092d78640d5e` |
| checkpoint SHA-256 | `6914195d5526fb7b629eedaac7f33ae04f77eccd8448c75fed9104ccc57b728c` |
| training-manifest internal SHA-256 | `f1d72c34cc535ddefbf7e24cf45d1be6e0aee40ba881228b4edd092d78640d5e` |
| training-manifest file SHA-256 | `1daee9317425007a408646fdf07747fe5b5b59e6ef20104788bc577c659e1afb` |
| capsule-descriptor file SHA-256 | `44693fc40beb64c3c556b8f411c8f4f5e04088388ab24660b47340a09af03801` |
| capsule-descriptor internal SHA-256 | `036e6c6dc85d8b7d619de6e7f7fc943975a088ae5be1c27e5c786b57be9b7a64` |
| conformance-manifest file SHA-256 | `e2406ce5119f8261fd15224ceaf831c35e34004ffd0920b6d4fd0aca7196640a` |
| conformance-vectors SHA-256 | `7682148b18edc3058ce2038300234c243c973dc06bf2f2ba0b7c7913fa7a0bde` |
| architecture | `saracura-universal-ranker.v0` |
| foundation encoder | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| encoder revision | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| encoder snapshot-complete SHA-256 | `62827e84e66f4546f996b7ac92f01c0bf96e322e309618d69101f1925fe5b823` |
| training source commit | `cbff1989c2c3f802f4b4186c65f339466310296c` |
| disposition | `synthetic_only_research` |

The package may contain a small public candidate registry carrying these identities and hashes. It must not contain the checkpoint, encoder weights, conformance tensors, raw rows, holdout identities, private paths or any other private artifact. The runtime accepts an operator-selected filesystem location only when its complete bytes match this package-owned identity; the path itself does not select a model.

Checkpoint, capsule-descriptor file and training-manifest internal hashes come from the merged public Phase 4E.3B execution record. The remaining file/internal hashes, encoder snapshot-complete digest and full source commit above were extracted read-only from the same sealed live capsule by the existing training-time verifier on 2026-09-26; the source commit was independently resolved by Git. Before merge, live QA must independently re-read the sealed capsule, reproduce every registry value and confirm the source commit object exists. A registry value copied only from this spec is not sufficient evidence.

## SDD review state

Critique round 1 returned `NO-GO` because the initial draft carried an invalid full training commit and proved only projection-head conformance. This revision corrects the commit, records the provenance of registry-only hashes and adds end-to-end parity against the training pipeline below. Critique round 2 classified both blockers as resolved and returned `GO` with no blocker.

## Current execution path and code ownership

The implementation must preserve and extend the following reviewed path:

1. `src/saracura/cli.py:83-118` defines the closed CLI choices and optional backend arguments. `src/saracura/cli.py:211-271` parses and validates request-only constraints before backend construction, but currently hard-codes the Laya workflow and candidate. `src/saracura/cli.py:357-439` owns backend description and universal execution. This phase must make universal prevalidation backend-specific without weakening the existing Laya lane.
1. `src/saracura/runtime/engine.py:207-335` dispatches by `BackendCapabilities.execution_tier`. The universal branch validates a request, scores each question and already produces the required uncalibrated response. The engine contract remains unchanged and does not claim state-once encoding for universal backends.
1. `src/saracura/runtime/workflows.py:38-137` admits a dynamic workflow only when the selected universal backend advertises the exact `(id, revision)` pair. The Phase 4E revision constant exists, while the current special-case dispatcher admits only the Phase 4D Laya revision.
1. `src/saracura/backends/laya.py:361-592` is the installed external-control implementation. It demonstrates delayed optional imports, explicit CPU/MPS selection, no MPS fallback, a resident-model lifecycle and research-only universal scoring. Its candidate identity, snapshot, rendering, weights and workflow must remain separate from Saracura.
1. `src/saracura/backends/minilm.py:85-355` contains the existing verified MiniLM encoder construction, tokenizer reconstruction, mean pooling and device checks. Shared encoder mechanics may be extracted into a package-private helper only if the compiled MiniLM backend remains behavior-compatible and its tests continue to pass.
1. `src/saracura/verified_bytes.py:228-606` owns the descriptor-safe, no-follow, exact-file-set readers and the reviewed MiniLM snapshot reader. The new capsule boundary must reuse these primitives or an equivalently strict package-owned primitive; runtime code must not import checkout-only benchmark modules.
1. `src/saracura/universal/checkpoint.py:17-103` owns the closed nine-tensor Saracura projection schema. `src/saracura/universal/rendering.py:12-80` owns the exact renderer and token limits. `src/saracura/universal/tasks.py:13-78` owns the Phase 4E locales, domains and training-task limits. Their stale planned-only wording must be corrected without changing the trained ABI.
1. `benchmarks/saracura_universal_training.py:3727-3975` is the authoritative training-time verifier, but it depends on checkout-only policy and historical Git evidence and is excluded from the sdist. It is a design reference only. Installed inference must use a smaller self-contained verifier bound to the exact approved capsule registry.
1. `benchmarks/inspect_wheel.py:10-144`, `pyproject.toml:21-128` and `.github/workflows/ci.yml:66-127` own package purity, optional dependencies and CI isolation. They must prove that the default install remains lightweight and that no weight or private research artifact enters a distribution.

The intended execution sequence is: parse request bytes; validate API/schema/workflow/cardinality/domain/NFC/code-point/byte limits and backend-exclusive arguments; verify the sealed capsule without ML imports; derive and compare the immutable request model revision; verify the exact encoder snapshot; import optional ML libraries; construct the encoder and Saracura projection ranker; run device conformance; score the request; emit only research semantics; close the backend.

## Scope

### 1. Package-owned candidate registry

Add one closed package resource for the approved Saracura universal candidate. Its parser rejects duplicate JSON keys, unknown fields, aliases, alternate revisions, uppercase or malformed hashes and any candidate count other than one. It pins the table above plus the exact expected training-capsule filename set. The registry contains metadata only and is included byte-for-byte in wheel/sdist inspection.

The exact 14-file runtime capsule set is `accepted-packet-receipt.json`, `architecture.json`, `capsule-descriptor.json`, `checkpoint.safetensors`, `conformance-manifest.json`, `conformance-vectors.safetensors`, `dependency-versions.json`, `dev-selection.json`, `embedding-descriptor.json`, `epoch-ledger.json`, `holdout-release.json`, `holdout-report.json`, `pre-holdout-gate.json` and `training-manifest.json`. The descriptor ledger contains the other 13 names in sorted order. No `.npy`, split conformance file, renamed manifest or synthesized runtime capsule format is compatible.

The full request revision uses the internal canonical training-manifest digest, not the raw file digest: `phase4e-saracura-ranker.v1.<manifest_sha256>`. No `latest`, branch, tag, shortened digest or caller-defined model identity is accepted.

### 2. Installed capsule verifier

Add an installed, ML-free verifier for the explicit sealed training-capsule directory. It must:

- open one current-user `0700` directory through a no-follow descriptor walk;
- accept exactly the 14 Phase 4E final-capsule files and no sibling, link, hardlink, device, pipe or socket;
- read each current-user `0600` regular file once under a fixed per-file byte ceiling and prove directory/file identity did not change;
- require the capsule descriptor raw digest and internal digest fixed above, its exact 13-entry sorted ledger, and each listed size/digest to match the bytes already read;
- parse JSON with duplicate-key rejection, finite-number checks and canonical byte checks where the training format is canonical;
- require `phase4e-training-manifest.v2`, `synthetic_only=true`, `outcome=passed`, the exact manifest file/internal hashes, checkpoint hash, architecture, base encoder identity/revision/snapshot digest, renderer revision, source commit and public evidence bindings fixed by the approved capsule;
- validate the checkpoint safetensors only after optional ML import against the closed tensor names, shapes, FP32 dtype, finite values and bounded log scale in `universal/checkpoint.py`;
- validate the conformance manifest and tensor bytes, shapes, dtypes, masks, option counts and finite values before the model becomes available;
- retain only immutable in-memory bytes and public metadata after the verified descriptor closes, so later model construction never reopens a caller path.

The installed verifier must not run Git, inspect the working tree, read the Phase 4E private-root configuration, discover caches, import `benchmarks`, access holdout registries or require a source checkout. Exact approved hashes replace those training-time provenance dependencies at the installed boundary. Errors map to one public `BACKEND_UNAVAILABLE` envelope without paths, filenames, tensor values or private metadata.

The conformance artifact is exactly one safetensors mapping with `context_embeddings:[14,384]` FP32, `criterion_embeddings:[14,8,384]` FP32, `option_masks:[14,8]` uint8, `option_counts:[14]` int64 and `expected_logits:[14,8]` FP32. Its manifest schema is `phase4e-conformance-manifest.v1`; the ordered cells are `pt-BR:2` through `pt-BR:8`, then `en:2` through `en:8`. Runtime verification must match the training-time contract in `benchmarks/saracura_universal_training.py:3883-3963` rather than define a new wire format.

### 3. Explicit `saracura-universal` backend

Add `SaracuraUniversalBackend` as a `UniversalBackend` with:

- explicit `encoder_snapshot`, `training_capsule` and `device=cpu|mps` inputs;
- optional reviewed CPU thread count from 1 through 8, defaulting to 1 to match the sealed training/conformance environment;
- model reference equal to the approved table;
- execution boundary `explicit-verified-local-saracura-universal`;
- one-process resident encoder/ranker lifecycle with idempotent `prepare()` and `close()`;
- no network, remote fallback, telemetry, cache discovery, request-triggered download or Laya import;
- delayed imports of `torch`, `transformers`, `tokenizers` and `safetensors` until request-only and capsule-identity validation have passed;
- exact reconstruction of the reviewed MiniLM tokenizer and encoder from verified bytes, frozen/eval FP32 inference, attention-mask mean pooling in FP32, and the sealed Saracura projection/ranking math;
- explicit MPS availability and `PYTORCH_ENABLE_MPS_FALLBACK` checks; requested MPS must fail instead of executing any layer on CPU;
- one joint encoder batch per question containing the rendered context followed by its criteria, preserving criterion order and returning summed attention-mask token usage.

The installed torch ranker lives under `src/saracura/universal/` and is the single runtime source for the exact operation sequence `linear -> GELU -> LayerNorm -> L2 normalize -> cosine dot product -> exp(log_scale)`. It must not import the checkout-only training module. A later refactor may make future training import this installed math, but historical capsule verification remains bound to its recorded source commit.

This is a question-conditioned universal backend. It does not satisfy or claim the compiled tier's encode-state-once optimization: the state and instruction form one context per question. High-volume compiled specialization remains a separate tier and future optimization.

The backend advertises only `universal-choice@phase4e-saracura-ranker.v1`, locales `pt-BR` and `en`, the 12 Phase 4E domains, 1 through 10 questions and 2 through 8 criteria per question. The 8-option runtime limit follows the checkpoint's trained and held-out evidence even though the lower-level renderer can structurally frame 20 criteria. Unsupported locale, domain, revision or cardinality fails before encoder inference.

The CLI and backend apply the existing Phase 4E request limits before capsule access or tokenization: exact workflow revision, `pt-BR|en`, one of the 12 domains, at most 10 questions, at most 8 criteria, NFC everywhere, instruction at most 120 code points and 480 UTF-8 bytes, each criterion description at most 120 code points and 480 bytes, state with 1 through 64 top-level fields and canonical JSON at most 200 code points and 800 bytes. The verified tokenizer then enforces complete, non-truncated inputs: context at most 128 tokens and each criterion at most 96. Capacity failure returns `CAPACITY_EXCEEDED`; nothing is shortened.

Inference adds a label-free `render_choice` adapter taking only locale, domain, instruction, state and ordered criteria. Existing `render_task(UniversalTask)` delegates to that adapter and must remain byte-identical. Runtime code never fabricates `task_id`, `family_id` or `selected_criterion_id` merely to satisfy a training-row schema.

### 4. Device conformance and deterministic behavior

At `prepare()`, run the capsule's 14 raw-text-free locale-by-option-count conformance cells through the loaded projection ranker. CPU must reproduce the sealed valid logits exactly. MPS may use `rtol=1e-4` and `atol=1e-5`, but must preserve all valid option counts, finite logits and top-ranked positions; no masked value may enter a score. Conformance and decisions use the same configured CPU thread count, with 1 as the installed default. The backend documents its process-global torch thread effect and restores the prior thread count on `close()` when safe; it does not enable a process-global deterministic-algorithms mode solely for inference.

The conformance tensors are content-free derivatives of the consumed synthetic holdout. The installed verifier may hash them and the prepared ranker may evaluate them, but errors, `describe-backend`, benchmarks, tests and execution records must never emit tensor values, cell embeddings, expected logits or holdout-report contents.

The live QA pass first proves end-to-end CPU parity against the training pipeline without opening holdout data. For both public PT-BR and English smoke requests, it runs the read-only training path using the sealed renderer and verified snapshot (`_load_verified_minilm`, `render_task` and `_extract_bounded_embeddings`, with 128-token context and 96-token criterion limits) and runs the installed runtime against the same rendered strings and checkpoint. It requires identical unpadded token IDs and attention-mask extents for every rendered input, encoder embeddings and final logits within `rtol=1e-5` and `atol=1e-6`, and identical ordered rankings. The training path may tokenize contexts and criteria as separate padded batches while runtime uses one joint batch; only padding outside the attention mask may differ, and that batching difference is the declared reason for tolerance rather than exact tensor equality. The execution record stores the commands, dependency versions, tolerances and aggregate pass/fail result without paths, input text, tokens, embeddings or logits.

After that parity gate, live QA runs the same public requests twice on CPU and twice on MPS in separate backend lifecycles. Each device must be deterministic within its own repeated run, both devices must return the same ordered choice IDs, and cross-device logits must satisfy `rtol=1e-4` and `atol=1e-5`. Differences are reported; they are not required to be byte-identical across devices. If MPS is unavailable, the implementation may merge only with an explicit documented unavailable result, never a fallback result. The capsule's dependency-version manifest and the live package versions are reported as environment evidence; a version difference is permitted only when both projection conformance and the end-to-end parity gate pass.

### 5. CLI and description contract

Add `saracura-universal` to `decide` and `describe-backend` with the exclusive form:

```bash
saracura decide --backend saracura-universal \
  --request <request.json> \
  --encoder-snapshot <verified-minilm-snapshot> \
  --training-capsule <sealed-saracura-capsule> \
  --device cpu
```

It rejects `--calibration`, `--model-snapshot`, `--training-manifest`, `--checkpoint` and any mixed backend artifact. Invalid or unsupported request structure, including every Phase 4E locale/domain/cardinality/code-point/byte constraint, must fail before capsule access. A structurally valid Saracura request then verifies the capsule metadata and compares its derived revision with `request.model` before encoder snapshot access or optional ML import. The CLI must avoid double-reading the capsule by passing the immutable verified snapshot into backend construction; direct library construction performs the same verification internally. Cross-backend tests must prove that the Phase 4E capsule cannot enter `minilm-routing` and a human MiniLM capsule cannot enter `saracura-universal`, even though both commands use the explicit `--training-capsule` option.

`describe-backend` verifies and prepares the exact backend, emits no local path, and reports model identity, encoder identity/revision/snapshot digest, tokenizer and renderer revisions, architecture, device execution path, workflow limits, synthetic-only disposition, conformance result and the same safe response semantics used by `decide`.

The decision response remains exactly:

- `status=uncalibrated`;
- `score_semantics=ranking_weights`;
- `abstained=true`;
- `reason=uncalibrated_research`;
- `calibration=null`;
- `automation_allowed=false`.

Temperature-1 softmax values are API-compatible ranking weights, not confidence. No threshold, policy or downstream action may interpret them as calibrated probabilities.

### 6. Public example, documentation and packaging

Add one synthetic PT-BR request and one synthetic English request using the exact model and workflow revisions. Update `AGENTS.md`, both READMEs, universal module docstrings and the Phase 4E execution record to state that the Saracura backend is installed but remains opt-in, private-artifact, synthetic-only and research-only. Replace planned/unsupported wording only for the exact backend and revision; calibration, automation, model publication and broader dynamic workflows remain unsupported.

Reuse the `universal-local` optional dependency group; add no default dependency. Default import, `--help`, schema parsing and rejected request paths must remain free of optional ML imports. Update package inspection for the metadata-only registry and retain the blanket rejection of safetensors, model binaries, raw data, reports, capsules and private paths in wheel/sdist.

### 7. Latency evidence

Add a checkout-only, offline benchmark entry point or an equivalent deterministic QA command that keeps one verified backend resident, performs three warmups and at least twenty measured single-question decisions for each available device, and reports cold `prepare`, warm p50/p95, throughput, input shape and peak RSS without request text or private paths. It uses only the public synthetic examples and cannot alter the checkpoint or acceptance gates. Documentation states that `describe-backend` performs full preparation/conformance and therefore pays the same cold-start class of cost.

This phase records measurements but sets no speed marketing claim and no pass/fail latency threshold. Any comparison with Laya, Jev or TypeSafe remains Phase 4E.4 work.

## Tests and failure cases

1. Candidate-registry tests reject every changed identity or hash, duplicate key, extra field, alias and ambiguous candidate.
1. Registry-coupling tests prove that changing the package encoder registry or its snapshot receipt invalidates the fixed Saracura candidate until a separately reviewed candidate registry update binds the new bytes.
1. Capsule tests use an injected synthetic registry and create-only synthetic bytes to prove exact file-set, owner/mode, symlink/hardlink/FIFO, size, descriptor, manifest, checkpoint and conformance rejection. Production registry loading remains non-injectable through CLI arguments.
1. Backend tests cover PT-BR and English, every allowed domain, 2 and 8 choices, 9-choice rejection, 10 and 11 questions, NFC, byte/code-point limits, tokenizer token limits, order preservation, repeated descriptions, stable first-declared tie breaking, non-finite tensors/logits, invalid token usage, lifecycle and resource cleanup.
1. CLI tests prove backend-exclusive arguments, every request-only Phase 4E limit before artifact access, capsule-before-snapshot/ML order, cross-backend capsule rejection, exact revision matching, JSON-safe error redaction, close-on-success/failure and `describe-backend` output.
1. Workflow tests replace the planned Phase 4E rejection with exact Saracura capability admission while preserving crossed Laya/Saracura revision rejection before backend access.
1. Network tests block socket construction through verifier, prepare, conformance and decision paths.
1. Default-environment tests prove import/help/invalid-request paths do not import or require optional ML packages.
1. Packaging tests prove the wheel/sdist include only the new metadata registry and no weight, tensor, dataset, report, capsule, path, credential or executable binary.
1. Existing fixture, compiled MiniLM, Laya, calibration and training suites remain green and byte/behavior compatible outside the intentionally updated Phase 4E runtime contract.
1. One live private-artifact QA pass independently reproduces every candidate-registry value, runs the training-time verifier once read-only, then the installed verifier, training-pipeline-to-runtime CPU parity, CPU/MPS conformance, deterministic public smoke requests and latency measurement. It must not read or regenerate holdout data and must not modify the sealed capsule.

## Acceptance criteria

1. `saracura-universal` executes the real approved checkpoint locally from explicit verified artifacts and returns only research-safe rankings.
1. A modified or different capsule, checkpoint, manifest, conformance vector or encoder snapshot fails before inference.
1. Laya is not imported, loaded or required by any Saracura backend path.
1. No training code, Git history, configured private root or checkout-only benchmark module is required by an installed runtime.
1. CPU projection conformance passes exactly, and public-input tokenizer, encoder, pooling and final-logit output match the training pipeline within the declared CPU tolerance.
1. MPS projection and public smoke paths pass within the declared tolerance when available, with no fallback; decisions are deterministic per device and preserve the same ranked choice order across available devices.
1. The full validation suite, build and package inspection pass.
1. The repository and built distributions contain no model weights or private evidence.
1. Documentation makes no calibrated-confidence, production, representative PT-BR, speed-superiority or automation claim.

## Non-scope

- No retraining, new corpus, provider request, prompt/model change, hyperparameter adjustment, checkpoint averaging or reopened holdout.
- No calibration, temperature fitting, selective-risk threshold, confidence gate or `automation_allowed=true`.
- No public checkpoint upload, Hugging Face publication, Git LFS artifact or bundled model weight.
- No Laya/Jev/TypeSafe comparison or public performance claim.
- No mailbox, browser, AIOS, customer, Felhen-private or human-original data.
- No HTTP server, daemon, deployment, telemetry, remote fallback or model download.
- No boolean/ordinal workflow, arbitrary locale/domain, more than 8 choices or compiled high-volume optimization.
- No cleanup or deletion of the sealed private corpus, capsule, claims, terminal records or training evidence.

## Validation commands

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

The live private-artifact commands must use environment-held paths and emit only public-safe hashes and aggregate measurements. They must not print local paths, request contents, tensor values, secrets or private registry contents.
