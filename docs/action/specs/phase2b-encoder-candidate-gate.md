---
title: Phase 2B encoder candidate gate
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase2b-encoder-candidate-gate.md
globalRef: qmd://saracura/docs/action/specs/phase2b-encoder-candidate-gate.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-21
sourceRefs:
  - https://huggingface.co/jhu-clsp/mmBERT-base
  - https://huggingface.co/FacebookAI/xlm-roberta-base
  - https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
related:
  - docs/action/specs/phase2a-benchmark-harness.md
  - SECURITY.md
  - AGENTS.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 2B encoder candidate gate

## Context

Phase 2A established a reproducible decision-scaling harness using only the deterministic fixture backend. The next safe increment is an opt-in encoder research lane on the MacBook: freeze candidate provenance, acquire only reviewed files, load them without remote code or pickle weights, and measure encoder behavior without connecting an untrained encoder to the decision runtime.

This phase does not select a production model. It creates evidence for the later native architecture bake-off. Dataset ingestion, head training, calibration fitting, external baselines, publication, and AIOS integration remain separate gates.

## Outcome

An operator can explicitly install the optional research dependencies, acquire an eligible candidate at an immutable revision, verify every acquired byte against the repository manifest, and run a fixed PT-BR/English encoder probe on CPU or Apple MPS. Default installation and CI remain lightweight and offline.

## Candidate disposition

The versioned registry starts with three distinct roles:

1. `mmbert-base`: required architectural candidate, MIT, pinned to `jhu-clsp/mmBERT-base@c5955035435e2bf121cde7f3c8863ef52ff35d82`, but **blocked** because that upstream revision publishes `pytorch_model.bin` and no safetensors weights. This phase must not download or deserialize it.
2. `xlm-roberta-base`: established multilingual control, MIT, pinned to `FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, eligible for the probe through its safetensors artifact.
3. `multilingual-minilm-l12`: smaller speed/footprint control, Apache-2.0, pinned to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2@e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, eligible for the probe through its safetensors artifact.

The registry records source repository, immutable revision, declared license, role, disposition, reason, reviewed timestamp, exact allowlisted files, byte sizes, and SHA-256 digests. A mutable branch or tag is never an acquisition identity. Candidate disposition is an explicit closed enum; absence or an unknown value fails closed.

The license values below are upstream declarations recorded for provenance, not an independent legal opinion. Every eligible config and tokenizer config was reviewed without an `auto_map` entry.

| Candidate | model type | loader/tokenizer | Files | Total bytes |
| --- | --- | --- | --- | ---: |
| `mmbert-base` | `modernbert` | none; blocked | no allowlist; upstream snapshot exposes `pytorch_model.bin` and no `.safetensors` | 0 |
| `xlm-roberta-base` | `xlm-roberta` | `XLMRobertaModel` / `XLMRobertaTokenizerFast` | frozen below | 1,129,734,061 |
| `multilingual-minilm-l12` | `bert` | `BertModel` / `PreTrainedTokenizerFast` | frozen below | 484,793,579 |

`xlm-roberta-base` exact allowlist:

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 615 | `d66ed8cd4f2a93b358c245e50736fa389ed4f35c0bae7aad0b32abb20c62b579` |
| `model.safetensors` | 1,115,567,652 | `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb` |
| `sentencepiece.bpe.model` | 5,069,051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `tokenizer.json` | 9,096,718 | `a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c` |
| `tokenizer_config.json` | 25 | `994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce` |

`multilingual-minilm-l12` exact allowlist:

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 645 | `6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e` |
| `model.safetensors` | 470,641,600 | `eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b` |
| `sentencepiece.bpe.model` | 5,069,051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `special_tokens_map.json` | 239 | `378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d` |
| `tokenizer.json` | 9,081,518 | `2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8` |
| `tokenizer_config.json` | 526 | `5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b` |

## Scope

### 1. Optional dependency isolation

Add an `encoder-eval` optional dependency extra containing only the versions needed for safe local acquisition and inference. `uv sync --locked --dev` must continue to omit Torch, Transformers, Hugging Face Hub, model weights, tokenizers, and dataset loaders. The optional lane is installed explicitly with `uv sync --locked --dev --extra encoder-eval`.

The package wheel must not contain weights. The registry may ship as package data because it is small, declarative, and required for fail-closed validation.

### 2. Closed registry contract

Create strict Pydantic models for registry version `encoder-candidates.v1`. Reject unknown fields, mutable or malformed revisions, duplicate candidate IDs, duplicate file paths, absolute/traversal paths, non-positive file sizes, malformed digests, unsupported licenses, and eligible candidates without exactly one `.safetensors` weight file.

Parse registry JSON with duplicate-key rejection before Pydantic validation. Reject any allowlisted extension outside `.json`, `.model`, and `.safetensors`; reject `.bin`, pickle, Python, shared-library, symlink, device, socket, and other executable/special-file paths regardless of disposition. Blocked candidates may document upstream files but expose no downloadable allowlist. Code must never silently promote or rewrite a blocked entry. Every acquisition and probe result includes the SHA-256 of the exact bundled registry bytes.

### 3. Explicit acquisition boundary

Provide a research command that accepts only a candidate ID from the bundled registry and requires an explicit network flag. It downloads only the exact allowlisted files at the immutable revision into a candidate/revision-specific user cache under `platformdirs.user_cache_path("saracura") / "encoders"`. It must use a staging directory, validate file set, size, and SHA-256 before atomic promotion, and refuse to overwrite an existing immutable snapshot.

The command must never accept a repository ID, model URL, arbitrary revision, arbitrary file list, or request payload as acquisition input. It must not download blocked candidates. Errors and machine-readable output must not echo local paths, URLs containing credentials, environment values, or model input text.

Acquisition uses public anonymous HTTPS with no token, cookies, `.netrc`, proxy credentials, or Hugging Face credential environment. URLs are constructed only from the registry repository, revision, and file path. Redirect history and the final host must remain within `huggingface.co`, its subdomains, or `*.hf.co`. Each file streams into an exclusive regular file and aborts at `expected_size + 1`; free-space preflight requires the candidate total plus 128 MiB. The registry-provided SHA-256, not an HTTP ETag, is authoritative.

The concurrency primitive is an exclusive `mkdir` claim named from the candidate revision. Only its owner creates a UUID-named staging sibling. Files use mode `0600`, directories `0700`; every file and directory is fsynced. After exact-set staging verification, installation uses a two-phase, no-clobber commit: atomically create the final directory with `mkdir` and fail if anything already exists there; move the verified files into that owned `0700` directory; re-verify and fsync them; then create `snapshot.complete.json` with an exclusive regular-file create, fsync it, and fsync the final and parent directories. The marker contains candidate ID, revision, registry digest, and the verified file ledger and is the atomic commit point. Readers reject the final directory until that valid marker exists. This avoids a check-then-`rename` replacement window on macOS and never replaces a pre-existing final directory, including an empty one. Tests must inject destination creation immediately before the final `mkdir` and prove no clobber. Graceful failure removes only that invocation's staging, incomplete final, and claim after verifying ownership; a crash may leave them, and later acquisition fails closed and requires operator cleanup. A separately valid completed final snapshot may be verified and used without network even if a stale claim remains. Symlinks, hard links with link count other than one, and non-regular files fail verification.

If the final snapshot already exists, a verification-only pass is allowed and must perform no network request. Partial staging state or a snapshot without a valid completion marker must not be considered installed.

### 4. Safe local loading

Load exclusively from the verified snapshot directory with `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`, and fast tokenizers. Re-verify the exact file set, file metadata, completion marker, and content hashes before every load. Config and tokenizer config must have no `auto_map`; runtime `model_type`, architecture, hidden width, loader class, and tokenizer class must equal the registry. `tokenizer.is_fast` must be true. Loader dispatch is a closed mapping to `XLMRobertaConfig` + `XLMRobertaModel` + `XLMRobertaTokenizerFast` or `BertConfig` + `BertModel` + `PreTrainedTokenizerFast`; do not use `AutoModel`, `AutoTokenizer`, SentenceTransformers, or dynamic registration. Never fall back to a remote repository, `.bin`, pickle, plugin, or dynamic module.

The loader is research-only and is not registered as a Saracura decision backend in this phase. An encoder embedding is not a decision score and must not enter a calibration artifact.

### 5. Fixed encoder probe

Add a checkout-only probe with self-authored, immutable PT-BR and English inputs. The corpus is the following exact UTF-8 JSON line plus LF, stored as `benchmarks/fixtures/encoder-probe-corpus.v1.json`; its exact-byte SHA-256 is `f5d09a8b1d968579dfd31c656c855b83530c8dac1ffde866cdc380792e7f6b64`:

```json
{"schema_version":"encoder-probe-corpus.v1","cases":[{"id":"ptbr-support-refund","locale":"pt-BR","text":"A cliente informou uma cobrança duplicada e solicita o estorno de um dos pagamentos."},{"id":"en-support-refund","locale":"en","text":"The customer reported a duplicate charge and requests a refund for one payment."}]}
```

The order is fixed as written. One batch contains both cases. Tokenization is `padding=True`, `truncation=True`, `max_length=128`, and tensor output; a separate untimed check rejects the run if either untruncated sequence would exceed 128 tokens, so a valid result always reports `truncated=false`. Pooling is the attention-mask-weighted mean of `last_hidden_state` converted to float32, with no vector normalization. The embedding digest is SHA-256 over the pooled tensor in case order, copied to CPU as contiguous little-endian IEEE-754 float32 C-order bytes.

Each command invocation is a new process and probes exactly one candidate. It records three untimed warmups and exactly 20 measured samples. The measured boundary includes tokenization, transfer to the selected device, model forward, and pooling; it excludes snapshot verification, model/config/tokenizer loading, result validation, and file writing. MPS is synchronized immediately before and after every measured region. `fresh_process_load_ms` begins before config/tokenizer/model construction and ends after `eval()`, device transfer, and device synchronization; the report labels it as affected by the OS page cache, not a guaranteed cold start.

It measures:

- fresh-process load time as an explicitly named page-cache-sensitive observation;
- warm encode latency after fixed warmups, with raw samples plus nearest-rank p50/p95;
- tokenizer output lengths, parameter count, snapshot bytes, process RSS before load/after load/after probe, embedding width, deterministic embedding digest, device, dtype, and candidate provenance;
- sanitized hardware/software fields using the same privacy grammar as Phase 2A.

The probe supports `cpu` and `mps`; unsupported devices fail closed. It uses inference mode and `eval()`, fixes the maximum input length, detects truncation, synchronizes MPS around timed regions, and records no input text or token IDs. Each invocation probes exactly one candidate and emits schema `phase2b.v1` through immutable atomic JSON plus a Markdown report. Output filenames are derived from candidate ID and run ID, not user input.

`phase2b.v1` is a closed contract with: `schema_version`; UUID `run_id`; UTC timestamp; candidate ID/repository/revision/declared license/registry digest/verified file ledger; corpus ID/digest/case IDs/locales with no text or token IDs; protocol constants; sanitized OS family/release, architecture and machine class; Python, Saracura, Torch, Transformers and Tokenizers versions; device and float32 pooling dtype; `fresh_process_load_ms`; parameter count; snapshot bytes; the three RSS observations; two token lengths; hidden width; truncation flag; embedding digest; 20 non-negative finite raw samples, each paired with its embedding digest; and recomputed nearest-rank p50/p95. Summary/sample mismatches, provenance mismatches, counts outside the frozen constants, unknown fields, or unsafe environment strings fail validation.

RSS is the current resident set in bytes from `psutil.Process().memory_info().rss`, not peak RSS. It is sampled (1) immediately after snapshot verification and optional imports but before config/tokenizer/model construction, (2) immediately after `eval()`, device transfer, synchronization, and the load timer, and (3) immediately after the final measured synchronization but before result construction. Parameter count is `sum(parameter.numel() for parameter in model.parameters())`. Token length is the integer sum of each case's attention-mask row from the untimed validation tokenization. Every one of the 20 measured samples records the digest produced inside that sample; all 20 must equal the top-level digest. The two clean executions required for a reported candidate/device/protocol must have the same digest. CPU and MPS digests are not required to equal each other.

These numbers are encoder-only observations. The report must state that they do not measure decision quality, trained-head performance, calibration, end-to-end Saracura latency, or readiness for automation.

### 6. Commands, errors, and validation surface

The only entrypoint is `python -m benchmarks.encoder_gate` with these closed commands:

```text
python -m benchmarks.encoder_gate acquire --candidate <registry-id> --allow-network
python -m benchmarks.encoder_gate verify --candidate <registry-id>
python -m benchmarks.encoder_gate probe --candidate <registry-id> --device cpu|mps --output-dir <directory>
python -m benchmarks.encoder_gate validate-registry
```

There is no repository, revision, URL, token, file, loader, model path, corpus, warmup, sample-count, batch-size, maximum-length, pooling, or dtype flag. Candidate IDs are bounded registry identifiers. `--output-dir` controls only result placement and is never used for loading. Errors map to bounded public codes/messages and never echo paths, URLs, environment values, exception text, or inputs. JSON and Markdown filenames are `<candidate-id>-<device>-<run-id>` and are immutable atomic siblings.

Validation commands are:

```bash
uv sync --locked --dev
uv run python benchmarks/validate_default_environment.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.encoder_gate validate-registry
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl
git diff --check
```

`validate_default_environment.py` fails if Torch, Transformers, Hugging Face Hub, SentenceTransformers, or a dataset loader is importable in the clean default virtual environment. `inspect_wheel.py` rejects weights, `.bin`, pickle, Python bytecode, shared libraries, and anything outside the expected package/code/registry set. The optional MacBook lane runs only after the default proof in a distinct virtual environment or after recreating `.venv`.

### 7. Tests and documentation

Default CI tests the registry, duplicate-key rejection, acquisition policy, host and redirect allowlist, anonymous client configuration, streaming size cap, free-space guard, hashing, path and file-type safety, claim/race/crash behavior, atomic promotion, optional-dependency error, closed loader dispatch, output contract, percentile semantics, input redaction, and deterministic probe orchestration using fakes. It performs no network request and imports no optional ML runtime.

Document the explicit opt-in flow in English and PT-BR. Update the security boundary to distinguish operator-triggered acquisition from forbidden request-triggered downloads. Document why mmBERT is blocked rather than weakening the safetensors-only invariant.

## Out of scope

- Downloading or converting mmBERT pickle weights.
- Choosing a winning encoder.
- Connecting an encoder to the decision runtime.
- Training or fitting any head, adapter, tokenizer, or model.
- Dataset discovery beyond a later, reviewed data/license matrix.
- Quality, calibration, accuracy, or multilingual coverage claims.
- External-service baselines, model publication, package release, server, AIOS adapter, or paid compute.

## Acceptance criteria

1. Default `uv sync --locked --dev` remains free of Torch, Transformers, Hugging Face Hub, model files, and dataset loaders.
2. The registry validates the three pinned candidates and marks mmBERT blocked because its reviewed revision lacks safetensors.
3. Blocked, unknown, mutable, or tampered candidates fail before any network or model load.
4. Acquisition is explicit, allowlisted, revision-pinned, hash-verified, atomic, and idempotent without overwriting immutable snapshots.
5. Loading is local-only, safetensors-only, remote-code-disabled, and preceded by full snapshot verification.
6. The fixed PT-BR/English probe records no source text or token IDs and produces strict `phase2b.v1` JSON plus a bounded Markdown report.
7. CPU and MPS paths have deterministic orchestration tests; actual MacBook evidence requires two clean runs per eligible candidate/device combination that is reported.
8. Reports label the evidence encoder-only and research-only and make no decision-quality or calibration claim.
9. `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src tests benchmarks`, `uv run pytest -q`, manifest validation, secret/privacy scan, `uv build`, and `git diff --check` pass.
10. Default CI performs no model or dataset download.
11. The clean default environment proves optional ML modules absent, and wheel inspection proves no weight or executable binary is packaged.

## Implementation sequence

1. Add the registry models, bundled registry, and offline validation tests.
2. Add isolated optional dependencies and the explicit acquisition/verification boundary.
3. Add the safe local loader behind optional imports.
4. Add strict probe contracts, runner, immutable output, and report generation.
5. Add documentation and CI coverage.
6. Run the full default suite.
7. On the MacBook, explicitly install the extra, acquire eligible candidates, run two CPU probes, and evaluate MPS only if the CPU path and model support are clean.

## Rollback

This phase has no deployment, training job, package publication, or remote mutation beyond read-only model acquisition into a local cache. Rollback is a Git revert plus deletion of the candidate-specific local cache by the operator. Existing snapshots are never rewritten by the program.

## MacBook evidence

The approved local lane was executed on a 24 GB Apple Silicon MacBook using the
frozen PT-BR/English batch, Python 3.14.5, Torch 2.14.0, Transformers 4.57.6,
and Tokenizers 0.22.2. Each row is a separate process with three warmups and 20
measured iterations. Raw JSON and Markdown remain ignored local artifacts by
design; the table records the sanitized result needed for this gate.

| Candidate | Device | Run | Load ms | p50 ms | p95 ms | RSS after probe | Embedding SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `multilingual-minilm-l12` | CPU | 1 | 790.194 | 7.100 | 7.325 | 841,203,712 | `b091561ff0e6e3852c6e4c5ca32f930d5331db25a3a0e8b6ea62d050fe417c6a` |
| `multilingual-minilm-l12` | CPU | 2 | 2,378.302 | 7.537 | 9.108 | 892,715,008 | `b091561ff0e6e3852c6e4c5ca32f930d5331db25a3a0e8b6ea62d050fe417c6a` |
| `multilingual-minilm-l12` | MPS | 1 | 1,219.187 | 5.571 | 6.108 | 888,602,624 | `7f2a68102e4635d34393a927935bc76d4cb5f83fa973835534ae5fe8ea562869` |
| `multilingual-minilm-l12` | MPS | 2 | 1,057.585 | 6.842 | 10.816 | 882,606,080 | `7f2a68102e4635d34393a927935bc76d4cb5f83fa973835534ae5fe8ea562869` |
| `xlm-roberta-base` | CPU | 1 | 765.644 | 13.674 | 13.864 | 1,130,807,296 | `c4dd0039cfa4b6f208e1fb6ec6d8397408062f37b848802f54fa7d7c83dfeeff` |
| `xlm-roberta-base` | CPU | 2 | 1,376.807 | 39.712 | 52.534 | 1,128,480,768 | `c4dd0039cfa4b6f208e1fb6ec6d8397408062f37b848802f54fa7d7c83dfeeff` |
| `xlm-roberta-base` | MPS | 1 | 1,715.824 | 7.527 | 8.826 | 728,416,256 | `edbb6d9457e5308b80c1233ff3ba5b15126cb9124494a74f3355053ac0b9346c` |
| `xlm-roberta-base` | MPS | 2 | 1,073.068 | 9.706 | 10.452 | 891,174,912 | `edbb6d9457e5308b80c1233ff3ba5b15126cb9124494a74f3355053ac0b9346c` |

All eight runs reported token lengths 20 and 18, `truncated=false`, and the
expected parameter counts (117,653,760 for MiniLM; 278,043,648 for XLM-R).
The two runs for every candidate/device pair produced the same embedding digest.

This evidence keeps both eligible candidates in the later architecture bake-off.
MiniLM showed the lower CPU latency and smaller footprint, while XLM-R on MPS
was competitive but CPU latency varied materially between clean processes.
MPS was not consistently faster for the smaller model. These observations do
not select a production encoder and do not measure decision quality, trained
head performance, calibration, end-to-end Saracura latency, or high-question
throughput.

## Next gate

After Phase 2B, create and approve the concrete PT-BR data/license/LGPD matrix before any ingestion or training. Only then may a separate Phase 3 increment implement the known-workflow shared encoder and small `choice` head under reproducible train/calibration/test splits.
