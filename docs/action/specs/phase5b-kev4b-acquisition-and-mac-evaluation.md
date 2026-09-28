---
title: Phase 5B Kev-4B acquisition and Mac evaluation
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase5b-kev4b-acquisition-and-mac-evaluation.md
globalRef: qmd://saracura/docs/action/specs/phase5b-kev4b-acquisition-and-mac-evaluation.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-27
sourceRefs:
  - https://huggingface.co/jaredpalmer/kev-4b/tree/139fdd94f1b6a6ad80cc15e08fcb99cac885a101
  - https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b
  - https://github.com/jaredpalmer/kev/tree/9c41005b2180347c3c646dfc9e50c4428483ec6b
related:
  - docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md
  - docs/decisions/0002-open-model-runtime-and-readiness.md
  - benchmarks/manifests/phase5-open-model-candidates.v1.json
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 5B Kev-4B acquisition and Mac evaluation

## Managed CUDA amendment

The first executable systems lane moved from the memory-constrained MacBook to
a managed RTX 5090 host after the immutable acquisition completed. The service
ran out of process under a non-renewable, clinically preemptible lease. A live
Q=50 preemption test restored the protected workload within 105.3 seconds, and
a preliminary bilingual PT-BR/English matrix completed 240 measured requests
with a 721.687 ms aggregate p95.

This is preliminary systems evidence, published in
`benchmarks/results/phase5b-kev4b-managed-cuda-preliminary.json`. It does not
satisfy the full Phase 5B disposition because the reduced managed-host runner
did not collect the frozen memory and oversize/state-truncation evidence. It
does not establish language quality, calibration, Gate A completion,
production readiness or automation authorization. The immutable acquisition
may advance to `reviewed_acquisition`; broader claims remain blocked on the
formal public protocol.

## Outcome

Saracura will acquire and evaluate one immutable Kev-4B release on the local Apple Silicon development envelope without adding model dependencies, weights, upstream source, or request-triggered downloads to the installed package. The result is a reproducible systems report and a bounded candidate disposition, not a production backend or a quality claim.

Phase 5B may establish the pinned-acquisition and local-systems portions of Gate A. It cannot mark Gate A complete because the public bilingual development-quality lane belongs to Phase 5C.

## Reviewed upstream identities

- Kev model repository: `jaredpalmer/kev-4b` at immutable Hugging Face revision `139fdd94f1b6a6ad80cc15e08fcb99cac885a101`.
- Kev serving source: `jaredpalmer/kev` at immutable Git revision `9c41005b2180347c3c646dfc9e50c4428483ec6b`.
- Training-source provenance declared by the checkpoint: Git revision `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`.
- Base model: `Qwen/Qwen3.5-4B-Base` at immutable revision `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Upstream declares Apache-2.0 for Kev code, adapter/head, and the Qwen base. Phase 5B records that review as artifact-specific; it does not approve every upstream training dataset for redistribution or a future Saracura-owned checkpoint.
- The adapter snapshot is approximately 160 MB. The pinned base snapshot includes approximately 9.32 GB of safetensors weights. These observations are acquisition planning data, not installed package contents.

No branch name, tag, `main`, mutable URL, unverified mirror, or abbreviated revision is accepted by the acquisition contract.

## Execution boundary

Kev runs as an untrusted, opt-in research service in a separate Python 3.12 environment. Saracura communicates only with a fixed loopback endpoint using the public TypeSafe-style `POST /v1/systemone` contract. The Saracura process does not import Kev, PyTorch, Transformers, PEFT, MLX, or upstream Python modules.

The operator performs three explicit steps:

1. `prepare`: acquire the pinned Kev source, model snapshot and base snapshot into a private cache, verify the complete ledger and build a locked Python 3.12 environment;
2. `serve`: start the pinned service on `127.0.0.1` with Hugging Face and Transformers offline flags, a per-run bearer token, and no remote fallback;
3. `evaluate`: validate `/v1/models`, run only public fixtures, emit local evidence atomically, and stop without mutating the installed Saracura runtime.

Ordinary requests never download artifacts. A missing, extra, mutable, group/world-accessible, digest-mismatched, or wrong-revision cache fails closed before service launch. Arbitrary URLs, arbitrary local source paths, non-loopback endpoints, and inherited provider credentials are rejected.

The private root is `platformdirs.user_cache_path("saracura") / "phase5b" / <descriptor-digest>` with mode `0700` by default. An operator may set `SARACURA_PHASE5B_CACHE_ROOT` to an absolute private directory on another local volume; the Mac acceptance run uses `/Volumes/DevSSD/Caches/saracura/phase5b` so model artifacts do not consume the internal SSD. The descriptor digest remains the next path component, and the same permission and containment checks apply. It contains five disjoint areas:

- `payload/source`, `payload/checkpoint` and `payload/base`: immutable, regular-file-only trees checked against the descriptor's complete ledgers; no symlink or `__pycache__` is allowed here, directories become read/execute-only and files read-only before import;
- `hf-home`: an isolated Hugging Face cache template built only from the verified checkpoint/base payloads. Before each launch, a disposable runtime view is regenerated below `derived`; its exact expected relative symlink set must resolve below the same private root, and branch refs such as `refs/main` are forbidden;
- `environment`: the Python 3.12 environment resolved from the pinned source's `uv.lock`; and
- `derived`: disposable HF runtime views, Python cache destination, logs and raw evidence. MLX merges the adapter in memory and creates no authoritative converted checkpoint. Derived files are deleted before a clean rerun, remain below the private root and never enter Git.

The service receives the verified checkpoint as a local directory. Its pinned `head.pt` supplies `base=Qwen/Qwen3.5-4B-Base` and `base_revision=1001bb4d826a52d1f399e183466143f4da7b741b`; the isolated `HF_HOME` contains only that exact revision. Before launch, Saracura resolves both snapshots offline and re-verifies their content ledgers. This prevents the global user cache or an unpinned adapter field from selecting another base.

The complete tracked source tree at Git revision `9c41005b2180347c3c646dfc9e50c4428483ec6b` is listed in the descriptor by relative path, byte count and SHA-256. GitHub archive bytes are not used as identity. The descriptor separately binds the upstream `uv.lock` SHA-256 `a9922dbb89acdef78299fd2b4a8c3f7f0fa1b2bc08b55595b6926fa785a9c466`.

`prepare` sets `UV_PROJECT_ENVIRONMENT=<private-root>/environment` and runs `uv sync --locked --extra serve --python 3.12 --no-install-project --no-dev` from the verified source. Kev itself is not built or installed editable; launch uses the environment's Python with `PYTHONPATH=<verified payload/source>`, `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPYCACHEPREFIX=<derived>/pycache-<run-id>`. The prefix is defense in depth even though bytecode writes are disabled. The payload remains read-only. Source is re-verified after dependency installation and again after evaluation; `.venv`, `kev.egg-info`, `__pycache__`, bytecode, build output or any other extra fails. Skipping the project prevents an unbound setuptools build requirement from being resolved. The upstream `.python-version` value `3.13` is not authoritative for this capsule; explicit Python 3.12 is permitted by the upstream `>=3.12,<3.14` package constraint and must satisfy the unchanged lock without a lock rewrite or re-resolution.

Network access exists only in `describe` and `prepare`. The accepted repository and evidence URLs are restricted to exact reviewed hosts: `github.com`, `raw.githubusercontent.com`, `pypi.org` and `files.pythonhosted.org`, plus the Hugging Face delivery suffixes `huggingface.co`, `hf.co` and `xethub.hf.co`. Hugging Face downloads use `token=False`, an explicit private staging/cache directory, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and `HF_HUB_DISABLE_XET=1`. After download, the runner requires the Hub API to resolve the exact requested commit, compares the complete returned file set, and verifies every available LFS size and SHA-256 against the local bytes before removing transport metadata and authoring the ledger. Git source acquisition disables credential helpers and prompting, checks the exact resolved `HEAD`, then removes `.git` before ledgering. An isolated `UV_CACHE_DIR` is used and all provider/Hugging Face credentials are stripped from child environments.

## Closed acquisition descriptor

`benchmarks/manifests/phase5-kev4b-acquisition.v1.json` is the sole acquisition authority. It contains exactly:

- `schema_version`: `phase5-kev4b-acquisition.v1`;
- `reviewed_at`: ISO date;
- `candidate_id`: `kev-4b`;
- `licenses`: exact SPDX identifiers and immutable evidence URLs for source, adapter/head and base;
- `source`: repository URL, full Git revision, Python constraint, upstream lock digest and an ordered SHA-256 ledger for the complete tracked source tree executed by the service;
- `checkpoint`: model id, full Hugging Face revision and ordered SHA-256/byte ledger for every acquired adapter artifact;
- `base_model`: model id, full revision and ordered SHA-256/byte ledger for every acquired base artifact;
- `runtime`: Python `3.12`, loopback host, fixed research port range, offline environment flags, backend `mlx`, dtype `bf16`, and dependency-lock digest;
- `limits`: maximum total bytes, maximum source file count (at least the reviewed 1,818-file tree), maximum source bytes (at least the reviewed approximately 211 MB tree), minimum free disk, minimum physical memory, startup timeout and request timeout;
- `thresholds`: the exact latency, repeat, isolation, RSS, physical-footprint, swap, pageout, pressure and disposition thresholds frozen below;
- `allowed_readiness_evidence`: exactly `pinned_license_reviewed_acquisition` and `local_systems_report`.

Every file entry contains exactly `path`, `bytes`, `sha256` and `role`. SHA-256 values for Git-managed small files are computed over downloaded file bytes; Git blob OIDs are not substituted for SHA-256. Hugging Face LFS object IDs may be used only when they are independently confirmed as the SHA-256 of the downloaded bytes. The descriptor is validated offline and is never packaged under `src/saracura`.

The descriptor is authored by an explicit `describe` command outside CI. It downloads the pinned source and model/base files into a fresh private staging directory, computes byte counts and SHA-256 values, compares each available Hugging Face LFS SHA-256 with the downloaded bytes, enumerates the complete Git tree and emits a candidate descriptor to a caller-selected temporary file. It never overwrites the canonical manifest. A maintainer reviews the licenses, revisions and emitted ledger before the canonical file is added with `apply_patch`. `prepare` then independently downloads/verifies from that canonical authority.

Phase 5A's `phase5-open-model-candidates.v1` remains immutable historical evidence. Phase 5B adds `phase5-open-model-candidates.v2` as the current candidate source of truth. Its root contains exactly `schema_version`, `reviewed_at`, `supersedes_manifest_sha256`, `allowed_dispositions`, `candidate_claim_vocabulary` and `candidates`; `supersedes_manifest_sha256` binds the exact v1 bytes. Every candidate retains the v1 fields and adds exactly `source_revision`, `base_model_id`, `base_model_revision` and `acquisition_descriptor_sha256`, nullable for candidates without reviewed acquisition.

The new `reviewed_acquisition` rule requires `architecture_class: pointer_head_causal_lm`, `declared_license: Apache-2.0`, `license_review: reviewed_for_candidate_artifacts`, full model/source/base revisions, `revision_state: immutable_pinned`, `local_execution: research_service`, `evidence_authority: reviewed_upstream_identity`, a descriptor digest matching the exact acquisition file bytes, and only `candidate_for_evaluation` in `allowed_claims`. Other v1 dispositions require all four new acquisition fields to be `null` and otherwise preserve their v1 semantics. Both v1 and v2 are routed by `benchmarks.validate_manifests`; tests prove that modifying any linked revision, license or descriptor byte without updating the entire reviewed chain fails.

The canonical v2 starts with Kev still `planned_acquisition`. It advances to `reviewed_acquisition` in the final Phase 5B diff only after the canonical descriptor's `prepare` receipt completes. A `blocked_upstream` result leaves the candidate planned and the acquisition fields null; it cannot coexist with a public `reviewed_acquisition` claim. `conditional`, `continue` and `reject_local` occur only after successful immutable preparation and may retain `reviewed_acquisition` because they describe the hardware evaluation, not acquisition validity.

The public-readiness manifest remains byte-for-byte unchanged and both gates remain `not_met`. Advancing Kev from `planned_acquisition` to `reviewed_acquisition` records candidate evidence only. It does not allow `usable_developer_preview`, `ptbr_measured`, `calibrated_for_declared_protocol`, `production_ready`, or `automation_authorized`.

## Benchmark contract

The public fixture contains self-authored, non-sensitive PT-BR and English Choice requests with the same typed schema. It is integration and systems evidence only. Phase 5B does not compute or publish language-quality accuracy from these examples.

The runner executes:

- one cold service start and readiness probe;
- three warm-up requests excluded from timing;
- 20 measured repeats for Q=1, Q=10 and Q=50 against a new state;
- 20 measured repeats for Q=1, Q=10 and Q=50 against an identical cached state;
- three exact-repeat checks per workload;
- question-together versus question-separate equivalence through the required pinned `/v1/systemone/separate` endpoint;
- option-order permutations through the required pinned `/v1/systemone/permute` endpoint, recorded as descriptive evidence only and excluded from the disposition;
- explicit oversize and invalid-contract rejection probes.

The launcher builds the child environment from an allowlist rather than inheriting the operator environment. It sets `KEV_BACKEND=mlx`, requests `KEV_DTYPE=bf16`, fixes `KEV_LORA_SCALE=1`, `KEV_MERGE=1`, `KEV_DATE_FACTS=0`, `KEV_CUDA_GRAPHS=0`, `KEV_FUSED=0`, `KEV_PREFIX_CACHE=4`, `KEV_PREFIX_MIN_TOKENS=0` and `KEV_PREFIX_MAX_TOKENS=65536`, leaves temperature at the checkpoint's declared default, and rejects any unrecognized inherited `KEV_*`. It directs `HOME`, `TMPDIR` and `XDG_CACHE_HOME` to per-run directories below `derived`. A random `KEV_API_KEY` is passed through the child environment, never argv or report. It passes `--run <verified-local-checkpoint>` and the identical value as `--fallback`, making silent fallback impossible. Because MLX uses stored dtype and ignores the request knob, the effective gate is `/v1/models`: it must confirm backend `mlx`, dtype `bfloat16`, the pinned base id and exact local checkpoint run. A request without the bearer token must be rejected. The selected port is unused before launch, must be in the descriptor's fixed research range, and must be owned by the launched service PID before any token-bearing request is sent.

After `prepare`, the service/evaluator child environment sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, isolated `HF_HOME`, invalid HTTP/HTTPS proxy endpoints and `NO_PROXY=127.0.0.1,localhost`; all provider/Hugging Face credentials are removed. The runner polls the service process socket table every 100 ms during readiness and evaluation and rejects any non-loopback listener or established connection. Offline flags and invalid proxies remain the primary egress controls because a shorter connection could escape polling. Pytest uses a fake loopback server to exercise identity, backend, dtype, token, timeout, oversize, socket and response-shape failures without weights or network.

Oversize and invalid-contract probes run only after the measured Q=1/10/50 matrix and use a new idle-host memory baseline, so a large probe cannot contaminate the primary latency or memory measurements. The two oversize paths are evaluated independently: an oversized `branch` must return HTTP 422, while the pinned upstream currently truncates an oversized `state` and returns HTTP 200. That state result is recorded explicitly as `state_truncation_contract_failure` and deterministically produces `reject_local`; it must not be misreported as a hardware-capacity failure. Cleanup is explicit and restores owner write permission on immutable payload directories before removing them, without following symlinks; tests exercise cleanup against read-only trees.

The report binds the Saracura commit, acquisition descriptor and threshold digest, source/model/base revisions, fixture digest, OS, architecture, chip family, physical-memory bucket, backend, dtype, model bytes, cold load time, request and per-decision p50/p95, decisions per second, peak service RSS, peak macOS physical footprint, swap-used delta, pageout delta, memory-pressure state, invalid-output count, repeat stability, isolation delta, option-order behavior and all capacity failures. Swap/pageout/pressure have an idle-host baseline immediately before launch. It records raw upstream `confidence` only as distribution concentration and never as measured correctness. The absolute local path returned as `/v1/models.run` is compared internally and replaced by the public candidate id in sanitized output.

The Saracura default environment does not gain `platformdirs`, `psutil` or `huggingface_hub`. A new opt-in `phase5-candidate` extra contains only those lightweight orchestration dependencies. Imports are lazy inside explicit `describe`, `prepare`, `serve` and live-measurement entrypoints. Manifest validation and fake-loopback pytest paths run in the default dev environment and tests prove they do not import these optional modules.

Public results round hardware memory to a bucket and remove usernames, absolute paths, hostnames, serial numbers, cache locations, bearer tokens, process environment and raw telemetry unrelated to the benchmark. Raw evidence remains ignored and local; the sanitized report and its manifest may enter the repository.

## Acceptance criteria

1. The acquisition descriptor and candidate-manifest revision validate offline with closed shapes, exact revisions, complete file ledgers and tamper tests.
2. Default `uv sync --locked --dev`, CI, wheel and sdist remain free of Kev/model dependencies, weights, upstream source and network access.
3. The research tooling refuses implicit download, mutable revision, arbitrary endpoint/path, non-loopback host, unsafe cache permissions and missing or extra artifacts.
4. The Python 3.12 Kev environment is isolated from Saracura's default environment and reproducibly bound to the pinned source and dependency lock.
5. A clean `prepare` verifies bytes before execution; a subsequent `serve` and `evaluate` run with network-disabled model resolution.
6. The M4 Pro/24 GB Mac either completes the full Q=1/10/50 matrix or produces a sealed, specific capacity-failure report. Failure does not weaken the contract or silently switch hardware/model.
7. The sanitized report makes no PT-BR quality, calibration, production, automation, Jev-superiority or Gate A completion claim.
8. Full repository QA, manifest routing, default-environment validation, package inspection and secret scanning pass.
9. The readiness manifest remains byte-for-byte unchanged with both gates `not_met`, and the parent plan records the out-of-process narrowing.
10. The `phase5-candidate` extra remains opt-in and lazily imported; default validators and fake-server tests prove those optional modules are not imported.

## Candidate decision after the run

The following thresholds are frozen before the run:

- every measured request finishes within 60 seconds, with p95 at most 5 seconds for Q=1, 15 seconds for Q=10 and 45 seconds for Q=50;
- invalid outputs and unauthorized accepted requests equal zero;
- measured repeats run serially only after the service queue reports idle; selected choices are identical and maximum repeated probability delta is at most `1e-6`;
- together-versus-separate maximum probability delta is at most `0.04`. An argmax flip is accepted only when the top-two probability margin of the reference distribution is less than `0.04`; every accepted near-tie flip and margin is reported. Any other flip fails. This matches the pinned upstream MLX bf16 parity rule rather than the tighter fp32 path; `0.04` is numerical tolerance, not model quality;
- backend/dtype/identity and loopback-only egress checks pass;
- service peak RSS and peak physical footprint are each at most 22 GiB; swap-used delta at most 2 GiB is the preferred envelope, more than 2 GiB through 8 GiB is degraded, and more than 8 GiB or critical memory pressure is a hard capacity failure;
- cold readiness finishes within 600 seconds.

The disposition is deterministic:

- `continue`: every functional/identity/security threshold passes, cold readiness is at most 600 seconds, all latency p95 thresholds pass, peak RSS and footprint are at most 22 GiB, swap delta is at most 2 GiB and memory pressure never becomes critical;
- `conditional`: every functional/identity/security threshold passes, but at least one latency soft threshold, the 22 GiB RSS threshold or the 2 GiB preferred swap threshold fails while swap remains at most 8 GiB and no critical pressure/OOM occurs;
- `reject_local`: cold readiness exceeds 600 seconds after the process starts, any measured request exceeds 60 seconds, any functional/identity/security threshold fails, an oversize or invalid-contract probe is accepted incorrectly, Q=1/10/50 cannot complete, swap exceeds 8 GiB, or OOM/critical memory pressure occurs;
- `blocked_upstream`: immutable acquisition, license evidence, locked dependency installation or offline snapshot resolution cannot be established before model execution.

The report records exactly one disposition. It never substitutes another model or revision in the same run.

The parent Phase 5 plan's phrase “opt-in local adapter” is narrowed by this reviewed spec to an opt-in out-of-process research adapter. Any installed runtime registration requires a later phase.

## Rollback

The Phase 5B PR is independently revertible. Removing its benchmark tooling and manifests leaves the runtime and Phase 5A direction unchanged. Private cache cleanup is a separate explicit operator action and is not performed automatically by tests, rollback or uninstallation.

## Validation

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src benchmarks tests
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python -m benchmarks.validate_default_environment
uv build
uv run python -m benchmarks.inspect_wheel dist/*.whl dist/*.tar.gz
gitleaks detect --source . --no-git --redact --exit-code 1
```

The live acquisition and Mac benchmark are explicit acceptance commands defined by the implementation. They are not part of ordinary CI and must never run from `pytest`.
