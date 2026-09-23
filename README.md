---
title: Saracura
kind: reference
area: product
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: README.md
globalRef: qmd://saracura/README.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-23
sourceRefs: []
related:
  - docs/README.pt-BR.md
  - SECURITY.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura

[English](README.md) | [Português (Brasil)](docs/README.pt-BR.md)

Saracura is a PT-BR-first, local-first research engine for typed decisions at volume. Its architectural thesis is simple: encode one state once, then answer many calibrated questions with low incremental cost.

> One state. Many calibrated decisions. Open and local.

This repository is currently a **research-only alpha**. It does not contain trained weights, does not claim decision quality, and must not be used as an automation or authorization gate. The included deterministic fixture backend exists only to exercise the contract and runtime invariants.

PT-BR-first means Brazilian Portuguese is the first language for examples, data governance, annotation, and evaluation. It does not mean this alpha already ships a PT-BR-optimized checkpoint. The API and architecture remain language-neutral so the same evidence protocol can expand to English and other languages. A Portuguese translation of this README is available at [docs/README.pt-BR.md](docs/README.pt-BR.md).

English is the canonical language for technical documentation. The quickstart, public examples, and PT-BR evaluation materials are also maintained in Brazilian Portuguese where applicable.

## Current scope

- `v1alpha1` closed request, response, and error contracts
- `choice` questions in known, immutable workflows
- versioned NFC normalization followed by RFC 8785 canonicalization
- length-prefixed semantic segments
- one state encoding reused across Q=1, Q=10, and Q=50
- question-isolation checks with an absolute numeric tolerance of `1e-12`
- fail-closed schema-cache keys with complete canonical schema bytes and revision axes
- immutable, fail-closed calibration artifacts
- local in-process runtime and CLI
- one opt-in experimental `laya-universal` backend for the exact dynamic Choice workflow
- opt-in research lanes for encoder acquisition, synthetic head training, human PT-BR calibration, and external controls
- planned-only Phase 4E policy and deterministic ranker contracts; no `saracura-universal` runtime is installed

Not included in the installed runtime: bundled or request-triggered model downloads, bundled datasets or checkpoints, `saracura-universal`, dynamic labels outside the exact experimental Phase 4D Choice workflow, boolean or ordinal heads, HTTP serving, remote fallback, telemetry, or production automation. The planned `universal-choice@phase4e-saracura-ranker.v1` contract is unsupported until Phase 4E.3 seals a real checkpoint that passes its holdout gate. Research tooling can acquire reviewed encoder snapshots and train local experimental heads only through explicit, offline-first operator workflows.

## Two-tier research architecture

Saracura is pursuing one typed decision API with two execution tiers. The experimental `universal` tier is intended to accept new Choice schemas without requiring users to adopt predefined model packs or task-specific heads. The optional `compiled` tier specializes stable, high-volume decisions when that optimization is justified. The first universal implementation remains experimental, no model is production-approved, and results do not authorize automation. TypeSafe/Jev is a benchmark and design reference, not a Saracura runtime dependency or fallback. See [ADR 0001](docs/decisions/0001-two-tier-decision-architecture.md).

## Install and verify

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

No model backend or large ML dependency is installed by the default development environment.

## Run the self-authored PT-BR fixture

```bash
uv run saracura decide \
  --request examples/ptbr-support-request.json \
  --calibration examples/ptbr-support-calibration.json
```

The output distinguishes raw scores from normalized fixture probabilities and reports `fixture_only`. The artifact exercises compatibility checks but was not fitted on calibration data; it is not evidence that the fixture backend is accurate, calibrated, or safe for automation. `verified_for_research` is reserved for a future artifact that passes its declared held-out protocol.

## Design boundary

Saracura is standalone. Installation, tests, CLI, and examples do not require any private code, service, data, or configuration. All fixture content in this phase is self-authored and declared in `benchmarks/manifests/ptbr-fixture-v1.json`.

See [SECURITY.md](SECURITY.md) before adding model, tokenizer, dataset, or plugin loading. Contributions must follow [CONTRIBUTING.md](CONTRIBUTING.md).

Runtime dependency licenses and resolved versions are recorded in [docs/dependency-licenses.md](docs/dependency-licenses.md).

## Run the local scaling harness

The checkout-only benchmark runner consumes the self-authored fixture and executes the public `DecisionEngine` at Q=1, Q=10, and Q=50. It records raw samples and a Markdown report under the ignored `.artifacts/` directory:

```bash
uv run python -m benchmarks.run \
  --suite decision-scaling \
  --backend fixture \
  --warmups 1 \
  --iterations 10 \
  --code-revision local-smoke \
  --output-dir .artifacts/benchmarks
```

The result schema is `phase2a.v1`. Percentiles use nearest-rank semantics (`sorted[ceil(p*n)-1]`), and each sample records the answer digest and the single state-encoding observation. This fixture is a research instrument only: its timing is not learned-model performance, does not establish quality, and does not authorize automation or production decisions. The runner is not part of the installed wheel and does not download models, tokenizers, datasets, or ML runtimes.

## Optional encoder research gate

Phase 2B is an explicit, local research lane. It does not select a model,
train, calibrate, measure quality, or authorize automation. The default
environment remains lightweight and offline. After the default validation has
passed, an operator may install the isolated extra and acquire one reviewed
candidate at its immutable revision:

```bash
uv sync --locked --dev --extra encoder-eval
uv run python -m benchmarks.encoder_gate validate-registry
uv run python -m benchmarks.encoder_gate acquire \
  --candidate multilingual-minilm-l12 --allow-network
uv run python -m benchmarks.encoder_gate probe \
  --candidate multilingual-minilm-l12 --device cpu --output-dir .artifacts/encoders
```

Acquisition is never request-triggered. It accepts only a bundled candidate
identifier, verifies every byte against the registry, and refuses to replace
an immutable snapshot. `mmbert-base` is deliberately blocked because the
reviewed revision publishes `pytorch_model.bin` without a safetensors weight.
Probe reports are encoder-only observations and do not measure decision quality,
trained heads, calibration, end-to-end latency, or automation readiness.

## Phase 4A opt-in local MiniLM routing

Phase 4A can execute one frozen, synthetic-only five-label support-routing head
when an operator supplies an already verified local encoder snapshot, its sealed
training manifest, and its safetensors checkpoint. Nothing is downloaded or
discovered from a cache, environment variable, request, or network service.
The default installation remains unchanged; opt in explicitly:

    uv sync --locked --dev --extra local-minilm
    uv run saracura describe-backend --backend minilm-routing \
      --encoder-snapshot <verified-snapshot> \
      --training-manifest <training-manifest.json> \
      --checkpoint <checkpoint.safetensors> --device cpu

The loader verifies every supplied descriptor byte through local file
descriptors, constructs BERT and the tokenizer directly from those bytes, and
binds the runtime/device ABI to the immutable model revision. A Phase 4A
identity artifact can then be created for that exact revision and used with the
single frozen support-routing workflow. It is deliberately fixture_only: it has
no fitted data, no evaluation claim, no threshold, and never authorizes
automation. Apple MPS is explicit, has no CPU fallback, and is an operator
choice rather than a default.

## Phase 4D experimental universal Choice

`laya-universal` is an opt-in local, research-only backend for the single
dynamic `universal-choice@phase4d-laya.v1` Choice workflow. It requires the
isolated optional dependency group, an operator-supplied immutable local Laya
snapshot, and an explicit CPU or MPS device; it never downloads, discovers, or
contacts a model provider. The synthetic PT-BR request in
[`examples/ptbr-universal-request.json`](examples/ptbr-universal-request.json)
uses e-mail triage only as a future Phase 4F shadow-mode pilot and contains no mailbox
data.

```bash
uv sync --locked --dev --extra universal-local
uv run saracura describe-backend --backend laya-universal \
  --model-snapshot <verified-local-laya-snapshot> --device cpu
uv run saracura decide --backend laya-universal \
  --request examples/ptbr-universal-request.json \
  --model-snapshot <verified-local-laya-snapshot> --device cpu
```

The response is explicitly `uncalibrated`, `abstained`, and
`automation_allowed=false`. Its normalized values are ranking weights, not
confidence or a permission to archive, delete, move, reply, forward, or make
any other mailbox change. The candidate remains
`research_only_unresolved_provenance`; a successful local smoke demonstrates
execution compatibility only, not quality, calibration, licensing, or
production readiness.

## Phase 4E planned Saracura-owned universal checkpoint

Phase 4E.1 adds only offline policy, rendering, ranker, checkpoint, and
backend/workflow compatibility contracts for a future Saracura-owned universal
ranker. It does not download a MiniLM snapshot, call a provider, generate a
corpus, train a checkpoint, or register `saracura-universal` in the CLI or
runtime. The future `universal-choice@phase4e-saracura-ranker.v1` workflow is
therefore planned and unsupported until Phase 4E.3 seals a real synthetic-only
checkpoint and it passes the reviewed holdout gate. Laya remains an external
control, never a teacher, checkpoint source, or Saracura-owned model.

## Phase 2C data and privacy gate

Before any training packet can be authored, the repository validates the
metadata-only source-policy registry:

```bash
uv run python -m benchmarks.data_policy_gate validate-registry
```

The registry has exactly eight source categories and approves no dataset bytes.
Only future first-party human-original cases and deterministic derivatives may
be considered for PT-BR training, subject to a separate artifact manifest and
human privacy/rights review. The official Amazon MASSIVE `pt-PT` source is
restricted to a separate external-control role; it cannot become PT-BR
evidence. Model-assisted, private, customer, and unvetted-public sources remain
quarantined or blocked. This is an engineering gate, not legal advice or proof
of dataset quality. No validator path downloads data or imports a dataset
loader.

## Phase 3A first-party packet gate

The support-routing annotation protocol and deterministic split gate are
offline plumbing only. No dataset record is bundled, and the commands do not
authorize training, quality claims, publication, or automation:

```bash
uv run python -m benchmarks.first_party_gate validate-protocol
uv run python -m benchmarks.validate_manifests
```

Future maintainers provide a canonical, externally attested base-state JSONL
and a public seed to build a no-clobber split plan. Human authorship, privacy,
rights, representativeness, and independent review remain outside the tool.

## Phase 3C end-to-end throughput benchmark

Phase 3C is a checkout-only, synthetic-only performance instrument. It measures
the reviewed MiniLM head from PT-BR text through tokenization, Apple MPS
inference, pooling, CPU transfer, and choice materialization, with separate
load and warm-path evidence. Optional Jev control requests use the pinned
`typesafe/jev-1.13` model through OpenRouter only when explicitly enabled.

The default environment remains offline and lightweight: importing the runner
does not import Torch or Transformers. The real run requires a verified local
encoder snapshot, sealed Phase 3B manifests, and an explicit approved budget;
it must not be used to authorize automation or make general quality claims.
Offline protocol tests can be run with:

```bash
uv run pytest -q tests/test_e2e_benchmark.py
uv run ruff check benchmarks/e2e_benchmark.py tests/test_e2e_benchmark.py
uv run mypy benchmarks/e2e_benchmark.py tests/test_e2e_benchmark.py
```

## Phase 3D native TypeSafe control benchmark

Phase 3D is an explicit synthetic-only control experiment. It reuses the sealed
Phase 3C workload and request builder, pins `jev-1.13.0`, and keeps native
probability, confidence, token, latency, and computed-cost evidence separate.
The default environment remains offline and lightweight; no TypeSafe SDK or
runtime backend is added. A live run requires the generic `TYPESAFE_API_KEY`
environment contract, `--allow-network`, and the reviewed `0.25` local budget.
Computed cost is not a provider billing receipt and no result chooses an
automation threshold.

```bash
uv run pytest -q tests/test_typesafe_native.py
uv run ruff check benchmarks/typesafe_native.py tests/test_typesafe_native.py
uv run ruff format --check benchmarks/typesafe_native.py tests/test_typesafe_native.py
uv run mypy benchmarks/typesafe_native.py tests/test_typesafe_native.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
