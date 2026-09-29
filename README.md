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
lastReviewedAt: 2026-09-27
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

Saracura is an open decision runtime and model lab for local, high-throughput workflows, with a PT-BR-first benchmark and pluggable decision backends.

> One typed API. Pluggable research backends. Evidence before claims.

This repository is currently a **research-only alpha**. It does not contain trained weights, does not claim decision quality, and must not be used as an automation or authorization gate. The included deterministic fixture backend exists only to exercise the contract and runtime invariants.

PT-BR-first means Brazilian Portuguese is the first language for examples, data governance, annotation, and evaluation. It does not mean this alpha already ships a checkpoint with measured PT-BR quality. The API and architecture remain language-neutral so the same evidence protocol can expand to English and other languages. A Portuguese translation of this README is available at [docs/README.pt-BR.md](docs/README.pt-BR.md).

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
- one optional loopback-only `systemone` backend for the exact `universal-choice@phase5c-systemone.v1` Choice workflow
- one direct CPU `julia` backend for the pinned Apache-2.0 Julia-1 checkpoint and the exact `universal-choice@phase5c-julia.v1` Choice workflow
- opt-in `saracura-universal` for exactly `universal-choice@phase4e-saracura-ranker.v1`, using explicit verified private artifacts only

Not included in the installed runtime: bundled or request-triggered model downloads, bundled datasets or checkpoints, dynamic labels outside the exact Phase 4D and Phase 4E Choice workflows, boolean or ordinal heads, HTTP serving, remote fallback, telemetry, calibration, confidence thresholds, public checkpoint publication, or production automation. The installed Saracura-owned backend is opt-in, synthetic-only and research-only; it requires an operator-provided verified MiniLM snapshot and sealed training capsule, returns uncalibrated abstained ranking weights, and never authorizes automation. Research tooling can acquire reviewed encoder snapshots and train local experimental heads only through explicit, offline-first operator workflows.

## Phase 5 direction and two-tier research architecture

Saracura preserves one typed decision API with a pluggable `universal` tier for new Choice schemas and an optional `compiled` tier for stable, repeated high-volume workflows. The current `saracura-universal-ranker.v0` is a historical MiniLM bi-encoder projection ranker, not the presumed product foundation and not a parameter-count claim. Phase 5 evaluates evidence-supported open candidates before any new checkpoint training. TypeSafe/Jev remains an independent external control, never a runtime dependency or fallback. No candidate is production-approved, calibrated for a declared protocol, or authorized to automate. See [ADR 0001](docs/decisions/0001-two-tier-decision-architecture.md) and [ADR 0002](docs/decisions/0002-open-model-runtime-and-readiness.md).

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

## Run Julia-1 locally on CPU

Julia-1 is the recommended backend for this experimental preview. Saracura pins
the public source revision and checkpoint digest, loads the model directly from
an explicit local snapshot, and keeps every result research-only.

```bash
uv sync --locked --dev
uvx --from huggingface-hub hf download SupersonicLabs/Julia-1 \
  --revision a85b127321d580d65176c89ced8273f305745d85 \
  --local-dir ./Julia-1
uv pip install -e ./Julia-1

uv run --no-sync saracura decide \
  --backend julia \
  --request examples/ptbr-julia-request.json \
  --model-snapshot ./Julia-1 \
  --device cpu \
  --timing
```

English equivalent:

```bash
uv run --no-sync saracura decide \
  --backend julia \
  --request examples/en-julia-request.json \
  --model-snapshot ./Julia-1 \
  --device cpu \
  --timing
```

Julia-1 is a third-party open checkpoint, not a Saracura-trained model. Its
published multilingual results include Portuguese from Portugal, not a
Saracura PT-BR evaluation. The response therefore remains `uncalibrated`,
`abstained`, and `automation_allowed=false` until workflow-specific evaluation
and calibration exist.

## Native PT-BR FAQ benchmark

Phase 5D adds a reproducible four-way answer-selection benchmark over 373
Brazilian Portuguese questions from Banco Central do Brasil. It is a narrow
external control, not a general leaderboard: chance is `0.25`, and a frozen
lexical-overlap baseline scores 265/373 (`0.710455764075067`). Rejects and
errors remain in the denominator. Results are uncalibrated and cannot authorize
automation.

The data is not redistributed. Download the pinned `MTEB-BR/faq-bacen`
revision into a caller-controlled cache, then run with absolute paths. The BCB
source declares ODbL; the upstream `Itau-Unibanco/FAQ_BACEN` repository
declares Apache-2.0; the pinned MTEB-BR dataset card does not declare a license.
The BCB portal says its open-data catalog excludes personal or restricted
records, but Saracura has not performed a row-level privacy review.

The first Apple Silicon CPU run produced the following planned-set results:

| Backend | Correct / planned | Planned accuracy | Coverage | Rejected |
| --- | ---: | ---: | ---: | ---: |
| Julia-1 | 128 / 373 | 34.32% | 98.66% | 5 backend-capacity cases |
| Saracura-owned ranker | 103 / 373 | 27.61% | 98.66% | 5 request-capacity cases |
| Frozen lexical baseline | 265 / 373 | 71.05% | 100% | 0 |

Both model-backed results are only modestly above chance and substantially
below the lexical baseline. That is negative quality evidence, not a
superiority claim. The committed aggregate reports preserve exact model, code,
protocol, latency, and rejection provenance without dataset rows.

The opt-in Phase 5D.1 Julia strategy tests whether averaging the complete set of
cyclic criterion rotations reduces that position bias. It performs one Julia
inference per criterion (four for this benchmark), so latency rises with the
number of choices. The verified CPU report scores 161/373 (`43.16%`) at `98.66%`
coverage, up `8.85` percentage points from the 128/373 (`34.32%`) single-pass
result. Planned position accuracy narrows from `16.13%`–`58.51%` to
`40.86%`–`46.24%`. Warm p50/p95 latency is `87.31/100.11 ms` and throughput is
`8.78 decisions/s`, versus `21.70/25.01 ms` and `29.08 decisions/s` single-pass.
Both remain above the 25% chance level and well below the 265/373 (`71.05%`)
lexical baseline. This is a post-hoc experiment on the same development benchmark,
with no held-out confirmation; cyclic mean was the only aggregation explored in
that cycle. It wraps a third-party Julia-1 checkpoint and does not make it a
Saracura-owned model. Rotating criterion IDs together with their descriptions
also cannot isolate position bias from label-token preference. No result is
calibrated or permits automation.

To run the separate v2 cyclic-mean report, pass `--strategy cyclic_mean` and
write to its canonical filename:

```bash
uv run --no-sync python -m benchmarks.ptbr_native run \
  --backend julia --strategy cyclic_mean --device cpu \
  --model-snapshot /absolute/path/to/Julia-1 \
  --data-root /absolute/path/to/MTEB-BR-faq-bacen-revision \
  --output /absolute/path/to/phase5d1-ptbr-faq-bacen-julia-cyclic-mean-cpu.json

uv run --no-sync python -m benchmarks.ptbr_native verify \
  /absolute/path/to/phase5d1-ptbr-faq-bacen-julia-cyclic-mean-cpu.json
```

The committed v2 report records the inference strategy and per-decision inference count
alongside accuracy, position, coverage, latency, and throughput metrics. Its
code and ensemble-manifest digests are checked against the report's ancestor
Git tree, so later source edits do not rewrite historical evidence.

```bash
uv sync --locked --dev --extra ptbr-benchmark
uv pip install -e /absolute/path/to/Julia-1
uv run --no-sync python -m benchmarks.ptbr_native run \
  --backend julia --device cpu \
  --model-snapshot /absolute/path/to/Julia-1 \
  --data-root /absolute/path/to/MTEB-BR-faq-bacen-revision \
  --output /absolute/path/to/phase5d-ptbr-faq-bacen-julia-cpu.json

uv run --no-sync python -m benchmarks.ptbr_native verify \
  /absolute/path/to/phase5d-ptbr-faq-bacen-julia-cpu.json
```

For the Saracura-owned ranker, sync both `ptbr-benchmark` and
`universal-local`, then use `--backend saracura-universal` with the explicit
verified `--encoder-snapshot` and `--training-capsule`. That historical ranker
remains `synthetic_only_research`; its smaller state envelope is part of what
the benchmark measures. See the [frozen Phase 5D protocol](docs/action/specs/phase5d-native-ptbr-benchmark.md).

## Run the experimental System One loopback adapter

The System One backend is an optional, loopback-only adapter for the exact
`universal-choice@phase5c-systemone.v1` Choice workflow. It requires a
separately started open checkpoint listening on `http://127.0.0.1:<port>`, an
explicit model identity, and a bearer token supplied only through the
`SARACURA_SYSTEMONE_API_KEY` environment variable. The adapter disables all
proxy handlers and never follows redirects.

PT-BR quickstart:

```bash
SARACURA_SYSTEMONE_API_KEY=<token> uv run saracura decide \
  --backend systemone \
  --request examples/ptbr-systemone-request.json \
  --endpoint http://127.0.0.1:<port> \
  --model-id systemone-compatible \
  --model-revision <pinned-revision> \
  --checkpoint-sha256 <sha256>
```

English equivalent:

```bash
SARACURA_SYSTEMONE_API_KEY=<token> uv run saracura decide \
  --backend systemone \
  --request examples/en-systemone-request.json \
  --endpoint http://127.0.0.1:<port> \
  --model-id systemone-compatible \
  --model-revision <pinned-revision> \
  --checkpoint-sha256 <sha256>
```

The response is `uncalibrated`, `abstained`, and
`automation_allowed=false`. The model name returned by the checkpoint is
recorded but need not equal the request alias. This is an experimental
wire-interoperability smoke only, not candidate support or a systems
disposition.

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
uses e-mail triage only as a fictional reference scenario for Phase 4F.1's
read-only shadow evaluation. It contains no mailbox data and does not connect to
e-mail services.

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

## Phase 4E Saracura-owned universal checkpoint

Phase 4E.3C installs `saracura-universal` only for the sealed
`universal-choice@phase4e-saracura-ranker.v1` checkpoint. It accepts explicit
local paths only after descriptor, manifest, snapshot and conformance checks;
the package contains metadata, never weights, vectors, raw rows or a capsule.
It is synthetic-only research infrastructure, is uncalibrated and abstained,
and cannot enable automation. Laya remains an external control, never a
teacher, checkpoint source, or Saracura-owned model.

```bash
saracura decide --backend saracura-universal --request examples/ptbr-saracura-request.json \
  --encoder-snapshot <verified-minilm-snapshot> \
  --training-capsule <sealed-saracura-capsule> --device cpu
```

### Phase 4E operator lane

The checkout-only Phase 4E pipeline is explicit and offline-first. Its `plan`,
`extract`, `train`, and `verify` commands construct no socket. Generated
material stays under ignored `.artifacts/`; it is synthetic-only research
evidence, not a runtime registration, quality claim, or automation authority.

```bash
uv run python -m benchmarks.phase4e_pipeline plan \
  --output .artifacts/phase4e/plan.json

# Only corpus and pilot can use the network, and each requires literal
# --allow-network. OPENROUTER_API_KEY is supplied ephemerally by the operator
# environment; it is never an argument or an artifact.
uv run --extra local-minilm python -m benchmarks.phase4e_pipeline corpus \
  --plan .artifacts/phase4e/plan.json \
  --snapshot <verified-minilm-snapshot> \
  --work-dir .artifacts/phase4e/corpus-work \
  --packet .artifacts/phase4e/accepted-packet \
  --allow-network

uv run --extra local-minilm python -m benchmarks.phase4e_pipeline extract \
  --packet .artifacts/phase4e/accepted-packet \
  --snapshot <verified-minilm-snapshot> --device cpu \
  --output .artifacts/phase4e/embedding-capsule
uv run --extra local-minilm python -m benchmarks.phase4e_pipeline train \
  --packet .artifacts/phase4e/accepted-packet \
  --snapshot <verified-minilm-snapshot> \
  --embeddings .artifacts/phase4e/embedding-capsule --device cpu \
  --output-parent .artifacts/phase4e/training --output-name saracura-universal-v0
uv run --extra local-minilm python -m benchmarks.phase4e_pipeline verify \
  --packet .artifacts/phase4e/accepted-packet \
  --embeddings .artifacts/phase4e/embedding-capsule \
  --training .artifacts/phase4e/training/saracura-universal-v0
```

Corpus requests are pinned to OpenRouter HTTPS with redirects and proxies
disabled. Historical corpus and comparison ledgers retain their original hard
caps for reproducibility. The current acceptance-protocol pilot uses report-only
cost telemetry: financial amounts never authorize, block, or stop execution.
It reports every USD 10 of run-local provider spend and the final actual total.
A durable pre-send reservation that cannot be resolved still stops resume
instead of repeating a possibly charged request.

The current policy keeps the legacy `corpus` and `train` entry points
fail-closed. They can be re-enabled only by a reviewed post-pilot policy
revision after a pilot `PASS`; the pilot itself never changes those
authorizations.

The planned PT-BR recovery lane requires `SARACURA_PRIVATE_STATE_ROOT` to name an existing, user-owned `0700` directory outside every repository and Git worktree. Operators must choose a private, non-synchronized external volume: never a cloud-synchronized directory. That runtime-only setting moves Phase 4E research ledgers, raw evidence, recovery reports, and recovery packet artifacts; it never moves the immutable holdout-release registry. The recovery lane remains unsupported until its separately reviewed increments are complete.

```bash
uv run python -m benchmarks.phase4e_pipeline pilot-plan \
  --output .artifacts/phase4e/pilot-plan-v12/plan.json

uv run python -m benchmarks.phase4e_pipeline import-ledgers \
  --source .artifacts/phase4e \
  --artifact-root <durable-phase4e-research-ledger-root>

uv run --extra local-minilm python -m benchmarks.phase4e_pipeline pilot \
  --plan .artifacts/phase4e/pilot-plan-v12/plan.json \
  --snapshot <verified-minilm-snapshot> \
  --work-dir .artifacts/phase4e/pilot-work-v12 \
  --report .artifacts/phase4e/pilot-report-v12 \
  --artifact-root <durable-phase4e-research-ledger-root> \
  --allow-network
```

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

## Phase 4F.1 read-only shadow reference

The e-mail triage example is a local, read-only shadow workflow, not a mailbox
integration. Saracura does not connect to Gmail, Outlook, IMAP, a browser, or a
network service, and it does not read, move, label, delete, reply to, or otherwise
mutate messages. The reference policy and JSONL records contain fictional data;
operators must choose a minimized `subject` and `preview` before sending input
to the local command.

Run it only with the exact verified local encoder snapshot and sealed training
capsule for the policy's model revision. Those artifacts are not bundled or
downloaded. `shadow-decide` emits content-free decisions, not input state,
criterion descriptions, raw scores, or timing. The output is always
`uncalibrated`, abstained, and `automation_allowed=false`; its ordered ranking
weights are not confidence and do not authorize actions. Operator feedback is a
separate descriptive evaluation only: it must not be used to train, fine-tune,
retrieve, optimize prompts, or calibrate a model. A provider-specific mailbox
adapter, if ever approved, remains outside this public increment and belongs to
a later private Phase 4F.2.

```bash
uv sync --locked --dev --extra universal-local
mkdir -p .artifacts/shadow
uv run saracura shadow-decide \
  --policy examples/ptbr-email-shadow-policy.json \
  --encoder-snapshot <verified-local-encoder-snapshot> \
  --training-capsule <verified-local-training-capsule> \
  --device cpu \
  < examples/ptbr-email-shadow-input.jsonl \
  > .artifacts/shadow/decisions.jsonl
uv run saracura shadow-evaluate \
  --policy examples/ptbr-email-shadow-policy.json \
  --decisions .artifacts/shadow/decisions.jsonl \
  --feedback examples/ptbr-email-shadow-feedback.jsonl
```

The sample policy is editable data, not a hard-coded runner pack or a claim
that the checkpoint is optimized for e-mail. At execution, the verified
tokenizer enforces the existing limits of 128 context tokens and 96 tokens per
criterion. No preview token budget is published here: it requires the separate
operator smoke with the verified tokenizer, and the runner never truncates
input. A valid tokenizer result that exceeds those limits is reported as
`CAPACITY_EXCEEDED`; malformed tokenizer output is treated as
`BACKEND_UNAVAILABLE`. The CLI keeps batch output all-or-nothing and emits no
partial decisions on either failure. Integrations can call the public
`validate_shadow_state_capacity` helper before constructing a `ShadowItem`; it
owns the canonical serialized state byte/codepoint limits, while policy-key
matching remains a separate validation. The JSON policy may appear in the
source distribution but is not in the wheel; fictional JSONL inputs and
feedback remain excluded from both archives.

An 8-item synthetic operator smoke confirmed local execution only; its observed
agreement is not evidence of model quality or readiness. See the [Phase 4F.1
smoke result](docs/action/phase4f1-shadow-evaluation-result.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Phase 4E.4 blind comparison status

<!-- phase4e4-status:begin -->
<!-- phase4e4-status-binding:scored:080e9264b3d7fbaf2c95ff51d2599f0522daf27bf94861a144c1463e94ac5596 -->
The Phase 4E.4 live blind comparison is complete. Saracura scored 187/198 and Laya scored 139/198 on the same accepted synthetic records. See the [aggregate result](docs/action/phase4e-comparison-result.md). This remains synthetic research evidence and does not authorize automation.
<!-- phase4e4-status:end -->
