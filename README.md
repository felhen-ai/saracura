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
lastReviewedAt: 2026-09-21
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

Saracura is an early, local-first engine for typed decisions. Its architectural thesis is simple: encode one state once, then answer many calibrated questions with low incremental cost.

> One state. Many calibrated decisions. Open and local.

This repository is currently a **research-only alpha scaffold**. It does not contain trained weights, does not claim decision quality, and must not be used as an automation or authorization gate. The included deterministic fixture backend exists only to exercise the contract and runtime invariants.

Brazilian Portuguese is the first fixture language, while the API and architecture remain language-neutral. A Portuguese translation of this README is available at [docs/README.pt-BR.md](docs/README.pt-BR.md).

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

Not included: model downloads, training, external datasets, dynamic labels, boolean or ordinal heads, HTTP serving, remote fallback, telemetry, or production automation.

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

## License

Apache License 2.0. See [LICENSE](LICENSE).
