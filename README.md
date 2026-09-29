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
lastReviewedAt: 2026-09-29
sourceRefs: []
related:
  - docs/README.pt-BR.md
  - docs/decisions/0004-model-first-product-direction.md
  - SECURITY.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura

[English](README.md) | [Português (Brasil)](docs/README.pt-BR.md)

Saracura is an open, local-first decision model project for fast typed decisions at scale, with PT-BR-first evaluation and a language-neutral API.

> Small local models. Many typed decisions. Evidence on real hardware.

Saracura is currently a **model-first development preview**. The public runtime and evaluation tools exist today; the first generally usable Saracura-owned checkpoint is the next release milestone and is not published yet. Until that checkpoint and its measurements are available, this repository makes no comparative quality claim and must not be used as an automation or authorization gate.

## What we are building

Saracura is being built as a model people can download and run, not as a wrapper around another decision model and not as a benchmark laboratory whose main output is infrastructure.

The project focuses on the intersection of four properties:

- small models that run locally on practical CPU, Apple Silicon, or modest GPU hardware;
- typed decisions over new schemas through one stable API;
- high-throughput workloads, including many decisions over shared context;
- PT-BR-first training and evaluation without making the architecture language-specific.

The next public milestone is a Saracura-owned checkpoint with:

- a one-command local quickstart;
- public, immutable weights and a model card;
- reproducible PT-BR quality and hardware measurements;
- a scoped claim tied to the result the model actually demonstrates.

Calibration, selective prediction, and automation safety remain important later milestones. They do not block publishing a useful, honestly scoped model preview.

## Current release

The current package provides the typed decision API, local CLI, strict request and response contracts, benchmark tooling, and the runtime foundation for the upcoming model. It does **not** yet ship the public Saracura checkpoint described above.

Earlier releases integrated third-party checkpoints and external controls to validate the runtime and learn where existing approaches fail. Those integrations remain reproducible research evidence, but they are not the Saracura product, are not recommended as its default model, and do not define the public roadmap.

The durable product boundary is recorded in [ADR 0004: model-first product direction](docs/decisions/0004-model-first-product-direction.md).

## Roadmap

GitHub Issues are the public operational roadmap. The [v0.2.0 Model Preview milestone](https://github.com/felhen-ai/saracura/milestone/1) shows how much remains before the first model release; [roadmap issue #49](https://github.com/felhen-ai/saracura/issues/49) records the dependency order and definition of done.

Each roadmap issue must end in a user-visible model, evaluation, packaging, or release artifact. Work on calibrated automation is tracked separately and does not dilute progress toward v0.2.0.

## Architecture

Saracura exposes one typed decision API with two model roles:

- `universal`: the default Saracura model for new Choice schemas;
- `compiled`: an optional specialization for stable, repeated, high-volume workflows when measurements justify it.

The first model release targets the universal role. A compiled path is valuable only when it improves the quality-throughput frontier for a declared workload; it is not a reason to delay the universal checkpoint.

See [ADR 0001](docs/decisions/0001-two-tier-decision-architecture.md) for the two-tier architecture and [ADR 0002](docs/decisions/0002-open-model-runtime-and-readiness.md) for the evidence and release boundaries.

## Install for development

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/felhen-ai/saracura.git
cd saracura
uv sync --dev
uv run pytest -q
```

The default environment stays lightweight and does not download model weights or heavy ML dependencies. Model acquisition is always explicit and revision-pinned.

## Exercise the current API contract

The repository includes a self-authored PT-BR fixture so contributors can exercise the typed contract before downloading model artifacts:

```bash
uv run saracura decide \
  --request examples/ptbr-support-request.json \
  --calibration examples/ptbr-support-calibration.json
```

This fixture is test infrastructure, not a trained model and not evidence of decision quality. The README will replace this section with the model quickstart when the Saracura checkpoint is published.

## Evidence, not competitor marketing

Saracura keeps reproducible research records, including negative results. Third-party models may appear in those records only as controls under the same declared protocol. The homepage does not position Saracura as an adapter for, or a minor variation of, any one external checkpoint.

Relevant records:

- [open-model readiness and evaluation protocol](docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md);
- [native PT-BR benchmark protocol](docs/action/specs/phase5d-native-ptbr-benchmark.md);
- [historical blind comparison result](docs/action/phase4e-comparison-result.md);
- [read-only shadow evaluation result](docs/action/phase4f1-shadow-evaluation-result.md).

Historical benchmark results do not establish that the upcoming Saracura model is better. A comparative claim becomes public only when a Saracura-owned checkpoint and directly comparable evidence exist.

## Product guardrail

New adapters, harnesses, candidate integrations, or benchmark infrastructure must directly contribute to at least one of these outcomes:

1. selecting or training the Saracura checkpoint;
2. measuring model quality, footprint, latency, or throughput;
3. packaging and running the model locally;
4. improving a measured weakness in the released model.

Work that satisfies none of these outcomes is not on the model-release critical path. This keeps supporting infrastructure from becoming the product by accident.

## Safety and independence

- Saracura remains usable without Felhen, AIOS, private services, private data, or private configuration.
- No request triggers an implicit model download or remote fallback.
- Uncalibrated scores are not presented as confidence.
- A model preview does not authorize automatic high-risk actions.
- Training, calibration, and held-out evaluation records remain disjoint under their declared protocols.

See [SECURITY.md](SECURITY.md) before adding model, tokenizer, dataset, or plugin loading. Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
