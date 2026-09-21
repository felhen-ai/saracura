---
title: Contributing to Saracura
kind: policy
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: CONTRIBUTING.md
globalRef: qmd://saracura/CONTRIBUTING.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-21
sourceRefs: []
related:
  - README.md
  - AGENTS.md
  - SECURITY.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Contributing to Saracura

Saracura is intentionally narrow while its contract and evidence protocol are being established. Please discuss changes that expand supported decision types, workflows, data sources, model loading, networking, or automation before implementation.

## Development setup

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

Pull requests should include focused tests and explain contract compatibility. Keep optional backends isolated from the default dependency group. Never add private traces, customer data, secrets, locally identifying paths, external datasets without a license manifest, or generated labels without provenance and usage rights.

The fixture backend and its examples are contract tests. They must not be presented as evidence of model quality, calibration quality, or readiness for automation.

The encoder-eval extra is an opt-in research lane. Run the default suite first;
CI must not acquire models or datasets. Candidate acquisition is a local,
operator-triggered action restricted to the reviewed registry. Do not weaken
the safetensors-only boundary or add request-triggered downloads.

The Phase 2C data-policy registry is metadata-only. Run
`uv run python -m benchmarks.data_policy_gate validate-registry` when changing
its policy rows. Do not add examples, dataset bytes, prompts, generated labels,
or artifact approvals to the registry. A future data artifact needs its own
reviewed manifest with immutable bytes, record provenance, privacy and rights
evidence, split controls, and the exact registry revision. The gate is an
engineering control, not legal advice or a dataset-quality claim.

Phase 3A keeps the first-party packet repository-only and offline. Do not add
JSONL records, split plans, prompts, generated examples, author mappings, or
weights. Run `uv run python -m benchmarks.first_party_gate validate-protocol`
and the complete validation commands from the Phase 3A spec when changing the
guide, manifest, or benchmark gate.
