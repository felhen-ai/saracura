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
