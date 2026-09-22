---
title: Saracura security policy
kind: policy
area: platform
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: SECURITY.md
globalRef: qmd://saracura/SECURITY.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-22
sourceRefs: []
related:
  - README.md
  - AGENTS.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Security policy

> **Português (Brasil):** Não abra uma issue pública para relatar uma vulnerabilidade. Use o canal privado de vulnerability reporting do GitHub e não inclua segredos, datasets privados ou entradas sensíveis de modelos no relato inicial. O Saracura ainda é um alpha de pesquisa; somente o commit mais recente da branch padrão recebe correções de segurança até que uma política de releases seja publicada.

## Supported versions

Saracura is a research-only alpha. Security fixes apply only to the latest commit on the default branch until a release policy is published.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting for this repository. If that channel is unavailable, contact the repository owner through the private contact method listed on the GitHub organization profile. Do not include secrets, private datasets, or sensitive model inputs in an initial report.

## Current trust boundary

The optional Phase 4A MiniLM backend accepts only explicit operator paths to a
reviewed snapshot, sealed synthetic training manifest, and safetensors head. It
opens them by descriptor with no-follow, ownership, mode, link, size, digest,
and mutation checks; it never passes paths to a hub or auto-loader. BERT and
the fast tokenizer are constructed from verified bytes, dynamic code and pickle
formats are rejected, and the fixed public conformance vector must pass before
request data is accepted. The Phase 4A identity calibration is an immutable
no-fit transform with a separate synthetic identity dataset profile. Its
fixture_only result does not express calibration quality and cannot authorize
automation.

- There is no telemetry or network fallback.
- Requests cannot select arbitrary local model paths or trigger downloads.
- The default installation has no model, tokenizer, dataset-loader, or remote plugin runtime.
- The fixture backend is deterministic test code, not a security sandbox or a decision model.
- State and criteria are untrusted data and are never instructions to execute tools.
- Pydantic contracts reject unknown fields and bound questions and criteria.
- The optional Phase 2B acquisition lane is operator-triggered only. It accepts
  reviewed candidate IDs from the bundled registry, uses immutable revisions and
  safetensors hashes, and never accepts model URLs, paths, file lists, or
  credentials as acquisition input. Redirect hosts, file types, size, free
  space, hashes, and atomic promotion are checked before a snapshot is readable.
- The optional loader is local-only, uses `trust_remote_code=False` and
  `use_safetensors=True`, and is not registered as a Saracura decision backend.
- The Phase 2C source-policy registry is metadata-only and fail-closed. It
  contains no examples or approved artifact bytes, requires immutable
  third-party revisions, rejects credential-bearing or mutable URLs, and
  cannot authorize private, customer, model-assisted, or unvetted-public text.
  Validation is offline and does not import a dataset loader. A later artifact
  manifest must prove record-level privacy, rights, provenance, and takedown
  controls before any training process can read data.

Future model integrations must pin immutable revisions and hashes, use safe weight formats, keep `trust_remote_code=False`, and review tokenizer code, dataset loaders, and plugins independently. Safetensors alone does not make the rest of the supply chain safe.

Future serving work must address byte and token limits, `Q × K`, cache capacity, concurrency, authentication, and loopback-only defaults before exposing a network listener. Process isolation must not be described as a security sandbox.
