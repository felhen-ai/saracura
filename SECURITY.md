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
lastReviewedAt: 2026-09-21
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

## Supported versions

Saracura is a research-only alpha. Security fixes apply only to the latest commit on the default branch until a release policy is published.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting for this repository. If that channel is unavailable, contact the repository owner through the private contact method listed on the GitHub organization profile. Do not include secrets, private datasets, or sensitive model inputs in an initial report.

## Current trust boundary

- There is no telemetry or network fallback.
- Requests cannot select arbitrary local model paths or trigger downloads.
- The default installation has no model, tokenizer, dataset-loader, or remote plugin runtime.
- The fixture backend is deterministic test code, not a security sandbox or a decision model.
- State and criteria are untrusted data and are never instructions to execute tools.
- Pydantic contracts reject unknown fields and bound questions and criteria.

Future model integrations must pin immutable revisions and hashes, use safe weight formats, keep `trust_remote_code=False`, and review tokenizer code, dataset loaders, and plugins independently. Safetensors alone does not make the rest of the supply chain safe.

Future serving work must address byte and token limits, `Q × K`, cache capacity, concurrency, authentication, and loopback-only defaults before exposing a network listener. Process isolation must not be described as a security sandbox.
