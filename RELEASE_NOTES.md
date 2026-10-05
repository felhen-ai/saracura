---
title: Saracura 0.1.0a2 experimental preview
kind: release
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: RELEASE_NOTES.md
globalRef: qmd://saracura/RELEASE_NOTES.md
reviewCadenceDays: 30
lastReviewedAt: 2026-10-05
sourceRefs:
  - benchmarks/results/phase5c-julia1-cpu-smoke.json
related:
  - README.md
  - docs/README.pt-BR.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Saracura PT-BR v0.1 (model and benchmark, 2026-10-05)

Published on the Hugging Face Hub: the PT-BR typed-decisions benchmark (7 tasks, human labels, permissive sources)
and the Saracura PT-BR v0.1 checkpoint, a Laya multilingual fine-tune at 68.5% mean balanced accuracy on the
benchmark and 86.8% to 89.6% agreement with an open-weight teacher on unseen questions. Details in the model and
dataset cards and in ADR 0005. The package release below predates this direction.

# Saracura 0.1.0a2

This experimental preview makes an open decision model directly usable through
Saracura. The recommended first backend is the pinned Apache-2.0 Julia-1
checkpoint, running locally on CPU with PT-BR and English Choice examples.

The clean-wheel smoke completed ten requests with zero rejection: five PT-BR
and five English. On an Apple M4 Pro CPU, the first request took about 3.2
seconds and warm requests had a median latency of about 13 ms. This measures
runtime interoperability, not decision accuracy.

The release also includes an experimental loopback-only adapter for compatible
System One servers. Both paths remain uncalibrated, abstained, research-only,
and unable to authorize automation.

This release does not satisfy Gate A, claim production readiness, or claim that
Saracura trained Julia-1. Julia's published Portuguese evaluation is pt-PT;
Saracura has not yet established PT-BR quality on a representative held-out
set.
