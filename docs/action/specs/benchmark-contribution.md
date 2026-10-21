---
title: Reproducible benchmark contribution
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/benchmark-contribution.md
globalRef: qmd://saracura/docs/action/specs/benchmark-contribution.md
reviewCadenceDays: 90
lastReviewedAt: 2026-10-21
sourceRefs:
  - research/ptbr_bench.py
related:
  - research/README.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Reproducible benchmark contribution

Preserve the existing seven-task protocol while making evaluation and manual submission reproducible. No training, new leaderboard, paid compute or private data.

## C1

A stdlib-only evaluator preserves all seven-task scoring, seed, option shuffling and 3500-character protocol while recording resolved immutable dataset SHA and caller-declared checkpoint revision/hardware/quantization/server version, protocol/evaluator identity and timing; absent metadata explicitly unknown, never invented.

## C2

--limit runs are explicitly smoke/partial and never submission-ready; full runs need all seven complete task sizes, no failures, nonempty valid data and immutable model provenance to be submission-ready; malformed outputs or transport failures count wrong with bounded diagnostics, and report is saved for failures. Invalid numeric CLI inputs rejected early.

## C3

Submission artifact has per-item predictions and sufficient protocol/provenance to independently recalculate scores; provide stdlib validator that rejects partial/malformed/inconsistent reports, recomputes score rather than trusting summary, flags unknown metadata, no arbitrary code execution or overwriting published results. Keep existing results readable; new validator applies new schema.

## C4

Simple bilingual contribution guide covers install/serve actual listed CPU-friendly upstream model, smoke/full commands, metadata, output interpretation, failures/certificate issues, safe submission with no private prompts or credentials, and public/private track/manual review plus lm-eval protocol separation. Link from README, PTBR README and research README; a minimal issue template collects model/revision/runtime/report evidence.

## C5

Meaningful network-free regression tests cover complete vs partial report, dataset SHA resolution, failure/malformed answers, invalid CLI, score/provenance tampering and validator behavior; required repo checks pass without ML dependencies. No weights or real dataset fixtures committed.

## C6

Provide version-controlled small source for leaderboard contribution instructions or update script only if needed, avoiding forked whole Space; integrate with existing Space after main merge and verified read-back. Do not claim model live run until root supplies evidence.

## Critique resolutions

Validator fetches public dataset at resolved SHA (or verified local cached bytes checked against downloaded source); compares seven task counts and ordered gold labels against source, recomputes scores, validates prediction types/options per row. No report-supplied gold as authority. No cryptographic proof of inference is claimed.

Immutable checkpoint provenance accepts 40-hex HF commit or sha256:<64hex> artifact digest only; main/latest/missing do not qualify. model subfolder and context may be declared. All fields remain declarations, not verified running identity.

Root owns live CPU integration after executor. Executor implements rest C4 with serving recipe explicitly pending until root supplies evidence; final acceptance and publishing wait for real test. If full test impossible, report blocker, do not publish unsupported working recipe.

Evaluator never stores endpoint URLs, prompts, raw HTTP bodies or credentials; use safe basename names, exclusive output creation before expensive work, self-file SHA256; retain legacy items/tasks/mean fields additively. Record task-local errors as bounded codes.

Capture backup HF instruction bytes and hashes before mutation; root has saved current Space index/README/results under local cache. No existing ranking data modification. Smoke option order not directly comparable due existing shared RNG, document it.
