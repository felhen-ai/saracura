---
title: Phase 5D.1 position-invariant Julia inference
kind: spec
area: platform
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/phase5d1-position-invariant-julia.md
globalRef: qmd://saracura/docs/action/specs/phase5d1-position-invariant-julia.md
reviewCadenceDays: 30
lastReviewedAt: 2026-09-28
sourceRefs:
  - github:#42
related:
  - docs/action/specs/phase5d-native-ptbr-benchmark.md
  - docs/action/specs/phase5c-fast-public-preview.md
  - docs/decisions/0003-experimental-systemone-runtime-effect.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 5D.1: position-invariant Julia inference

## Objective

Ship the smallest runtime correction justified by the Phase 5D native PT-BR
benchmark. The pinned Julia-1 path scored `34.32%` planned top-1 accuracy, but
its accuracy fell from `58.51%` with the gold answer first to `16.13%` with the
gold answer fourth. A bounded exploratory run using all four cyclic rotations
and mean probability aggregation scored `43.16%` planned accuracy and produced
planned position accuracies between `40.86%` and `46.24%`.

This phase makes that strategy reproducible and opt-in. It does not train a
checkpoint, claim calibration, or treat the result as competitive with the
frozen lexical baseline (`71.05%`).

## Runtime contract

`JuliaBackend` gains an explicit `position_ensemble` constructor option whose
default is `False`. The existing default path and
`universal-choice@phase5c-julia.v1` behavior remain byte-for-byte compatible.
The opt-in strategy uses the distinct request workflow revision
`universal-choice@phase5d1-julia-cyclic-mean.v1`, so logs and response workflow
metadata do not misrepresent an ensemble as the single-pass path. Model identity
remains Julia-1; the strategy does not create a new checkpoint.

The mode and workflow bind in both directions. A default backend advertises and
accepts only `phase5c-julia.v1`; an ensemble backend advertises and accepts only
`phase5d1-julia-cyclic-mean.v1`. Each rejects the other revision. CLI
prevalidation derives its allowed dynamic workflow from the flag, so both
`flag + phase5c request` and `no flag + phase5d1 request` fail before model
loading. The runner maps `(julia, cyclic_mean)` to the Phase 5D.1 revision and
all other current Julia benchmark requests to Phase 5C.

When enabled for one Choice question with `N` criteria, the backend:

1. evaluates exactly `N` cyclic rotations of the criteria in their declared
   order;
2. keeps state, instruction, criterion IDs and criterion descriptions
   unchanged apart from criterion order;
3. maps every returned probability to the criterion's original position;
4. takes the arithmetic mean for each original criterion using `math.fsum` and
   normalizes the resulting vector once for floating-point closure;
5. chooses the highest normalized mean, resolving an exact tie by the request's original
   criterion order; and
6. returns the same post-normalization vector as both raw scores and normalized
   probabilities.

Each rotated Julia response must pass the existing strict response validator.
Any failed rotation fails the complete decision under the existing sanitized
error contract; partial ensembles are forbidden. The option creates no network
path, download, retry, fallback, cache, confidence value or calibration claim.

The CLI exposes the strategy only through
`--julia-position-ensemble`, valid with `--backend julia`. Supplying the flag to
another backend is a request error. Help text states that the strategy performs
one inference per criterion and therefore trades latency for reduced position
bias.

The ensemble accepts at most 20 criteria across the complete request. This
bounds one request to 20 Julia predictions even though the single-pass backend
continues to support its existing question and criterion envelope. Exceeding
the ensemble budget returns `CARDINALITY_EXCEEDED` at `/questions` before the
first prediction.

## Benchmark evidence

The native PT-BR runner accepts the closed strategy values `single_pass` and
`cyclic_mean`. `cyclic_mean` is valid only with backend `julia`; existing
commands default to `single_pass`. A new v2 aggregate
report adds only:

- `inference_strategy`;
- `inferences_per_valid_decision`; and
- the existing latency, throughput, coverage, accuracy and per-position fields.

The v1 report schema and the two committed Phase 5D reports remain valid and
immutable. The v2 schema is exactly
`phase5d-ptbr-faq-bacen-report.v2`; its canonical result filename is exactly
`phase5d1-ptbr-faq-bacen-julia-cyclic-mean-cpu.json`. The closed loader
dispatches by schema version. Validation enumerates the two existing v1 names
and this exact v2 name rather than relying on an open glob. Filename identity is
validated from `(schema_version, backend, inference_strategy)`, and
`inferences_per_valid_decision` is exactly `1` for `single_pass` or `4` for the
four-choice `cyclic_mean` report.

A new closed manifest
`benchmarks/manifests/phase5d1-position-ensemble.v1.json` binds the exact digest
of the Phase 5D manifest, Julia backend, `cyclic_mean` strategy, Phase 5D.1
workflow revision, complete cyclic rotations, mean-plus-numerical-closure
aggregation, 20-criterion request budget, v2 report schema and canonical
filename. It contains no dataset rows and changes no Phase 5D manifest byte.
The v2 report includes `position_ensemble_manifest_sha256`, and verification
requires the exact digest. The new manifest is also part of the v2 closed
release paths.

Historical report verification recalculates `release_code_sha256` from the Git
tree named by `report.code_commit`, over the same closed release paths, instead
of from the current checkout. The commit must exist and be an ancestor of the
checkout. This applies to v1 and v2, so later source changes neither invalidate
honest historical evidence nor permit a forged digest or unrelated commit.
Generation uses this same Git-tree function at `HEAD`; filesystem traversal is
not a second digest implementation.

A real `cyclic_mean` CPU run against the same pinned dataset and Julia snapshot
is committed. The runner refuses tracked, untracked or ignored release-path
changes before loading the model.
Raw rows, prompts and per-record scores remain outside the repository.

Documentation must show the quality/latency tradeoff against the existing
single-pass Julia result, chance and the frozen lexical baseline. It must call
the result an experimental inference strategy around a third-party checkpoint,
not a Saracura-owned model.
It must also say that the strategy was selected post-hoc from this same
development benchmark, has no held-out confirmation yet, and that cyclic mean
was the only aggregation evaluated in this exploratory cycle. Published numbers
must come from the committed v2 report, not the exploratory probe. Limitations
also note that rotating IDs with descriptions cannot separate position bias
from label-token preference.

The repository `AGENTS.md` current-product boundary is updated in the same PR to
name the second opt-in research-only Julia revision, its N-inference cost, and
its unchanged uncalibrated/non-production status.

Implementation updates every workflow consumer together:
`src/saracura/backends/julia.py`, `src/saracura/runtime/workflows.py`,
`src/saracura/cli.py`, `benchmarks/ptbr_native/protocol.py`, the English and
PT-BR README files, `AGENTS.md`, and their tests.

## Non-goals

- Training, fine-tuning, distillation, calibration or threshold selection.
- Combining Julia with the lexical baseline or selecting weights on the test
  set.
- Changing the default Julia result or replacing the existing workflow.
- Gate A or Gate B promotion, production readiness, automation authorization,
  or general PT-BR superiority claims.
- Supporting arbitrary permutations; the strategy is exactly the complete
  cyclic set.

## Acceptance criteria

1. Default Julia behavior remains one inference with identical validation,
   scores and selected choice.
2. Opt-in inference evaluates every cyclic rotation exactly once, restores
   scores to original criterion IDs, uses mean aggregation and original-order
   tie breaking, and rejects the whole request on any invalid rotation.
3. Unit tests cover two, four and twenty criteria; rotation order; score
   remapping; normalization; deterministic ties; failure atomicity; default
   compatibility; the four mode/workflow mismatch combinations; CLI flag
   validation; the 20-criterion request budget; and sanitized errors without
   importing Julia or model weights.
4. Benchmark tests preserve v1 compatibility and validate the closed v2
   strategy fields, exact filename discovery, metrics, provenance and report
   sanitization. They prove a later source edit does not invalidate a historical
   report while digest or commit tampering does.
5. A real pinned CPU report records the strategy's measured quality and systems
   cost under the unchanged Phase 5D plan.
6. English and PT-BR docs and the `AGENTS.md` product boundary state the measured
   improvement and remaining lexical gap, distinct workflow, N-inference cost,
   post-hoc limitation and third-party ownership without a calibration,
   readiness or superiority claim.
7. Ruff, formatting, mypy, the focused suite, the full suite, manifest
   validation, build and distribution inspection pass.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q tests/test_julia.py tests/test_ptbr_native.py
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
```

## Rollout and rollback

Land through one independently revertible PR. The option stays off by default.
The PR must use a merge commit rather than squash or rebase so committed report
code ancestry remains verifiable on `main`.
Rollback is a Git revert of the runtime option, benchmark v2 report and docs;
the pinned model and Phase 5D v1 evidence are unchanged.
