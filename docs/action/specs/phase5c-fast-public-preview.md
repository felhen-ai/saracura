---
title: Phase 5C fast public preview
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase5c-fast-public-preview.md
globalRef: qmd://saracura/docs/action/specs/phase5c-fast-public-preview.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-28
sourceRefs:
  - github:#40
  - https://api.typesafe.ai/openapi.json
  - https://github.com/jaredpalmer/kev/blob/main/kev/api.py
related:
  - README.md
  - docs/decisions/0002-open-model-runtime-and-readiness.md
  - docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md
  - docs/action/specs/phase5b-managed-cuda-systems-closure.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 5C fast public preview

## Outcome

Ship the smallest honest public experimental preview of Saracura around an existing
open decision checkpoint. A clean user can run one documented local command,
send a PT-BR or English Choice request through Saracura, and receive a typed
research-only response. The preview optimizes for time to a usable release;
larger human evaluation, Gate A, a Saracura-owned checkpoint, production hardening and
automation authorization follow after publication.

## Evidence correction

The published Phase 5B `reject_local` result does not establish that Kev-4B is
unusable. Saracura required the serialized probabilities to sum to one within
`1e-5`, while the TypeSafe OpenAPI describes them as summing approximately to
one and the pinned Kev serializer rounds every probability to four decimal
places. The public failure report records no observed residual. Phase 5C must
therefore preserve that historical run and describe four-decimal serialization
as a plausible, unverified explanation that motivates the corrected adapter
boundary. It must not rewrite the immutable report, reclassify the historical
run, claim that Kev passed the systems protocol, or use that report alone to
select a candidate.

## Product decision

The preview keeps the generic, explicit, loopback-only `/v1/systemone` adapter,
but uses the Apache-2.0 Julia-1 checkpoint as the recommended first runnable
backend. Julia-1 is loaded directly from an operator-supplied local snapshot;
Saracura does not download weights, start a server, manage a GPU lease or fall
back to a remote provider. The initial public lane supports Choice only and
runs on CPU so the release is reproducible on an ordinary development machine.

At the adapter boundary:

1. preserve and validate the raw serialized values;
2. require exact question and option identity and finite values in `[0, 1]`;
3. accept only a documented serialization residual bounded by
   `max(1e-6, option_count * 0.00005 + 1e-9)`;
4. renormalize accepted values to an exact internal distribution;
5. reject a larger residual;
6. retain the raw serialized values in the existing `raw_scores` response field
   and record residual/normalization evidence only in the sanitized live-smoke
   report, without changing the public response schema;
7. never present the result as calibrated by Saracura or automation-ready.

The bound follows four-decimal round-to-nearest error per option and is tighter
than a blanket `0.02` tolerance for ordinary small Choice schemas.

The installed engine contract gains optional `normalized_probabilities` and
`selected_choice` fields on `ScoredChoice`. Existing backends leave them absent and preserve the
current logits-to-softmax behavior. The System One adapter supplies both the raw
serialized mapping and the exact normalized mapping; the engine validates both,
uses the normalized mapping directly, and never applies softmax to probabilities.
Normalization uses `math.fsum` and is accepted when the final sum is within
`1e-12` of one. The upstream `choice` must have probability within `1e-12` of
the maximum normalized probability; on a rounded tie, that upstream choice is
preserved. Upstream `confidence` is range-validated but is not exposed as a
Saracura confidence claim.

The wire contract is the public TypeSafe OpenAPI shape as instantiated by the
pinned Kev source revision already recorded in the acquisition descriptor:
questions are keyed by ID; every Choice question contains exactly `type`,
`instructions` and `criteria`, where criteria maps option ID to description.
Request fields are
exactly `model`, `questions` and `state`; a success response requires `model`,
`answers` and `usage`, with only the known optional `latency_ms` extension. The
returned model name is recorded but need not equal the request alias, as the
OpenAPI permits alias resolution. Choice answers contain exactly `type`,
`choice`, `confidence` and `probabilities`. Unknown fields, duplicate JSON keys,
responses above 1 MiB, redirects and non-200 responses fail closed. Each call
has a finite 30-second timeout and sends exactly one question; no together-call
or request cache is claimed in this preview. Usage contains exactly non-negative
integer `input_tokens` and `output_tokens`. The request reuses the existing NFC
validation at the universal-backend boundary.

The CLI requires explicit model identity fields independent of the HTTP alias:
`--model-id`, `--model-revision` and `--checkpoint-sha256`. They populate the
existing `ModelReference`; the adapter never invents or infers checkpoint
identity from `/v1/models`. Authentication, when required, is read only from
`SARACURA_SYSTEMONE_API_KEY`; a bearer value is never accepted through argv,
included in errors, diagnostics or reports. The HTTP opener disables all proxy
handlers so loopback state and credentials cannot leave the host.

## Scope

### Included

- one optional loopback-only System One client backend for the exact
  `universal-choice@phase5c-systemone.v1` Choice workflow;
- one optional direct Julia-1 backend for the exact
  `universal-choice@phase5c-julia.v1` Choice workflow;
- explicit endpoint, model alias and bearer source supplied by the operator;
- closed request/response validation and bounded normalization;
- PT-BR and English Julia-1 synthetic examples;
- lightweight fake-server tests that exercise the real wire shapes without
  model downloads;
- one-command documentation for running Saracura against a separately started
  open checkpoint;
- public correction of the Phase 5B candidate-selection interpretation;
- ADR 0003 recording the installed experimental runtime effect without changing
  Gate A or the historical candidate disposition;
- package version and release notes suitable for a `0.1.0a2` experimental
  GitHub prerelease.

### Excluded

- bundled weights, request-triggered downloads or arbitrary internet endpoints;
- starting or supervising Kev, Decider, Laya or any other model process;
- copying the private VM/GPU/clinical arbiter into the public repository;
- calibration fitting, confidence thresholds or automation authorization;
- a Saracura-owned checkpoint or comparative superiority claim;
- Gmail, mailbox or other private adapters;
- a 300-500-row human benchmark before the preview.

## Acceptance

1. The adapter accepts only literal `http://127.0.0.1:<port>` endpoints, disables
   environment proxies, and never follows a redirect.
2. A four-decimal distribution inside the calculated rounding bound is accepted,
   preserved in diagnostics and normalized internally to sum to one.
3. A distribution outside that bound, with unknown options, non-finite values,
   changed question IDs or malformed answer types fails closed.
4. The public response remains uncalibrated, abstained and
   `automation_allowed=false` regardless of upstream confidence wording.
5. Ordinary installation stays lightweight and offline; model/client
   dependencies are optional and tests use only a local fake server.
6. README material gives a PT-BR-first quickstart and an English equivalent,
   clearly naming the third-party checkpoint and the research-only boundary.
7. The historical Phase 5B bytes remain unchanged, while current docs no longer
   use that run as evidence that Kev itself is invalid.
8. Repository lint, typing, tests, build and artifact inspection pass.

## Candidate and claim contract

ADR 0002 and candidate manifest v3 remain immutable historical inputs. This
increment adds ADR 0003 but does not create a candidate-manifest successor.
Kev remains `reject_local` for the frozen managed-systems protocol with no
candidate claims. The README and release notes may say that the direct Julia-1
backend completed an experimental local CPU smoke with the pinned public source
and checkpoint; this is not calibration, production readiness, or a comparative
superiority claim. The terms `usable_developer_preview`,
`third_party_checkpoint_supported`, `production_ready`, calibrated and
automation-ready remain forbidden; the readiness manifest stays unchanged.

## Falsifiable prerelease gate

The live smoke runs the exact README command against the pinned Julia-1 source
and checkpoint on CPU. It sends five synthetic PT-BR and five synthetic English requests,
each with one Choice question. It passes only when all ten return zero transport
or schema rejection, preserve exact question/option identity, satisfy the
per-question residual bound and produce exact normalized internal distributions.
The sanitized closed-schema report records only code commit, a deterministic
release-code digest over the tracked bytes of `src/saracura`, `pyproject.toml`,
`uv.lock`, `README.md`, `docs/README.pt-BR.md` and the two public request
examples `examples/ptbr-julia-request.json`,
`examples/en-julia-request.json`, `examples/ptbr-systemone-request.json`, and
`examples/en-systemone-request.json`, public checkpoint and source revisions,
hardware class, request counts, latency summary, maximum
absolute raw-sum residual and normalization count. It contains no endpoint,
hostname, username, process/service name, local path, payload, label, bearer or
private operator receipt; tests reject unknown fields and private-path patterns.

Clean installation means building the wheel, creating a new temporary virtual
environment, installing that wheel without model extras, running
`saracura --help`, and executing the documented PT-BR and English commands
against the repository's fake loopback server. Both responses must be closed,
research-only and automation-disabled. Failure of either gate blocks the GitHub
prerelease but not the merged experimental adapter.

The version bump to `0.1.0a2` precedes both smokes. The GitHub prerelease tag
must reproduce the report's release-code digest exactly; after the smoke, only
release notes outside both READMEs and sanitized evidence may change before
tagging. Any change to a digested path, including either quickstart or request
example, invalidates the gate and requires both smokes to run again. The digest
algorithm sorts tracked relative paths bytewise and hashes each UTF-8 path
length, path bytes, file length and file bytes with SHA-256 and unsigned 8-byte
big-endian length prefixes. Inputs are read from the Git tree being tested, not
the mutable index or worktree. The smoke report stores the exact ordered path
list. Tests assert that every `examples/` path referenced by an experimental
backend quickstart command in either README belongs to that list; untracked files never
enter it.

## Rollout

1. Implement and validate the adapter and evidence interpretation correction,
   update the repository policy boundary, and add a new exact dynamic workflow.
2. Run the ten-request live CPU smoke against the pinned Julia-1 checkpoint on
   the Mac; no managed GPU window is required.
3. Publish `0.1.0a2` as an experimental GitHub prerelease only after the real
   smoke and clean installation path pass. This release does not satisfy or
   weaken Gate A and does not claim `usable_developer_preview`.
4. After release, run the compact PT-BR/English bakeoff and select whether Kev,
   Decider or another open checkpoint becomes the recommended backend.

## Rollback

The adapter is optional and isolated. Before publication, revert its release
commit and documentation without touching historical evidence. After a GitHub
prerelease is published, mark it withdrawn in the release notes, delete no
immutable tag or artifact silently, and publish a corrective `0.1.0a3` when
needed. No PyPI publication is part of this increment.
