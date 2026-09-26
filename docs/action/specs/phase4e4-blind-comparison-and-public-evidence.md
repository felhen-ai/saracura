---
title: Phase 4E.4 blind comparison and public evidence
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md
globalRef: qmd://saracura/docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-26
sourceRefs:
  - docs/action/specs/phase4e-saracura-owned-universal-checkpoint.md
  - docs/action/specs/phase4e3b-first-owned-checkpoint.md
  - docs/action/specs/phase4e3c-owned-runtime-registration.md
related:
  - AGENTS.md
  - README.md
  - docs/README.pt-BR.md
  - benchmarks/saracura_universal_corpus.py
  - benchmarks/saracura_universal_pilot.py
  - benchmarks/saracura_universal_training.py
  - src/saracura/backends/laya.py
  - src/saracura/backends/saracura_universal.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.4 blind comparison and public evidence

## Outcome

Produce the first decision-quality comparison of the Saracura-owned universal checkpoint against the separately installed Laya control, without allowing control results or comparison records to influence the candidate. Publish only sanitized aggregate evidence. The result is a research measurement, not a winner declaration, calibration claim, production claim, or automation authorization.

This increment closes Phase 4E only when the comparison set is independently generated after the training capsule was sealed, both runtimes receive the same accepted records in fresh processes, all unsupported outcomes remain in denominators, and the public artifact can be reproduced from a private result without exposing task text, local paths, provider identifiers, tensors, scores, or credentials.

## Immutable boundaries

- Saracura is the candidate. Laya is an optional external control and remains isolated behind the `universal-local` extra and its explicit snapshot.
- The comparison starts from the already sealed Phase 4E.3 training capsule. It may not modify weights, renderer, tokenizer, thresholds, manifests, or runtime registration.
- Comparison records are synthetic-only, privately stored outside the repository and outside synchronized Felhen roots.
- Provider access requires literal network opt-in and `OPENROUTER_API_KEY` from the authorized secret runtime. Credentials never enter files, command arguments, reports, logs, or exceptions.
- Cost control is report-only. Generation does not stop at a monetary threshold. Durable cost telemetry emits a progress event whenever settled provider-reported cost crosses another USD 10 interval and reports the final total.
- No real mailbox data, personal data, customer data, Felhen artifact, AIOS dependency, or private operating detail enters this phase.
- The public repository and built wheel contain no accepted task text, model snapshot, checkpoint, embedding, provider payload, response identifier, reservation identifier, or private absolute path.

## Verified implementation context

The planning pass inspected the current source before critique. These are implementation anchors, not assumptions:

- `benchmarks/saracura_universal_corpus.py` around lines 867-980 owns the deterministic training-plan shape; around lines 1280-1600 owns the provider client and durable ledger; around lines 2240-2420 owns semantic fingerprints, normalized duplicate detection, and acceptance assembly; and around lines 2780-3020 owns accepted-packet validation and coverage accounting.
- `benchmarks/saracura_universal_corpus.py` around lines 54-71 defines the only closed difficulty axes: `explicitness`, `negation`, `distractor_overlap`, and `urgency`.
- `benchmarks/saracura_universal_pilot.py` around lines 90-240 owns the current blind reviewer wire schema and binding; around lines 900-1760 owns resumable batch resolution and diagnostics; and `run_pilot` around line 2142 proves the current explicit-network, verified-tokenizer, report-only ledger, Azure-only provider, five-attempt recovery path that this phase must reuse rather than fork semantically.
- `benchmarks/phase4e_pipeline.py` around lines 1815-2165 owns the proven three-consecutive-uncertainty circuit, no-replay resolution, incomplete operational states, USD 10 progress telemetry, and terminal failure-report path used by the successful corpus recovery.
- `benchmarks/saracura_universal_training.py` around line 820 owns full accepted-packet binding after holdout release, while `verify_training_capsule` around line 3727 owns the sealed training manifest and checkpoint verification path.
- `src/saracura/backends/laya.py` around lines 98-160 owns the Laya renderer and its 1,024/256-token envelope, marker positions, special-token rejection, and length framing; around lines 361-605 it owns explicit snapshot construction and universal scoring; its `score_universal_choice` entry point begins around line 543.
- `src/saracura/universal/rendering.py` around lines 1-100 owns the Saracura renderer and its 128-token context and 96-token criterion envelopes. `src/saracura/backends/saracura_universal.py` around lines 492-615 verifies the installed training capsule, while the backend begins around line 618 and its universal scoring entry point begins around line 803.
- `.github/workflows/ci.yml` around lines 104-128 owns the offline Phase 4E matrix and its default-install/local-ML separation.
- `pyproject.toml` around lines 60-66 packages only `src/saracura`; `.gitignore` and `benchmarks/validate_manifests.py` were inspected as the private-artifact exclusion and immutable-manifest paths that must be extended without changing the default-install boundary.

No existing module implements the Phase 4E.4 comparison. `benchmarks/universal_bakeoff.py` is intentionally a Phase 4C systems-only harness and cannot be extended into an accuracy or winner benchmark without violating its contract.

## Phase 4E.4A: frozen protocol and offline implementation

Add `benchmarks/saracura_universal_comparison.py` as an isolated module with explicit subcommands. Do not extend the Phase 4C systems-only bakeoff or make comparison dependencies part of the default runtime.

### `plan`

`plan` creates one canonical, no-clobber `phase4e-comparison-plan.v1` object with exactly 200 slots from the fixed seed `saracura-phase4e-comparison-v1`.

The plan must:

- allocate 100 paired decision families, one PT-BR and one English record per family;
- give both records in a family the same option count, gold position, scenario, and four difficulty-axis values; only locale-specific natural language may differ;
- cover every domain and every option count from 2 through 8 in both locales;
- deterministically balance domain, option count, gold position, scenario, and the exact closed axes `explicitness`, `negation`, `distractor_overlap`, and `urgency` as closely as integer allocation permits;
- use only `synthetic_comparison` identities and a comparison-specific task/family namespace;
- prove that task IDs and family IDs are disjoint from the validated accepted train/dev/holdout packet before sealing;
- bind the training packet receipt, sealed training manifest digest, source commit, policy digest, planner revision, and tokenizer-receipt digests without carrying private paths;
- precommit the common Saracura/Laya capacity target, the primary comparative device, and the public underpowered threshold of fewer than 10 terminal rows per slice before provider calls; and
- contain no natural-language task content.

The plan is immutable once any provider journal entry exists. Resume accepts only a byte-identical plan.

### `generate`

`generate` consumes the sealed plan, the full validated accepted training packet, both explicit verified tokenizer snapshots, and a private work root. It requires literal `--allow-network` and uses the current Phase 4E author/reviewer wire protocol:

- author: `openai/gpt-4.1` through Azure only;
- reviewer: `openai/gpt-4.1-mini` through Azure only;
- fallbacks disabled, provider parameters required, data collection denied, ZDR required;
- reviewer temperature zero and no reviewer reasoning extension;
- at most five attempts per provider request;
- reviewer blind to author answer, gold position, sibling, pair ID, planner target, split, and author attestation; and
- author and reviewer stages separately journaled with request/response digests and settled cost.

The generator reuses the closed Phase 4E task and review schemas but writes comparison-specific journals, diagnostics, resolutions, and packet descriptors. It constructs one canonical semantic task and projects it into two `DecisionRequest` objects. Locale, domain, state, question ID, instruction, criterion IDs, criterion order, and criterion descriptions must be byte-identical across projections. Only the workflow and immutable model identity may differ. The parent verifies that invariant.

Capacity acceptance must execute each backend's actual source-of-truth renderer with its explicit verified tokenizer: `render_choice` plus `validate_rendered_capacity` for Saracura, and `render_laya_choice` for Laya. A shared approximate counter is forbidden. Capacity failure, Laya tokenizer-special-token rejection, privacy failure, schema failure, review disagreement, normalized duplicate, semantic duplicate, training-content overlap, or family overlap rejects the record rather than replacing it after outcome inspection.

Transport uncertainty is not a content rejection. Every uncertain call is resolved conservatively without replay, but three consecutive uncertain provider calls open the same fail-closed circuit used by the Phase 4E recovery path. Generation terminal states are closed and distinct:

- `passed`: all generation and evidence minimums pass;
- `insufficient_comparison_evidence`: all 200 identities resolve without an operational failure, but content-quality acceptance minimums fail; and
- `inconclusive_transport` or `inconclusive_operational`: transport circuit or another operational failure prevents a quality conclusion.

The comparison module maps the recovery pipeline's existing `incomplete_transport_circuit` and `incomplete_operational` evidence to the public protocol names `inconclusive_transport` and `inconclusive_operational`; it does not change their semantics. Nonconsecutive uncertain calls remain explicit transport resolutions even when a later settled call resets the consecutive circuit streak. If any content minimum fails and any comparison identity resolved as `author_transport_uncertain` or `reviewer_transport_uncertain`, the terminal state is `inconclusive_transport`, never `insufficient_comparison_evidence`. Thus transport loss cannot be reported as quality insufficiency or release the next phase.

Only `passed` may start scoring. `insufficient_comparison_evidence` closes the synthetic-comparison attempt and may release Phase 4F with that limitation. An inconclusive state does not close Phase 4E and does not release Phase 4F. Every terminal state emits a create-only failure or success report with resolved/unresolved, uncertainty streak, total and nonconsecutive transport-uncertainty counts, provider-call, and cost counts.

The accepted packet must contain at least 150 tasks and preserve:

- both locales with at least 60 accepted tasks each;
- all 12 domains with at least 5 accepted tasks each;
- every option count from 2 through 8 with at least 10 accepted tasks each;
- at least 60 complete PT-BR/English family pairs; and
- zero exact, semantic-fingerprint, or normalized-near-duplicate overlap with accepted train/dev/holdout content.

If any content minimum fails after all 200 identities resolve operationally, generation seals `insufficient_comparison_evidence`; scoring cannot start. Rejected and unsupported records remain accounted for in private terminal evidence.

### `score-backend`

`score-backend` is an internal worker entry point launched by the evaluator in a fresh process. It loads exactly one backend, reads the same descriptor-bound accepted comparison packet, performs an explicit pre-tokenization capacity check, and records a private terminal row for every accepted task.

The Saracura worker receives only its verified encoder snapshot, sealed training capsule, and explicit device. The Laya worker receives only its verified Laya snapshot and explicit device. A worker may not discover snapshots, caches, or credentials implicitly. It may not read the other backend's outputs.

Every task result is one of `correct`, `incorrect`, `unsupported_capacity`, `unsupported_runtime`, or `runtime_error`. Every status remains in the total denominator; only `correct` contributes to accuracy. The worker records selected criterion, stable error code, token counts, warm latency, and deterministic-repeat status privately. It never records raw logits or calibrated confidence.

Each worker also records cold process/load time, peak RSS, artifact byte size, throughput, runtime/model digests, dependency versions, platform, and device. Determinism is checked by repeating every task once in the same worker and requiring the same terminal status and selected criterion. Only the first scored pass contributes warm latency and throughput; the second contributes only determinism evidence. Performance measures are descriptive and are never mixed across devices.

### `evaluate`

`evaluate` validates the accepted comparison packet, training capsule, source commit, and explicit snapshots before spawning workers. It launches Saracura first and Laya second in separate fresh subprocesses. If Laya is not supplied or cannot be verified, Saracura measurement may complete but the comparison result is explicitly `control_unavailable`; the missing control is never silently replaced.

The result has one precommitted primary comparison cell: Saracura CPU versus Laya CPU, with identical thread count and the same accepted packet. The parent may also launch one Saracura-only MPS worker as a separate descriptive device observation. The MPS observation is never paired with the CPU control, never changes the primary result, and never receives `control_unavailable`; it is omitted or reported as a separate unavailable descriptive cell. Devices are forbidden from mixing within a cell.

The parent binds worker results by task ID and refuses duplicates, omissions, changed labels, changed option order, changed canonical semantic fields, or mixed devices within a cell. Workers receive a closed explicit input manifest and one private output directory; the Laya worker manifest contains neither the Saracura output path nor its content. The parent seals the complete Saracura CPU output, makes it read-only, and records its digest before starting Laya. A worker-isolation test asserts that the control subprocess arguments, environment, and manifest contain no candidate-output path. It produces one private `phase4e-comparison-result.v1` with a `primary_comparison` object and zero or one `candidate_device_observations` object containing:

- planned, resolved, accepted, rejected, unsupported, and failed counts;
- overall and sliced accuracy for locale, domain, scenario, option count, and the exact four closed difficulty axes;
- two-sided 95% Wilson intervals for every reported accuracy cell;
- paired candidate/control outcomes where both terminal rows exist;
- cold load, p50/p95 warm latency, throughput, peak RSS, and artifact size by backend;
- deterministic-repeat failures;
- incomparable cells and every slice with fewer than 10 terminal rows marked `underpowered: true`; underpowered cells still publish their denominator, correct count, accuracy, and interval and are never hidden or zero-filled;
- all source, plan, packet, policy, tokenizer, checkpoint, training-manifest, runtime, and result digests; and
- the final provider-reported generation cost as telemetry, not an authorization gate.

No difference is statistically promoted to a superiority claim. A lower Saracura result is preserved unchanged and cannot trigger tuning within the same checkpoint revision.

### `publish`

`publish` is the only command allowed to write a tracked comparison artifact. It consumes the verified private result and creates `benchmarks/results/phase4e-comparison-v1.json` followed by `docs/action/phase4e-comparison-result.md` under the ordered create-only protocol below.

The public projection contains only aggregate counts, metrics, Wilson intervals, performance summaries, device/platform classes, immutable digests, protocol limitations, and the final status. It explicitly states that author/reviewer models and prompt family are shared with training generation and therefore make this synthetic comparison in-distribution for Saracura but not necessarily for Laya. It rejects any value containing task text, state, criterion descriptions, selected criterion IDs, provider request/response/reservation IDs, credentials, tensor values, score vectors, usernames, home directories, volume names, or absolute paths.

The public limitations also state that the shared comparison set is restricted to Saracura's materially smaller 128-context/96-criterion token envelope, rather than exercising Laya's larger 1,024/256 envelope. The primary comparison may publish the aggregate paired 2x2 correctness table (`both_correct`, `candidate_only_correct`, `control_only_correct`, `both_not_correct`) as descriptive evidence, without a superiority test.

Publication is create-only and ordered, not transactionally misrepresented. The canonical JSON is written and verified first. The Markdown is deterministically derived from the exact JSON bytes and binds their digest. Retry accepts an existing byte-identical file and refuses different bytes. If Markdown publication fails after JSON publication, the JSON remains the canonical complete result and retry may create only its matching Markdown projection.

When no completed live result is supplied, the repository may ship the implementation and schema but must not fabricate a result artifact. Fixture-generated results remain test-only and cannot be published under the live result names. `control_unavailable` is an evaluation-time state only: generation still requires both verified tokenizers so that accepted records fit the precommitted intersection. `control_unavailable` does not close Phase 4E or release Phase 4F. A later evaluation with the control available must reuse the already sealed Saracura CPU output by digest; it may not rerun or choose between candidate outputs. `insufficient_comparison_evidence` produces no `phase4e-comparison-v1.json`; the `publish-status` subcommand consumes the verified terminal generation report and updates both README status sections with only aggregate counts and the explicit absence of a scored comparison, under the same sanitizer and create-or-byte-identical rules.

## Phase 4E.4B: live blind execution

After implementation review and offline QA:

1. Create the private plan against the validated training packet and sealed training capsule.
2. Generate and independently review the 200 planned slots using the authorized provider route.
3. Stop with `insufficient_comparison_evidence` only when all identities resolved operationally and the acceptance minimums do not pass; stop as inconclusive on transport circuit or operational failure.
4. Run the primary Saracura CPU measurement, seal its digest, and then run the Laya CPU control with the same explicit thread count. Optionally run Saracura MPS as a separate descriptive observation; never merge it into the primary comparison.
5. Run the Laya control only after the Saracura candidate output is terminal. Laya output cannot feed any candidate code, artifact, or decision.
6. Seal the private result, project the sanitized public evidence, and scan the diff for private artifacts and paths.
7. Update English and PT-BR README status consistently and link the public result without claiming optimization, calibration, production readiness, automation readiness, or superiority.

Live provider spend is reported to the operator each time cumulative new Phase 4E.4 provider-reported cost crosses USD 10 and again at completion. There is no hard financial stop in this protocol.

The real comparison runs from one checkout environment synced with both `local-minilm` and `universal-local`; optional imports remain delayed until their worker starts. CI keeps default-install checks in `phase4e-offline`, runs Saracura fixture comparison tests with the local MiniLM extra there, and runs Laya fixture coverage in the existing `universal-local` job. Tests are split by marker or backend-specific fixture so neither CI invocation requires the other optional extra; dependency absence is an asserted skip only for the unrelated backend, never for the backend owned by that job.

All live paths derive from the existing configured `SARACURA_PRIVATE_STATE_ROOT` contract and the current private-root validators. Public code must not contain a Felhen-specific absolute path. A valid configured root must be local, restrictive, outside the checkout and synchronized roots, and descriptor-bound before any content write.

## Files and ownership

Expected implementation scope:

- `benchmarks/saracura_universal_comparison.py`
- `benchmarks/manifests/phase4e-comparison-policy.v1.json`
- `benchmarks/validate_manifests.py`
- `tests/test_phase4e_comparison.py`
- `.github/workflows/ci.yml`
- `.gitignore`
- `pyproject.toml`
- `README.md`
- `docs/README.pt-BR.md`
- `docs/action/phase4e-comparison-result.md` only after a valid live result exists
- `benchmarks/results/phase4e-comparison-v1.json` only after a valid live result exists

Changes to runtime scoring, checkpoint tensors, training policy, corpus schemas, or Laya architecture are out of scope. A necessary correction there requires a new reviewed spec before implementation.

## Verification

Offline verification must include:

```bash
uv run pytest -q tests/test_phase4e_comparison.py
uv run pytest -q tests/test_phase4e.py tests/test_phase4e_corpus.py tests/test_phase4e_pipeline.py tests/test_phase4e_pilot.py
uv run --extra local-minilm pytest -q tests/test_phase4e_training.py tests/test_phase4e_comparison.py
uv run --extra universal-local pytest -q tests/test_universal_local.py tests/test_phase4e_comparison.py
uv run python -m benchmarks.validate_manifests
uv run python benchmarks/validate_default_environment.py
uv build
```

The CI `phase4e-offline` lane must run the comparison tests without network, model downloads, private artifacts, or device discovery. Tests use synthetic fixtures and injected transports only.

Before publication, run the repository's full test suite, build the wheel, inspect its contents, validate both public result formats, search the Git diff for secrets/private paths/provider identifiers/raw task fields, and exercise the installed standard `saracura` CLI from the wheel in an isolated environment. The comparison CLI remains checkout-only because `pyproject.toml` intentionally excludes `benchmarks/` from the wheel.

## Acceptance criteria

1. A canonical 200-slot plan is deterministic, text-free, balanced, no-clobber, and cryptographically bound to the sealed Phase 4E.3 evidence.
2. Tests prove task/family/content isolation from accepted train/dev/holdout rows using exact IDs, semantic fingerprints, and normalized near-duplicate scans.
3. Generation requires explicit network opt-in, explicit tokenizers, Azure-only ZDR provider policy, blind review, durable retry/journal state, no replay of uncertain calls, a three-uncertainty circuit, report-only costs, and USD 10 progress reporting. A test proves that nonconsecutive uncertain calls which cause a content minimum to fail seal `inconclusive_transport`, not `insufficient_comparison_evidence`.
4. At least 150 accepted tasks, 60 per locale, all domains, all option counts, and 60 complete locale pairs are required before scoring.
5. Capacity acceptance uses the real renderer and verified tokenizer for each backend over byte-identical semantic fields; only workflow and model identity differ.
6. Saracura and Laya run from explicit verified artifacts in separate fresh processes and cannot read each other's outputs.
7. Capacity and runtime failures remain in denominators and are reported by stable status; no record is removed after either backend outcome.
8. Candidate CPU scoring is terminal and digest-sealed before control CPU scoring, and no control result changes the candidate checkpoint, code, plan, records, or outputs.
9. The primary comparison is CPU versus CPU. Optional Saracura MPS evidence remains a separate descriptive observation and cannot be aggregated into or substituted for the primary cell.
10. Accuracy is reported overall and by locale, domain, scenario, exact closed difficulty axes, and option count with two-sided 95% Wilson intervals; every slice below the precommitted 10-row threshold remains visible and marked underpowered.
11. Cold load, warm latency, throughput, peak RSS, artifact size, device, and determinism are reported separately by backend and device.
12. Public projection is create-only, aggregate-only, digest-bound, bilingual in status documentation, and mechanically rejects raw content, private paths, provider lineage, secrets, tensors, and score vectors.
13. Default install/import/help and invalid-input paths remain offline and do not import optional ML dependencies.
14. The wheel excludes comparison records, work journals, provider payloads, private results, snapshots, and checkpoints.
15. No public wording claims that Saracura is better, PT-BR optimized, calibrated, production-ready, or safe for automation from this synthetic comparison.
16. A lower completed Saracura result is published faithfully and never repaired by tuning the same checkpoint revision. An operationally inconclusive run is disclosed as incomplete in status documentation and cannot produce the live aggregate result artifact.

## Gate to Phase 4F

Phase 4F may begin only after the implementation passes review and QA and the live comparison is either fully scored with both backends or ends as `insufficient_comparison_evidence` after all 200 identities resolved without any transport or operational failure. `control_unavailable`, `inconclusive_transport`, and `inconclusive_operational` do not release Phase 4F. An inconclusive attempt cannot be retried under this immutable plan: a later attempt requires a reviewed plan revision with a new seed and namespace plus task, family, semantic-fingerprint, and normalized-content disjunction from every accepted record of the earlier attempt. Public documentation must accurately reflect the result. Phase 4F is a separate read-only e-mail shadow pilot: it may classify messages and collect operator corrections, but it cannot delete, archive, move, reply, forward, mark read, or train on private mail without a separately reviewed policy and explicit authorization.

## Critique record

Round 1 returned `NO-GO` with four blockers. Each was verified in the worktree before revision:

- Transport versus quality: `docs/action/phase4e-execution.md` around lines 296-315 records 81 uncertain calls in the first corpus run; `benchmarks/phase4e_pipeline.py` around lines 1815-2165 provides the proven circuit and incomplete states. The spec now distinguishes insufficient content evidence from inconclusive operations, and only the former can release Phase 4F.
- Capacity parity: `src/saracura/backends/laya.py` around lines 98-160 and `src/saracura/universal/rendering.py` around lines 1-100 implement different real envelopes. The spec now requires both source renderers and a byte-identical semantic projection.
- Difficulty precommitment: `benchmarks/saracura_universal_corpus.py` around lines 54-71 has four closed axes and no `state_length`. The spec now uses those exact axes and fixes the public underpowered threshold at 10 rows.
- Device topology: the prior text simultaneously required CPU and MPS while forbidding mixed devices in one result. The spec now fixes CPU-versus-CPU as the primary cell and treats Saracura MPS as a separate descriptive observation.

The round's non-blocking improvements were also verified and incorporated where they closed ambiguity: ordered create-only publication, candidate-output digest sealing, in-distribution limitation, evaluation-only `control_unavailable`, checkout-only comparison CLI, configured private root, and no quadratic-scan performance claim without QA measurement.

Round 2 kept one round-1 blocker open. Inspection of `benchmarks/phase4e_pipeline.py` around lines 1350-1481 confirmed that uncertain author or reviewer calls conservatively reject identities, while `_v4_streak` around lines 1670-1681 resets on later settled calls. The spec now makes any minimum failure with even nonconsecutive transport-uncertain resolutions `inconclusive_transport` and requires a regression test for that exact case. It also records the existing-state mapping and closes the round's non-blocking ambiguities around `control_unavailable`, insufficient-evidence publication, shared family attributes, CI extras, capacity-range bias, and paired descriptive outcomes.

Round 3 returned `GO` with no blockers. Its non-blocking findings were checked against the already inspected renderer, packaging, and execution paths. The final spec explicitly classifies Laya special-token rejection, defines retry as a new reviewed plan, reuses a sealed candidate output after temporary control unavailability, assigns README-only insufficient evidence to `publish-status`, measures performance only on the first pass, and makes worker path isolation testable. Private results and journals remain outside the checkout; QA still inspects wheel and source distributions rather than relying on filename exclusions.
