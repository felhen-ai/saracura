---
title: Phase 4E.3a PT-BR transport recovery
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4e3a-v4-ptbr-transport-recovery.md
globalRef: qmd://saracura/docs/action/specs/phase4e3a-v4-ptbr-transport-recovery.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-24
sourceRefs:
  - docs/action/phase4e-execution.md
  - docs/action/specs/phase4e3a-post-pilot-corpus.md
related:
  - benchmarks/phase4e_pipeline.py
  - benchmarks/saracura_universal_corpus.py
  - benchmarks/saracura_universal_pilot.py
  - tests/test_phase4e_post_pilot.py
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4E.3a PT-BR transport recovery

## Decision and task identity

`task_id=saracura-phase4e3a-ptbr-recovery-v1`.

Recover the fixed PT-BR locale minimum with a new, disjoint 100-task PT-BR supplement while preserving every accepted and rejected corpus-v3 decision and never replaying its 81 uncertain provider calls. Build a new immutable combined packet only from verified corpus-v3 evidence plus verified supplemental evidence. Add a transport circuit breaker, terminal failure reports and scalable private evidence storage before making any new provider call.

This is a recovery of Phase 4E.3A, not training. Embedding extraction, checkpoint training, calibration, holdout release, runtime registration and quality claims remain blocked until the combined packet seals and passes the pre-holdout validator.

## Verified context

- Corpus-v3 completed all 1,600 planned identities in `.artifacts/phase4e/corpus-work-v47`. The final ledger is `ledger/ledger-8454.json`, SHA-256 `a257032653e8078ab5d03c62f68a77c943d39912400dd2384fb4c09d1b4429d2`, with 2,737 settled calls and 81 uncertain author calls. The immutable plan SHA-256 is `13ca8304cbf18c8bd5871b135b5c609740c952805d656003959155f13d7a3adc` (`docs/action/phase4e-execution.md`).
- The result accepted 1,457 rows: 848 PT-BR and 609 English. It passed the global, split, domain, cell, family and complete-pair gates but failed `locale_minimum` because PT-BR was 58.2% instead of 60%. Solving `(848+x)/(1457+x) >= 0.60` requires at least 66 additional accepted PT-BR rows. A 100-task supplement gives rejection margin without weakening the gate.
- Rejections were operationally concentrated: PT-BR had 80 `author_transport_uncertain` and 32 `review_disagreement`; English had 2 `author_transport_uncertain`, 16 `review_disagreement`, 12 `semantic_duplicate` and 1 `near_duplicate`. The next plan must not regenerate or reinterpret any of those identities.
- The current runner continues after every uncertain call and seals before writing the report. Consequently two outage bursts produced 80 uncertain author calls, and the minimum failure prevented both packet and report creation (`benchmarks/phase4e_pipeline.py:1267-1399`).
- `_run_post_pilot_batch` already closes the exact task set of an uncertain call as `author_transport_uncertain`, records diagnostic lineage and returns without retry (`benchmarks/phase4e_pipeline.py:1070-1125`). `_require_post_pilot_uncertain_resolutions` proves every uncertain ledger entry is bound to exact rejected rows (`benchmarks/phase4e_pipeline.py:926-975`).
- `build_post_pilot_plan` and `_post_pilot_author_batches` deterministically bind the 1,600 v3 slots and interleave split, locale and cardinality cells (`benchmarks/saracura_universal_corpus.py:473-582`). `validate_plan` currently accepts only exact v2 or v3 plans, and `_validate_plan_structure` requires exactly 1,600 slots and 300 pairs (`benchmarks/saracura_universal_corpus.py:652-700`).
- `_minimums` applies the existing fixed global, split, domain, cell, family, locale and cross-locale gates and is the source of the observed `locale_minimum` result (`benchmarks/saracura_universal_corpus.py:2291-2339`).
- `seal_packet` writes a sibling staging directory, runs the pre-holdout validator and publishes with an exclusive same-volume rename; it deletes staging and leaves the destination absent on validation failure (`benchmarks/saracura_universal_corpus.py:2718-2796`). The pre-holdout validator reads train/dev plus holdout identities but does not open the holdout payload (`benchmarks/saracura_universal_corpus.py:2607-2683`).
- Durable research scanning currently discovers every `ledger-*.json` directory, validates its chain and inventories every research file (`benchmarks/saracura_universal_pilot.py:802-1108`). Corpus-v3 has 8,455 cumulative ledger snapshots occupying 8.9 GB; `resolved` and `diagnostics` together occupy about 11.3 MB and the final ledger is about 2.1 MB. `_validate_ledger_chain` now streams snapshots one transition at a time (`benchmarks/saracura_universal_pilot.py:880-896`), but copying the raw chain to the internal platform state root exhausted that volume.
- `canonical_research_ledger_root` always resolves through `platformdirs`, while `_require_post_pilot_artifact_paths` rejects any other artifact root (`benchmarks/saracura_universal_pilot.py:1133-1140`; `benchmarks/phase4e_pipeline.py:126-151`). This prevents selecting a private external state volume even when it is the safer durable location.
- The existing tests prove completed work can re-enter seal/report code without transport and that missing credentials create no work, packet, report or research root (`tests/test_phase4e_post_pilot.py:295-378`). They do not cover minimum failure reports, consecutive-uncertainty stopping, supplemental composition or compact research capsules.

## Scope

1. Add a deterministic `phase4e-universal-plan.v4` recovery plan. It contains the exact 1,600 v3 base slots plus 100 new PT-BR-only slots generated from seed `saracura-phase4e-ptbr-recovery-v1`. It binds the base plan SHA-256, base final-ledger SHA-256, recovery policy SHA-256, the exact supplemental task-ID set and a separate supplemental author-batch order. The v4 workflow revision is `phase4e-saracura-universal-synthetic.v3`.
1. Keep all 1,600 base task IDs, rows, statuses and lineage byte-identical. After the raw copy and capsule validation complete, base accepted/rejected data is reconstructed only from the compact external capsule's `resolved/call-*.json` files after `_load_resolved`, `_require_settled_task_resolution`, `_load_post_pilot_diagnostics` and `_require_post_pilot_uncertain_resolutions` pass against the exact v3 plan and final ledger. The ignored worktree artifact is no longer an execution dependency. No base provider request is reachable from the recovery command.
1. Generate exactly 100 new singleton PT-BR slots with unique families and no `pair_id`: 70 train, 15 dev and 15 holdout. Train allocates exactly ten slots to each option count 2 through 8. Dev and holdout distribute 15 slots as evenly as possible across those seven cardinalities. Every split covers all 12 domains, the domain-compatible scenario map is preserved and task/family IDs are disjoint from every v3 identity. Supplemental batch order round-robins split, cardinality and domain through the final task.
1. Reuse the exact corpus-v3 author, reviewer, provider, privacy, quality, duplicate, answer-blindness and lineage contracts. At the start of every invocation, including a circuit-breaker resume, seed the supplemental dedup index with all 1,457 accepted base rows across train, dev and holdout plus every previously accepted supplemental row. Each new accepted candidate is then added in deterministic batch order. A supplemental row equal or near-equal to any base or earlier supplemental row is rejected through the existing `semantic_duplicate` or `near_duplicate` path before packet sealing. The supplement changes no model, prompt, schema, timeout, retry count, price, provider routing, acceptance threshold or minimum.
1. Add a per-invocation transport circuit breaker. After an uncertain author or reviewer call is durably closed against its exact task set, increment a consecutive-uncertainty counter; any settled or overspent provider response resets it. Three consecutive uncertain calls stop the invocation before the next batch, append a ledger snapshot with `stop_reason=transport_circuit_open`, and leave every later identity unresolved. A later explicit invocation validates that stop snapshot and the entire ledger/resolution chain, skips every resolved task and resumes at the first unresolved supplemental batch with an in-process consecutive counter of zero. No uncertain request is replayed.
1. Separate execution result from packet sealing. Every terminal invocation writes an immutable numbered `report-*.json` before returning or raising after network work. Closed outcomes are `incomplete_transport_circuit`, `incomplete_operational`, `minimum_failed`, `seal_validation_failed` and `sealed`. The report includes accepted, rejected and unresolved counts; minimum errors or a closed seal-validation error code; split/locale/domain/cardinality/reason counts; uncertainty/consecutive-stop diagnostics; provider and conservative cost; base/recovery plan and ledger hashes; and an optional packet hash. It contains no task text, response body, reasoning, secret or absolute local path. A missing credential before any call continues to create no work or report. Re-entry with unchanged ledger and resolutions is idempotent by report payload hash and reuses the existing numbered report; a changed state appends the next number. Only the latest `sealed` report whose ledger, resolution and packet hashes still match may authorize Phase 4E.3B.
1. `minimum_failed` is a normal fail-closed research outcome: write the final report and durable evidence, keep the packet absent, print final actual spend and return a non-zero command result. `seal_packet` is called only when `_minimums` returns no error. The report path must not depend on a packet manifest existing.
1. Compose the packet from the verified v3 accepted/rejected rows plus verified supplemental rows. Build a combined ledger only after both input ledgers validate, have the same post-pilot policy, use disjoint reservation IDs and provider journals, and cover disjoint task identities. Concatenation order is base then supplement. The combined ledger exists only inside the packet staging/destination under `model-artifacts`; it never resides below `research-ledgers`, cannot be discovered by `_ledger_directories` and is never used as a resume ledger.
1. Add explicit packet/plan v4 validation rather than weakening v3. The v4 validator requires exactly the immutable 1,600-slot base prefix and exact 100-slot supplement, verifies 1,700 unique task IDs, preserves the 300 original complete pairs and accepts only null supplemental pair IDs. Packet v4 keeps the same physical train/dev versus holdout separation and runs the existing minimums over all accepted identities. Historical v2/v3 plans and packets remain byte- and behavior-compatible.
1. Keep live holdout content closed to human inspection. During private corpus curation, the recovery process may load all accepted base rows, including holdout rows, into an in-memory dedup index and may compare new rows against it; this is the same global duplicate gate used before packet sealing and does not expose text in reports or logs. After the combined packet seals, the training boundary resumes: consumers receive train/dev content plus holdout identities until the existing one-time claim. Tests prove a supplemental duplicate of a base holdout row becomes `semantic_duplicate` without placing holdout text in any report.
1. Add a configurable private state root through `SARACURA_PRIVATE_STATE_ROOT`. When set, it must be an absolute, non-symlink directory outside the repository, Git common directory and every path returned by `git worktree list`, owned by the current user and mode `0700`; the phase root is `<configured>/phase4e`. When absent, the existing `platformdirs` research root remains unchanged. The setting moves only the Phase 4E `research-ledgers`, `raw-evidence`, supplemental work, numbered reports and packet-v4 `model-artifacts` roots. `canonical_research_ledger_root` therefore changes for research/catalog commands, and a new v4-specific packet guard accepts only `<configured>/phase4e/model-artifacts/<reviewed-name>`; the historical v3 fixed packet guard remains unchanged. The one-time `holdout-releases` registry in `saracura_universal_training.py` remains permanently bound to its existing `platformdirs` path, has no environment/configuration surface and is never derived from `SARACURA_PRIVATE_STATE_ROOT`. Tests prove toggling the variable changes the listed private research/v4 paths but never the holdout-claim registry path. The value is runtime-only and never written to Git or report payloads.
1. Initialize the external catalog by copying the complete existing internal `research-ledgers` root create-only, including every historical directory, inventory and spend-index revision. Verify the copied inventory, spend index and fixed baseline before adding new evidence; never delete or rewrite the internal source. This is a storage migration, not a change in accounting scope. Every cumulative field remains global and must equal the historical baseline plus the corpus-v3 capsule plus supplemental work.
1. Split new durable evidence into `raw-evidence` and `research-ledgers` below the private phase root. Copy the complete 8.9 GB corpus-v3 work tree create-only into `raw-evidence/corpus-work-v47` on the configured external volume. Before capsule creation, validate the copied 8,455-snapshot ledger chain, every resolved/uncertain binding and the full raw inventory. Never delete or rewrite the source or destination. Catalog a compact `research-capsule.v1` containing all resolved and diagnostic files, the final ledger as `ledger/ledger-0000.json`, and a manifest binding the raw source inventory SHA-256, file count, byte count, first/final-ledger SHA-256, ledger snapshot count, successful chain-validation revision and every capsule file digest. The compact capsule is sufficient for spend scan, row/lineage reconstruction and tamper detection; the raw tier preserves transition-by-transition evidence.
1. Extend `_is_research_artifact`, inventory validation and mixed-root tests for the closed capsule manifest without treating the raw tier as a spend ledger a second time. `record_research_catalog` scans the migrated historical directories plus compact capsules. A closed capsule source key derived from its first/final ledger hashes may appear only once; the scanner also rejects any ordinary full ledger chain whose first/final hashes match a capsule source key, regardless of directory name. It must process corpus-v3 with memory independent of the number of ledger snapshots, produce exactly one accounting contribution for its 2,818 ledger entries and preserve totals equal to historical baseline plus capsule plus supplement.
1. Use create-only recovery paths below the configured private phase root for the supplement work, combined packet and numbered reports. The v4 command requires `SARACURA_PRIVATE_STATE_ROOT` and fails before work when it is absent; it never falls back to internal `platformdirs`. For the live run, preflight additionally fixes the configured root to a non-synchronized external path below `/Volumes/DevSSD` and rejects any path below `~/felhencloud`; the public environment contract documents that operators must not select a cloud-synchronized directory. Every terminal outcome after the first provider reservation, including `incomplete_transport_circuit`, `incomplete_operational`, `minimum_failed`, `seal_validation_failed` and `sealed`, creates or updates a compact supplemental capsule in `research-ledgers` and refreshes the catalog before the numbered report is finalized. No private task text, ledger, packet, model artifact or state-root path enters Git. The existing ignored `.artifacts/phase4e/corpus-work-v47` remains untouched until raw copy, capsule validation and catalog rescan all pass.
1. After the historical catalog is copied and verified, create a small immutable migration marker in the internal root that binds the external catalog root by non-sensitive identifier and inventory hash, never by absolute path. Subsequent Phase 4E research commands that resolve to the internal root while this marker exists fail closed with an instruction to configure the private root. This prevents the two global catalogs from silently diverging; reading and training claim registries remain unaffected.
1. Audit every schema dispatch found in `benchmarks/` for `phase4e-universal-plan.v3`, including `_author_batches`, `_post_pilot_config`, `_aggregate_preflight`, `_call_task_sets`, path selection, packet schema selection and training binding. Each v4 branch must select the reviewed post-pilot policy/model pair and supplemental-only transport order; falling through to legacy v2 behavior fails a direct test.
1. Define `resolution_sha256` as SHA-256 over the RFC 8785 canonical list of `{path, sha256}` entries for every `resolved/call-*.json` and `diagnostics/call-*.json` file in the relevant base capsule or supplemental work directory, sorted lexicographically by POSIX path relative to that capsule/work root. Every numbered report binds base and supplemental resolution hashes separately. If report writing itself fails, the original operational exception remains primary and the report-write failure is chained as secondary context.
1. Keep actual-cost reporting at each run-local USD 10 crossing and at every terminal outcome. There is no financial ceiling. The recovery report separates base historical cost from new supplemental cost so the new run cannot report corpus-v3 spend as a fresh crossing.

## Non-scope

- No new model/provider evaluation, prompt tuning, acceptance relaxation, English generation or replay of corpus-v3 identities.
- No embedding extraction, ranker training, checkpoint, calibration, threshold, holdout claim, runtime backend, service, telemetry or publication.
- No deletion or lossy compaction of raw corpus-v3 evidence. The compact capsule is an indexed derivative, not a replacement for the raw tier.
- No deletion, move or mutation of historical pilot artifacts in the existing internal root. A create-only verified copy into the external global catalog is required so accounting remains complete.

## Implementation phases

### A. Storage and terminal-outcome safety

Implement the explicitly scoped private root, immutable holdout-claim boundary, raw/capsule layout, capsule validation, streaming catalog compatibility and report-before-seal control flow. Create fixture-sized tests that prove raw evidence is counted once, a tampered capsule fails closed, the claim registry path never changes and minimum failure produces a report with no packet or post-seal holdout read.

### B. Deterministic recovery plan and composition

Add the v4 planner/validator, 100 supplemental slots, capsule-only base evidence loader and combined ledger/packet composition. Audit every v3 schema dispatch and prove v4 never falls through to legacy models/policy or iterates the base for transport. Prove exact base preservation, task/reservation disjointness, split/domain/cardinality allocation and v2/v3 compatibility.

### C. Circuit breaker and resume

Add the three-consecutive-uncertainty breaker and numbered outcome reports. Fake transport tests must produce uncertain, uncertain, settled, uncertain without opening the breaker; three consecutive author/reviewer uncertain calls must stop before a fourth call; settled and overspent results reset the streak. Resumption from `stop_reason=transport_circuit_open` must skip every settled/uncertain identity and continue from the first unresolved supplemental batch. The v4 `_author_batches` path returns the immutable supplemental order only; the base v3 order is never iterated for transport.

### D. QA, independent review and live recovery

Run all validation commands and a read-only reviewer before external calls. Configure the private state root on the external development SSD, copy and validate raw corpus-v3 evidence, create/catalog its compact capsule, then execute the supplemental run with the 1Password-injected OpenRouter key. If the combined packet seals, stop before training and record the packet/report/catalog hashes. If it does not seal, record the exact terminal outcome and write a new spec before changing protocol.

## Acceptance criteria

1. The v4 plan is deterministic, contains the exact v3 base plus exactly 100 disjoint PT-BR singleton slots, and has the specified 70/15/15 split and balanced cardinality/domain coverage.
1. Base corpus-v3 evidence validates byte-for-byte and no base task reaches transport. Any missing/tampered base resolution, uncertain binding, diagnostic or ledger fails before a new provider call.
1. Supplemental uncertain requests are never retried. Three consecutive author/reviewer uncertain calls stop before the next batch; settled or overspent responses reset the streak; resume from the circuit-open snapshot never repeats a settled or uncertain task and rebuilds dedup from the base capsule plus prior supplemental accepts.
1. Every post-network terminal path writes a closed numbered report. `minimum_failed` leaves the packet absent and records `locale_minimum` or other exact errors; packet/pre-holdout validation failure writes `seal_validation_failed` with a closed error code. Unchanged reruns are report-idempotent, and missing credentials before work still create nothing.
1. Packet v4 validates 1,700 planned identities, combined lineage, physical holdout isolation and all unchanged minimums. V2/v3 fixture bytes and validation semantics remain unchanged.
1. A combined result with 66 accepted supplement rows reaches PT-BR 60%; 65 accepted rows fails. This boundary is covered directly.
1. The configured state root rejects relative paths, symlinks, repository/Git-common/worktree descendants, foreign ownership and permissive mode; live preflight rejects the synchronized Felhen tree. The v4 command fails closed when configuration is absent. Other pre-migration reads preserve the platformdirs research path, while post-migration internal research writes fail on the immutable marker. Configured and unset modes resolve the exact same immutable holdout-claim registry path.
1. The complete historical internal catalog is copied create-only and verified at the external private root before new evidence. Raw corpus-v3 evidence is then copied create-only, chain-validated and inventoried. The compact capsule binds the complete raw inventory, contains only the final full ledger plus resolutions/diagnostics/manifest, is at least an order of magnitude smaller than raw evidence and contributes 2,818 ledger entries exactly once. The indexed total equals historical baseline plus capsule plus supplemental ledger.
1. Ledger-chain validation memory is independent of the number of snapshots, and catalog inventory memory remains proportional to the number of inventory entries. Fixture tests preserve exact transition-error behavior.
1. No report, tracked file, wheel or sdist contains secrets, private task content, absolute local paths, ledgers, packets, model bytes or the configured state-root value.
1. Full lint, formatting, isolated type check, isolated tests, manifest validation, default-environment validation, package build, artifact inspection and `git diff --check` pass. Independent read-only review returns `PASS` before live transport.
1. A sealed combined packet authorizes only the next separately reviewed Phase 4E.3B training increment. This recovery command never opens the live holdout payload through the training/full-packet validator.

## Validation commands

```bash
uv run --isolated --locked ruff check .
uv run --isolated --locked ruff format --check .
uv run --isolated --locked mypy src tests benchmarks
uv run --isolated --locked pytest -q
uv run --isolated --locked python -m benchmarks.validate_manifests
uv run --isolated --locked python benchmarks/validate_default_environment.py
uv build
uv run --isolated --locked python benchmarks/inspect_wheel.py <wheel> <sdist>
git diff --check
```

## Rollout and rollback

All repo contracts are additive and historical schemas remain unchanged. Before a provider call, rollback removes only the new uncommitted/feature-branch code and empty create-only recovery paths. After a provider call, supplemental work, ledger, report and raw/capsule evidence are immutable and are never deleted or replayed. The circuit breaker makes an outage resumable without inventing results. A failed minimum requires a new reviewed plan; it never permits lowering a threshold. Runtime behavior remains unsupported, so rollback has no user-facing inference effect.

## Risks

- A 100-task supplement could still accept fewer than 66 rows. The fixed boundary remains fail-closed, and the report preserves enough evidence for a new disjoint plan without replay.
- Combining two ledgers can accidentally double-count spend or bind the wrong lineage. The implementation requires disjoint reservations/task identities, exact source hashes and separate base versus supplemental spend fields before composition.
- A configurable root can become an arbitrary-output escape hatch. Closed ownership, mode, symlink and repository-boundary checks keep it a private state selector rather than a general path override.
- Compact evidence can be mistaken for deletion-safe replacement. The raw tier is mandatory and create-only; the capsule manifest explicitly binds it, and no cleanup is authorized in this increment.
- Numbered reports can disagree across resumes. Each report binds the exact latest ledger/resolution hashes and outcome; only a final `sealed` report may authorize the next phase.

## Critique resolution

Round 1 returned `NO-GO` on two concrete gaps. First, a supplemental row could duplicate a base accepted row because the new work directory would otherwise seed dedup only with supplemental rows; the revision now loads all 1,457 base accepted rows, including private holdout rows, into the in-memory corpus-curation dedup index and directly tests base-holdout duplicate rejection. Second, selecting an external canonical root would have omitted the existing internal history from cumulative totals; the revision now requires a create-only, fully verified copy of the historical catalog before adding the corpus-v3 capsule or supplemental ledger. The revision also adds an explicit seal-failure outcome, idempotent report rules, author/reviewer circuit semantics, overspent reset, Git-common/worktree path exclusion, raw-chain validation and a v4-only supplemental batch iterator.

Round 2 confirmed both prior blockers resolved and returned one new `NO-GO`: an imprecise statement that the configurable root applied to every Phase 4E command could be implemented to move the global holdout-claim registry, permitting two claims across configured and unset runs. This revision enumerates the movable roots, adds a v4-only packet guard and explicitly excludes/tests the immutable `platformdirs` holdout registry. It also rebuilds dedup on every resume, prevents capsule/full-chain and combined-ledger double counting, audits all v3 schema dispatches, defines resolution hashes, preserves primary operational errors and fixes the live root outside synchronized storage.

Round 3 returned `GO` with no blocker. Its non-blocking hardening points are now explicit requirements: content-based capsule/full-chain duplicate rejection, an internal migration marker that prevents catalog divergence, supplemental capsule/catalog publication on every post-reservation terminal outcome, mandatory v4 root configuration, relative resolution-hash paths and pre-copy Python validation of the internal inventory and spend history.
