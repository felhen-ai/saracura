---
title: Phase 3B-S synthetic research training
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase3b-synthetic-research-training.md
globalRef: qmd://saracura/docs/action/specs/phase3b-synthetic-research-training.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-21
sourceRefs:
  - https://openrouter.ai/terms
  - https://openrouter.ai/qwen/qwen3.5-9b/pricing
  - https://openrouter.ai/mistralai/ministral-8b-2512
  - https://openrouter.ai/typesafe/jev-1.13/api
related:
  - docs/action/specs/phase2b-encoder-candidate-gate.md
  - docs/action/specs/phase2c-data-governance-gate.md
  - docs/action/specs/phase3a-first-party-packet.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 3B-S synthetic research training

## Context

Phase 3A froze the human-original protocol and deterministic split gate but did
not create records or authorize training. The operator has now approved a
separate synthetic research lane, including up to USD 0.25 of existing
OpenRouter credit, offline tests, an actual local training run, and a Jev
comparison. The lane accelerates architecture learning without weakening the
human-original gate.

The MacBook is an Apple M4 Pro with 24 GiB unified memory. The existing encoder
registry already permits the immutable `multilingual-minilm-l12` safetensors
snapshot. The default install and public runtime remain lightweight and offline.

Current policy quarantines model-generated text from canonical `train`, `dev`,
`calibration`, and `blind_test`. This phase preserves that decision. It creates
different split names and artifacts whose schema and metadata make synthetic
origin impossible to confuse with first-party human evidence.

## Goal

Deliver and exercise a fail-closed, checkout-only research pipeline that:

1. plans 250 fictional PT-BR support-routing scenario families before generation;
2. uses Qwen3.5 9B as synthetic author and Ministral 3 8B as independent model
   reviewer through OpenRouter;
3. mechanically rejects unsafe, malformed, duplicate, ambiguous, or mislabeled
   cases;
4. trains one frozen-MiniLM five-class linear head locally on Apple MPS;
5. evaluates the frozen head and pinned Jev 1.13 on the same synthetic holdout;
6. persists immutable manifests, metrics, lineage, token usage, and cost without
   committing generated text, credentials, weights, or result artifacts.

The output is architecture evidence only. It is not human validation, a
calibration map, a blind test, a publishable dataset, a production checkpoint,
or evidence that automation is safe.

## Non-goals

- Reclassify model output as `human_original` or as a deterministic derivative.
- Modify or consume the Phase 3A first-party base-state schema or split plan.
- Generate from customer, employee, partner, AIOS, production, or private data.
- Fit temperature calibration, confidence gates, abstention thresholds, or risk
  policy.
- Publish dataset rows, prompts, weights, benchmark results, or quality claims.
- Fine-tune the encoder, tokenizer, Qwen, Ministral, or Jev.
- Use Jev output as a training label, reviewer decision, adjudication, or prompt
  revision signal.
- Add remote inference to the Saracura runtime or package.
- Select a permanent production architecture from this run.

## Provenance and policy boundary

Add `benchmarks/manifests/synthetic-research-policy.v1.json`. Its closed schema
binds exact model IDs, workflow revision, labels, prices used for the local
budget ceiling, OpenRouter/API/terms URLs, accessed date, privacy controls, and
the following literal authorizations:

- `synthetic_generation_authorized=true`;
- `synthetic_research_training_authorized=true`;
- `human_original=false`;
- `canonical_training_authorized=false`;
- `calibration_authorized=false`;
- `blind_test_authorized=false`;
- `publication_authorized=false`;
- `quality_claims_allowed=false`;
- `automation_authorized=false`.

This phase also amends `training-data-source-policies.v1.json` and its closed
validator with one explicit `synthetic_experiment` exception bound to this
workflow revision. Provider-model output remains globally `quarantined`; the
exception permits only local `synthetic_only` research training and evaluation,
and explicitly denies canonical training, calibration, blind-test use,
publication, quality claims, and automation. No other source or workflow may
inherit the exception. Synthetic artifacts use splits
`synthetic_train`, `synthetic_dev`, and `synthetic_holdout`; these values must be
rejected by every human-original parser. Human split names must be rejected by
the synthetic parser.

OpenRouter routes to model providers whose terms may change. Every live run
requires the exact manifest revision, `provider.zdr=true`,
`provider.data_collection=deny`, and `provider.enforce_distillable_text=true`
for author and reviewer requests. A request is refused if OpenRouter cannot
route under all three constraints. The response's resolved model and provider
metadata are recorded when available. This is an engineering provenance gate,
not legal advice or a permanent conclusion about third-party terms.

## Credential and spending boundary

The runner accepts no API key flag, file path, secret reference, or arbitrary
base URL. It reads only `OPENROUTER_API_KEY` from the environment and sends it
only in the HTTPS Authorization header to the literal host
`openrouter.ai`. It never prints, hashes, stores, or returns the key.

Live generation requires all of:

- `--allow-network`;
- `--budget-usd 0.25` exactly for the approved pilot;
- either a new output directory or a resumable directory whose immutable plan,
  append-only ledger, and completed response files all validate byte-for-byte;
- a successful read-only `/api/v1/key` preflight; `/api/v1/credits` is
  best-effort because OpenRouter documents it as management-key-only, and a
  successful response showing less than USD 0.25 available stops the run;
- no automatic credit purchase or fallback to another credential.

Before each request, a conservative upper bound uses UTF-8 prompt bytes plus a
fixed protocol allowance as maximum input tokens, fixed `max_tokens`, and the
manifest price ceilings. The USD 0.25 authorization is partitioned into hard
stage ceilings: USD 0.15 authoring, USD 0.06 review, and USD 0.04 Jev. Author
and review repair calls consume their own stage allocation; unused allocation
is not transferred. A request is refused if its worst-case bound could cross
either its stage ceiling or the cumulative ceiling. After each response,
server-reported usage/cost is recorded and becomes authoritative. Chat calls
record `usage.cost` and audit it against the eventually consistent
`/api/v1/key` usage delta with three bounded polls; Jev calls must provide
`usage.cost`. Missing or invalid response-level usage fails closed. A delayed
key-level delta is recorded as reconciliation evidence but does not override the
response-level cost or trigger an unbounded retry.
Timeouts, malformed responses, and billed failures count toward the budget if
the service reports cost. Retries are limited to one repair attempt per batch
and are separately budgeted. The runner stops at the first uncertain-cost
response and does not issue another request. Resume validates the immutable
plan and ledger, skips completed batches, and never retries a completed or
uncertain-cost request.

The initial credential is an existing unbounded OpenRouter key injected from
1Password. Therefore the local hard ceiling and the server usage reconciliation
are both required. No code may create credits, enable auto-reload, create API
keys, or change account settings.

## Frozen pilot plan

The pilot contains exactly 250 planned families: 50 for each label in protocol
order:

1. `billing`;
2. `technical_support`;
3. `account_access`;
4. `subscription_cancellation`;
5. `order_delivery`.

The public seed is `saracura-phase3bs-pilot-20260921-v2`. Family identity is a
SHA-256 derivation of workflow revision, label, and zero-based label index. The
planner assigns each label independently, before generation, to exactly 30
`synthetic_train`, 10 `synthetic_dev`, and 10 `synthetic_holdout` families by
SHA-256 rank. There are no variants in this phase.

Each family also receives deterministic generation axes selected from closed
lists: message style, length bucket, spelling-noise level, explicitness,
negation, urgency, and Brazilian regional-neutral wording. The plan contains no
generated text and is immutable create-if-absent canonical JSON.

Revision v2 uses a fresh predeclared seed and explicit label-boundary guidance
after the first local pilot stopped at its unchanged minimum-acceptance gate.
No row, split, decision, or artifact from that failed pilot is reused.

## Qwen author contract

The exact author model is `qwen/qwen3.5-9b`. A request contains ten planned
families of one intended label, the frozen annotation guide digest, the label
definition and priority rules, fictional-content requirements, and a strict
JSON Schema. Temperature is `0.8`, reasoning is disabled when supported, and
`max_tokens` is fixed at 1,800.

The response has exactly ten records and each record has exactly:

- the supplied `family_id`;
- one NFC PT-BR `text` from 12 through 320 Unicode code points.

The runner reattaches the immutable planned label and generation axes after the
response validates. They are deliberately omitted from model output so the
model cannot rewrite provenance and the JSON remains inside the fixed token cap.

The model may not choose IDs, labels, splits, reviewer outcomes, or provenance.
It may not mention real companies, people, domains, handles, account numbers,
URLs, secrets, or claim that a case is human-authored. Invalid batches receive
at most one schema-repair call containing validation codes but no accepted rows.

## Ministral reviewer contract

The exact reviewer is `mistralai/ministral-8b-2512`. It receives ten Qwen rows
containing only opaque family IDs and text, plus the same frozen guide digest
and priority rules. It receives neither planned labels, splits, generation axes,
nor Qwen reasoning. Temperature is `0`, `max_tokens` is fixed at 1,600, and its
strict JSON response has one decision per supplied family:

- `status`: `accepted`, `rejected`, `ambiguous`, or `out_of_scope`;
- `review_label`: one of the five labels or null;
- unique closed `reason_codes`;
- booleans for `natural_ptbr`, `fictional`, `single_owner`, and
  `contains_sensitive_pattern`.

The runner compares `review_label` with the hidden planned label only after the
response is sealed. Acceptance requires `status=accepted`, label equality, all three positive
quality booleans, no sensitive flag, no mechanical privacy hit, NFC, and no
exact/normalized/near duplicate across all 250 rows. Reviewer prose is neither
requested nor stored. A malformed review batch gets at most one separately
budgeted repair attempt.

Training is permitted only if accepted rows include at least 20
`synthetic_train`, 5 `synthetic_dev`, and 5 `synthetic_holdout` families per
label. Rejection never causes rerandomization, split movement, text rewriting,
or label correction. A failed minimum ends the run before encoder acquisition
or training.

## Artifact contract

Live output is rooted below a newly created operator-selected directory under
`.artifacts/`. Files are mode 0600, directories 0700, canonical UTF-8 JSON, and
atomic create-if-absent. The packet contains:

- text-free plan and prompt/template digests;
- raw author and reviewer response envelopes with provider request IDs, model,
  usage, cost, and content digests;
- accepted and excluded synthetic rows with full model provenance;
- exclusion ledger with mechanical and reviewer reason codes;
- exact/normalized/near-duplicate report;
- packet manifest binding every permitted packet file other than itself by
  relative filename, size, and SHA-256. The validator uses an exact filename
  allowlist, rejects links/non-regular files/extras, and treats the canonical
  manifest bytes as the packet root digest;
- cumulative cost ledger and pre/post account usage snapshots.

No API key, Authorization header, 1Password reference, username, hostname, home
path, absolute repo path, IP address, or environment value may appear. Generated
artifacts and weights remain gitignored and are never included in wheel/sdist.

## Training contract

Training is a separate explicit command that consumes only a validated packet
manifest from the same run. It refuses human packet schemas, missing provenance,
modified bytes, insufficient per-label counts, a non-final cost ledger, or any
authorization/claim literal inconsistent with this spec.

The frozen experiment is:

- encoder: registry candidate `multilingual-minilm-l12` at its exact immutable
  revision and verified safetensors bytes;
- embedding device: Apple MPS, with no silent CPU fallback;
- encoder mode: frozen `eval()` under `no_grad`;
- input: accepted text only, tokenized at maximum 128 tokens;
- pooling: attention-mask-weighted mean float32 without normalization;
- head: `torch.nn.Linear(384, 5)`, trained on CPU only from sealed embeddings;
- label order: protocol order above;
- seed: `20260921` for Python and Torch;
- optimizer: AdamW, learning rate `0.01`, weight decay `0.01`;
- batch size: 32;
- maximum epochs: 80;
- early stopping: synthetic-dev macro-F1, minimum delta `0.001`, patience 10;
- loss: unweighted cross entropy;
- selection: best synthetic-dev macro-F1, earliest epoch on ties.

The encoder is computed once for each row on MPS and cached only inside the
run's local artifact directory as canonical float32 embeddings. The embedding
artifact is sealed by digest before any head training. Two independent CPU head
runs then consume the identical sealed embedding bytes with deterministic
algorithms enabled; both must produce the same safetensors checkpoint digest.
The immutable
training manifest binds code revision, packet bytes, encoder registry/snapshot,
protocol, seed, hyperparameters, environment versions, epoch ledger, selected
epoch, checkpoint bytes, and train/dev metrics. It marks every metric
`synthetic_only=true` and every claim/automation/calibration authorization false.

Training success means embedding extraction completed on MPS, both CPU head
runs completed deterministically, loss and metrics are finite, the checkpoint
reloads to identical logits within absolute and relative tolerance `1e-6`, and
both runs produce the same checkpoint digest. A failure is reported as a
fail-closed capability result. There is no minimum accuracy acceptance threshold.

## Synthetic holdout and Jev comparison

After the checkpoint and all code are frozen, evaluate the local head exactly
once on `synthetic_holdout`. Record confusion matrix, per-label precision,
recall/F1, macro-F1, accuracy, and latency distributions. These are diagnostic
synthetic metrics only.

Then call pinned `typesafe/jev-1.13` through
`POST https://openrouter.ai/api/alpha/decisions` once per holdout record,
strictly sequentially. Each request supplies that record as the sole `state`
and one choice question with the frozen five criteria descriptions. No fanout or
multi-record envelope is allowed because the endpoint contract is one state plus
a question map. Record the returned resolved revision, choice,
probabilities, confidence when present, latency, usage, and cost. Jev responses
may be compared to the planned synthetic labels but must never alter dataset
rows, exclusions, training, prompts, or model selection.

If TypeSafe/OpenRouter model terms do not clearly authorize reuse of Jev output
as a published artifact, store only aggregate local comparison metrics and
response digests. Jev failure or terms ambiguity does not invalidate the local
training proof; it produces a separately reported `external_control_unavailable`
state. Do not fall back to `jev-latest` or another model.

## Implementation scope

Allowed:

- `benchmarks/**` checkout-only generation, review, packet, training, and
  comparison tools;
- `tests/**` offline tests with fake HTTP and fake encoder/model objects;
- `benchmarks/manifests/synthetic-research-policy.v1.json`;
- this spec, README, PT-BR README, CONTRIBUTING, SECURITY, `.gitignore`, and CI;
- optional dependency metadata only when needed for the existing encoder lane;
- minimal reusable helpers under `src/saracura/**` only if required by a future
  backend contract, not for this research run.

Prohibited:

- generated text, raw responses, weights, caches, result reports, credentials,
  or account metadata committed to Git;
- default-install remote clients or new runtime dependencies;
- modification of Phase 3A human packet semantics;
- remote runtime fallback, arbitrary models/base URLs, or request-triggered
  generation/training;
- calibration, blind-test, public claims, release, package publication, server,
  VM, AIOS, or production integration.

## Commands

Offline/default validation:

```bash
uv sync --locked --dev
uv run python benchmarks/validate_default_environment.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
git diff --check
```

Live pilot, with the secret injected only by the operator environment:

```bash
uv run python -m benchmarks.synthetic_research generate \
  --allow-network --budget-usd 0.25 \
  --output-dir .artifacts/phase3bs-pilot

UV_PROJECT_ENVIRONMENT=.venv-training uv sync --locked --dev --extra encoder-eval
UV_PROJECT_ENVIRONMENT=.venv-training uv run python -m benchmarks.encoder_gate acquire \
  --candidate multilingual-minilm-l12 --allow-network
UV_PROJECT_ENVIRONMENT=.venv-training uv run python -m benchmarks.synthetic_research train \
  --packet .artifacts/phase3bs-pilot/packet-manifest.json \
  --output-dir .artifacts/phase3bs-training
UV_PROJECT_ENVIRONMENT=.venv-training uv run python -m benchmarks.synthetic_research evaluate \
  --training .artifacts/phase3bs-training/training-manifest.json \
  --packet .artifacts/phase3bs-pilot/packet-manifest.json \
  --allow-network --budget-usd 0.25 \
  --output-dir .artifacts/phase3bs-evaluation
```

The evaluation command uses only the generation run's unspent portion of the
same USD 0.25 authorization. The cumulative ledger, not a fresh CLI flag, is
authoritative.

## Acceptance criteria

1. The synthetic policy validates offline and cannot be confused with the Phase
   2C or 3A human-source schemas.
2. Tests prove credential redaction, literal HTTPS host, privacy controls,
   price-bound preflight, cumulative USD 0.25 stop, uncertain-cost stop,
   timeouts, retry cap, and immutable output behavior.
3. Tests prove exactly 250 preplanned families, 50 per label, fixed 30/10/10
   splits, stable IDs, and no post-review movement or rewriting.
4. Strict fake Qwen and Ministral flows cover valid output, schema failure,
   duplicate IDs, missing rows, wrong labels, PII, duplicates, ambiguity,
   rejection, and minimum-count failure without network in CI.
5. The real pilot completes within USD 0.25 or stops without issuing a request
   that can exceed the remaining conservative ceiling.
6. Only reviewed, mechanically clean synthetic rows enter training.
7. MiniLM acquisition remains immutable, pinned, anonymous, verified, and
   safetensors-only through the existing gate.
8. One local MPS embedding extraction is sealed, and two deterministic CPU head
   training runs produce identical checkpoint digests and reload-equivalent
   logits; all artifacts remain outside Git.
9. Synthetic holdout is accessed only after checkpoint freeze and cannot affect
   selection. Jev is pinned to 1.13 and cannot feed back into the packet or head.
10. Reports and manifests state `synthetic_only`, prohibit quality/calibration/
    automation claims, and contain no credentials or machine identity.
11. Default environment remains offline/lightweight; CI makes zero remote calls
    and downloads no weights or datasets.
12. Ruff, format, strict MyPy, Pytest, manifests, build, wheel/sdist inspection,
    secret scan, and `git diff --check` pass.

## Rollout and rollback

Repository changes ship through a normal PR. There is no service deploy or
package release. Live artifacts remain local and untracked. Rollback is a Git
revert plus deletion of the explicitly named `.artifacts/phase3bs-*`
directories and, if desired, the already verified shared MiniLM cache. Spent
OpenRouter credit is not recoverable. No account setting or key is changed.

## Open risks

- Synthetic author and reviewer models may share training data and stylistic
  priors; cross-family model choice reduces but does not remove correlation.
- A synthetic holdout can substantially overestimate real PT-BR performance.
- Reviewer acceptance can create lexical shortcuts and overly clean classes.
- Provider/model terms and routing behavior can change after this dated review.
- An unbounded account key means the local budget gate is the active protection;
  a later operator should replace it with a dedicated server-capped key.
- Frozen embeddings plus a linear head may prove plumbing but underfit nuanced
  ownership decisions; this phase records the result instead of silently
  expanding architecture.
- Jev's Decisions endpoint is alpha and may change independently of chat APIs.

## Next gate

The result may justify a later architecture iteration or a genuinely human
evaluation packet. It cannot authorize calibration, automation, publication, or
claims about PT-BR users. Any public benchmark requires a new reviewed spec with
independent data and terms evidence.
