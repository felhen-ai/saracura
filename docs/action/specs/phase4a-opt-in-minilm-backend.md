---
title: Phase 4A opt-in MiniLM routing backend
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4a-opt-in-minilm-backend.md
globalRef: qmd://saracura/docs/action/specs/phase4a-opt-in-minilm-backend.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs:
  - docs/action/specs/phase3b-synthetic-research-training.md
  - docs/action/specs/phase3c-end-to-end-throughput-benchmark.md
  - docs/action/specs/phase3d-typesafe-native-control.md
related:
  - docs/action/specs/phase2b-encoder-candidate-gate.md
  - SECURITY.md
  - README.md
supersedes: []
supersededBy: []
sensitivity: public
---
# Phase 4A opt-in MiniLM routing backend

## Context and current truth

Saracura currently exposes a closed `v1alpha1` Choice API and a deterministic
fixture backend. The fixture proves serialization, workflow, calibration,
single-state-encoding, and failure contracts, but it is not a learned model.
Phase 3B-S produced a local, synthetic-only five-class PT-BR routing head over
the frozen `paraphrase-multilingual-MiniLM-L12-v2` encoder. Phase 3C measured
that exact stack on Apple MPS: tokenization on CPU, encoder and mean pooling on
MPS, an explicit MPS-to-CPU transfer, and the linear head on CPU. Neither the
encoder weights nor the synthetic head are distributed in the repository or
package.

The Phase 3B-S checkpoint is explicitly not a production model. Its source
packet is synthetic, its training manifest forbids calibration, automation,
and quality claims, and it has no independent human PT-BR test. Phase 4A makes
that stack executable through the existing research API without weakening
those boundaries. Human calibration and blind evaluation remain Phase 4B
gates; high-volume batching and schema reuse remain Phase 4C.

The existing engine derives calibration compatibility from fixture constants
and checks calibration only after state encoding. That is safe enough for the
zero-cost fixture but wrong for a real backend. This phase moves model-side
compatibility behind the backend protocol, keeps dataset-profile identity
separate, and resolves all calibrations before tokenization or device work.

## Goal

Add an explicit, local-only `MiniLMRoutingBackend` that can execute the frozen
five-label synthetic support-routing checkpoint through both the Python API and
CLI on the MacBook. It must:

1. load only operator-supplied, descriptor-verified encoder, training-manifest,
   and safetensors bytes;
2. perform no network request, implicit download, telemetry, remote fallback,
   or dynamic code loading;
3. reproduce the Phase 3C numerical path and bind its runtime ABI into
   calibration compatibility;
4. support one fully specified immutable PT-BR workflow and label mapping;
5. return `fixture_only` answers behind a no-fit identity transform, never a
   calibrated or automation-ready claim;
6. leave the default install, fixture flow, wheel, and source distribution free
   of weights and research data.

## Non-goals

This phase does not train, fine-tune, calibrate, quantize, publish, download, or
deploy a model. It does not add a server or remote backend. It does not support
arbitrary workflows, labels, questions, Boolean/Score primitives, multiple
states, batching, caches, multi-head execution, or request-selected paths. It
does not claim general PT-BR quality, statistical significance, production
readiness, or safe automation.

## Frozen workflow and output identity

The new workflow is exactly:

- id: `support-routing`;
- revision: `phase4a-local-minilm-routing.v1`;
- locale: `pt-BR`;
- domain: `support`;
- one question, id `department`, type `choice`;
- instruction: `Classifique a mensagem de suporte em exatamente uma categoria. Aplique as definições e a prioridade descritas nos critérios.`;
- criteria, in this exact checkpoint-index order:
  1. `billing`: `cobrança, pagamento, estorno, fatura ou método de pagamento, salvo cancelamento puro`;
  2. `technical_support`: `falha, configuração, compatibilidade ou ajuda de uso não bloqueada por acesso`;
  3. `account_access`: `autenticação, identidade, credencial ou acesso à conta; tem prioridade quando bloqueia outra ação`;
  4. `subscription_cancellation`: `parar assinatura, renovação ou plano recorrente sem cobrança ou estorno separado`;
  5. `order_delivery`: `envio, rastreio, entrega, pacote ausente/danificado ou pedido físico`.

The order above is semantic for the trained head even though normal request
serialization sorts criterion IDs. Add `ordered_question_bytes`, defined as
RFC 8785 JSON of the closed question object with its criterion array left in
tuple order. `WorkflowSchema` gains `criteria_order_semantic=false`; it is true
only for this revision and makes registry validation compare
`ordered_question_bytes` rather than the legacy sorted serialization. The
backend independently rejects direct calls unless those ordered bytes match the
frozen object; it never maps logits from caller-supplied order. The architecture
descriptor contains both the ordered label tuple and
`sha256(ordered_question_bytes(frozen_question))`. Reordering otherwise
identical criteria must be rejected by both boundaries. This reconciles the
serving workflow with the training manifest's
`phase3b-synthetic-research-training.v2` provenance without pretending those
two revisions are interchangeable.

## Model-side compatibility metadata

`BackendCalibrationMetadata` is a frozen value with only model/runtime axes:

- `architecture_config_sha256`;
- `tokenizer_revision`;
- `truncation_policy_id`;
- `precision`;
- `quantization`;
- `output_transform`.

The backend protocol requires a `calibration_metadata` property. The fixture
returns the six current constants, preserving existing fixture contexts and
response bytes. A legacy structural backend without the property fails with
`BACKEND_UNAVAILABLE` before encoding; fixture metadata is never substituted
for an unknown backend.

The MiniLM architecture descriptor is canonical RFC 8785 JSON with exactly:

- schema `minilm-routing-backend.v1`;
- encoder candidate `multilingual-minilm-l12`, revision
  `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, reviewed registry SHA-256,
  `BertModel`, and hidden width 384;
- SHA-256 of the complete training-manifest file, its recorded self-digest,
  packet-manifest SHA-256, protocol SHA-256, and training workflow revision;
- checkpoint SHA-256, ordered labels, width 384, classes 5, exact head tensor
  schema, and checkpoint-to-index mapping;
- frozen workflow id/revision, serialized-question SHA-256, locale, and domain;
- state extraction `single-text-field-v1`, mean pooling
  `attention-mask-fp32-v1`, maximum length 128, truncation true, padding true,
  and token count `post-truncation-attention-mask-sum-v1`;
- execution path, one of
  `encoder-fp32-mps-pool-fp32-mps-transfer-head-fp32-cpu-v1` or
  `encoder-fp32-cpu-pool-fp32-cpu-head-fp32-cpu-v1`;
- runtime ABI object containing full Python version, exact installed versions
  of torch, transformers, tokenizers, and safetensors, and concrete class names;
- `runtime_implementation_sha256`, computed over a canonical name/byte stream
  of the installed `serialization.py`, `runtime/engine.py`,
  `runtime/workflows.py`, `backends/minilm.py`, package verified-byte loader,
  and `calibration/models.py` resources;
- `platform_backend_abi_sha256`, the canonical digest of sanitized
  Python-implementation, OS name/release, machine architecture, torch version
  and git version, SHA-256 of `torch.__config__.show()`, MPS-built/available
  booleans, and the active execution path;
- output transform `linear-logits-softmax-v1`.

`architecture_config_sha256` is the SHA-256 of those canonical bytes.
`model.id` is `saracura-minilm-routing`; `model.revision` is
`minilm-routing-v1.<architecture_config_sha256>`; `checkpoint_sha256` is the
head file digest. `precision` is the exact execution-path string,
`quantization` is `none`, `tokenizer_revision` is
`minilm-tokenizer-e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, and
`truncation_policy_id` is `padding-truncation-max128-v1`. A dependency, device,
class, implementation byte, OS/backend ABI, workflow, checkpoint, or numerical
path change produces a new revision and cannot reuse a Phase 4B calibration.

## Calibration dataset profiles

Dataset evidence is not backend metadata. Add frozen
`CalibrationDatasetProfile(dataset_id, dataset_revision,
split_manifest_sha256)`. `calibration_context` receives both the backend's
model metadata and an explicit profile.

`DecisionEngine` gains optional `calibration_profiles: Mapping[question_id,
CalibrationDatasetProfile]`. Omitting it is backward-compatible only for
`DeterministicFixtureBackend`, for which the current fixture profile is used.
Every other backend fails before encoding when its question profile is absent.
The MiniLM CLI supplies the Phase 4A identity profile explicitly. Phase 4B adds
a new immutable human-evidence profile; it never mutates the backend or Phase
4A profile.

The Phase 4A profile literals are:

- dataset id `no-fit-synthetic-identity`;
- dataset revision `phase4a.v1`;
- split manifest digest = SHA-256 of canonical JSON
  `{"dataset_id":"no-fit-synthetic-identity","dataset_revision":"phase4a.v1","fit":[],"evaluation":[],"purpose":"identity-transform-only"}`.

This digest describes the deliberate absence of fit/evaluation records. It is
not the synthetic training packet digest.

## Identity artifact schema

Preserve the existing `CalibrationArtifact` class, parser behavior, and
canonical output for `method="temperature-scaling"`. Add a separate frozen
`IdentityCalibrationArtifact` and define `AnyCalibrationArtifact` as the closed
union accepted by loaders and the engine. The identity variant has:

- `schema_version=1`, `status="fixture_only"`, `method="identity"`;
- all existing `CalibrationContext` fields;
- parameters exactly `{"temperature":1.0}`;
- fit/evaluation sample counts, independent-state counts, and minimum count all
  exactly zero;
- metrics before and after exactly empty objects;
- no threshold, abstention, risk policy, bounds, or authorization field;
- `created_at` in canonical UTC `YYYY-MM-DDTHH:MM:SSZ`, parsed as a real
  calendar timestamp and round-tripped exactly.

The ID is
`identity-v1-<first-32-hex(sha256(RFC8785({context,method,parameters})))>`.
Validation recomputes it. The create helper accepts the timestamp, resolved
context, and destination; it writes canonical JSON with one trailing newline
using the existing atomic create-if-absent path. Existing destinations are
never replaced. Phase 4B uses `temperature-scaling`, positive counts, its new
dataset profile, and a different ID; no identity artifact may be edited or
relabelled.

## Verified-bytes input boundary

Runtime package code owns the loader and does not import `benchmarks`. The
package resource `saracura/encoder-candidates.v1.json` is the only runtime
registry. Editable and wheel installs both use `importlib.resources`; a test
compares its bytes and digest with the canonical repository registry. There is
no fallback path lookup.

The constructor accepts explicit paths for snapshot directory,
`training-manifest.json`, `checkpoint.safetensors`, and device (`mps` or
`cpu`). Every external directory/file is opened with `openat` and
`O_NOFOLLOW` where supported and checked by file descriptor. Directories must
be owned by the current UID and mode 0700. Files must be regular, owned by the
current UID, mode 0600, and link count one. A changed `(device,inode,size,mode,
uid,nlink,mtime_ns,ctime_ns)` between opening and final read is rejected.

Each allowlisted file is read once from its verified descriptor into a bounded
immutable `bytes` value while hashing it. Snapshot files must exactly match the
registry names, sizes, and SHA-256 values plus the validated
`snapshot.complete.json`; total snapshot bytes may not exceed the registry sum
plus 1 MiB. The manifest is capped at 1 MiB and the head at 16 MiB. Directory
entries are checked again before descriptors close. Symlinks, hardlinks,
devices, FIFOs, extra/missing files, ownership/mode drift, or mutation at any
point fail before model construction.

No verified path is passed to Transformers. Construction from the exact bytes
is:

1. parse `config.json`, `tokenizer_config.json`,
   `special_tokens_map.json`, and the training manifest with duplicate-key
   rejection and closed shapes;
2. construct `BertConfig` from the validated config dictionary and
   `BertModel(config)` directly;
3. decode `model.safetensors` with `safetensors.torch.load(bytes)`, require its
   key set to equal the fresh model `state_dict`, and require every tensor's
   exact expected shape and dtype, finite floating values, and strict
   `load_state_dict` with no missing/unexpected keys;
4. construct `tokenizers.Tokenizer.from_str(tokenizer.json UTF-8 bytes)` and
   `PreTrainedTokenizerFast(tokenizer_object=...)` with exactly
   `unk=<unk>`, `sep=</s>`, `pad=<pad>`, `cls=<s>`, `bos=<s>`, `eos=</s>`,
   `mask=AddedToken("<mask>", single_word=false, lstrip=true, rstrip=false,
   normalized=true)`, model max length 512, right padding, and right
   truncation; require class `PreTrainedTokenizerFast`, vocabulary/length
   250002, the seven-value special-token map, and the frozen conformance vector
   below;
5. decode the head from the exact checkpoint bytes and require only
   `weight: float32[5,384]` and `bias: float32[5]`, both finite; load strictly
   into `torch.nn.Linear(384,5,device="cpu")`.

`auto_map`, custom modules, pickle, `.bin`, `AutoModel`, `AutoTokenizer`,
`from_pretrained`, hub clients, environment cache resolution, and remote code
are forbidden. Constructor failures map to sanitized Saracura errors without
paths, content, or raw third-party exception text.

The package includes immutable `minilm-conformance.v1.json`, bound into the
architecture descriptor. It contains this self-authored, non-private vector:

- text: `Preciso atualizar o endereço de entrega do meu pedido.`;
- input IDs: `[0,70464,31,46581,14821,36,136689,8,23809,54,4346,39568,5,2]`;
- attention mask: fourteen ones;
- token-type IDs: fourteen zeroes;
- canonical token-payload SHA-256:
  `240d6a4c4bba8b4a69321cd32586369834d3abb5691dd273adb1cddc07256664`;
- CPU reference logits:
  `[-0.7078226208686829,-0.608864963054657,0.07758018374443054,0.17575061321258545,1.0344152450561523]`,
  canonical digest `cf86a3769256af7bbc418f67aff3798bbd1f1bb41e75c7b3a380b72d15cea5f7`;
- MPS reference logits:
  `[-0.7078224420547485,-0.608864963054657,0.07757969200611115,0.17575055360794067,1.0344157218933105]`,
  canonical digest `55a21300b18e6762ba969a8db2caef3d13c38402f127b65ac63a9487047911b4`;
- comparison contract: `torch.testing.assert_close` with `rtol=1e-5` and
  `atol=1e-6`, label order fixed above.

The reference values were produced by the Phase 3C path-based loader at the
reviewed encoder revision and checkpoint
`34a0fd2a787c6e54ead2ca5397f10f3b67ac8b9fd6f78238439294d4506220ae`.
Tests run that reference and the new byte loader side-by-side; construction also
runs the vector through the selected new path and fails closed on token/mask or
logit disagreement. This fixed conformance inference is the only constructor
inference and contains no request data.

The training manifest must have its exact v1 top-level and nested keys. It must
bind schema `synthetic-training-manifest.v1`, workflow
`phase3b-synthetic-research-training.v2`, `synthetic_only=true`, all three
authorizations false, the exact encoder/revision/registry/frozen values, exact
five-label order, seed/head contract, valid self-digest, and a checkpoint
ledger digest equal to the supplied checkpoint bytes. Training metrics are
validated as finite structured provenance but never interpreted as serving
quality. The packet and embeddings are not required at inference.

## State extraction and request parsing

`parse_request_json` uses a duplicate-key hook recursively before Pydantic, so
duplicate raw keys never disappear. Existing error envelopes stay stable.

The MiniLM backend accepts only the output of `serialize_state`. Its decoder:

- reads exactly four unsigned 64-bit big-endian length-prefixed segments;
- caps the whole frame at 1,000,000 bytes and each length before slicing;
- rejects truncation, overflow, a fifth segment, or trailing bytes;
- decodes the first three segments as strict UTF-8 and requires NFC;
- requires serializer `nfc-jcs-length-prefixed-v1`, locale `pt-BR`, and domain
  `support`;
- decodes the fourth as strict UTF-8 JSON with recursive duplicate-key
  rejection, requires canonical JSON byte equality after NFC/RFC 8785, and
  requires the exact shape `{"text": <string>}`;
- requires text already NFC, non-empty after validation, at most 320 Unicode
  code points and 1,280 UTF-8 bytes.

Tokenization receives only that text, with padding/truncation/max length 128.
`EncodedState.input_tokens` is the integer sum of the post-truncation attention
mask. Text, embeddings, logits, and tokens are never logged or persisted.

## Backend protocol evolution and execution

`EncodedState` keeps `payload: bytes` as its first/default-compatible field and
adds `input_tokens: int = 0` plus a private/opaque runtime value defaulting to
`None`; existing `EncodedState(payload=...)` callers remain valid. The fixture
remains bytes-backed and produces zero tokens. The MiniLM backend's opaque value
contains one CPU float32 `[1,384]` embedding only.

The engine order is:

1. validate request/backend capabilities and immutable workflow;
2. resolve the explicit calibration profile and every question's artifact;
3. compare every calibration context axis;
4. only then serialize and encode state once;
5. score each validated question and transform logits.

Thus missing/incompatible calibration produces zero encode and zero score
calls. The MiniLM backend advertises Choice only, one question, five criteria,
`quality_claims=false`, and rejects direct misuse of question or label schema.

For `mps`, MPS availability is required and
`PYTORCH_ENABLE_MPS_FALLBACK` must be absent/false. CPU is explicit and never an
automatic fallback. The exact Phase 3C path is preserved: tokenizer CPU →
encoder/pooling float32 MPS → synchronize → `.to("cpu")` → CPU float32 linear
head. CPU mode keeps encoder, pooling, and head on CPU. Model/head are eval-only
and inference uses `torch.inference_mode()`.

`encode_state` owns tokenization, encoder/pooling, transfer, and token count;
`score_choice` owns only the CPU head. Engine timing reports tokenization as
zero in Phase 4A because the public protocol cannot split a backend call safely;
`state_encoding_ms` includes tokenization through CPU embedding, and
`decision_ms` includes head plus response construction. MPS synchronization
occurs immediately before device timing starts and after pooling before the CPU
transfer. No per-component performance claim is made from API timing.

Phase 4A is strictly one state/one question, no cache and no batching. Any Phase
4C batching, schema cache, head multiplexing, quantization, compilation, or
different transfer point that may change logits must use a new execution path,
architecture digest, model revision, and calibration.

## CLI and Python API

Python users opt in by constructing `MiniLMRoutingBackend` with all explicit
paths and device. Importing the symbol is lightweight; heavy modules load only
inside construction.

Commands:

- `saracura describe-backend --backend minilm-routing ...` validates and loads
  the model and runs only the public fixed conformance vector, never an operator
  request; it emits only model reference,
  architecture digest, execution path, workflow reference, and identity
  profile—never paths, host data, or model tensors;
- `saracura create-identity-calibration --backend minilm-routing --request ...
  --output ... --created-at ...` requires the request model revision discovered
  above; construction runs only the same public self-authored conformance
  vector, never the operator request, and the command does not score or persist
  request state;
- `saracura decide --backend minilm-routing --request ... --calibration ...`
  executes locally;
- `--backend fixture` remains the `decide` default and rejects MiniLM-only
  flags.

MiniLM commands require `--encoder-snapshot`, `--training-manifest`,
`--checkpoint`, and `--device {mps,cpu}`. No path comes from home, cache, env, or
request data. Argument parsing uses a non-exiting parser: missing/invalid
arguments and file/loader failures share the sanitized v1alpha1 JSON error
envelope and exit 2. `--help` may retain normal help output/exit 0.

## Packaging and dependency boundary

Default dependencies remain Pydantic and RFC 8785. Add `local-minilm` with the
same dependency set currently exposed by `encoder-eval`; keep `encoder-eval` as
a compatibility alias for at least this release. Runtime registry parsing and
verified-byte loading live under `src/saracura`; `src` never imports
`benchmarks`.

Default import, fixture execution, CLI help, and `describe-backend --help` must
not import torch, transformers, tokenizers, safetensors, huggingface_hub, or
benchmark modules and must open no socket. Clean wheel and sdist installs must
provide the identical package registry bytes. Neither distribution may contain
weights, checkpoints, packets, embeddings, generated calibrations, local
paths, or benchmark outputs.

## Implementation scope

Expected changes:

- `backends/base.py`: metadata plus backward-compatible opaque encoded state;
- `backends/fixture.py`: explicit unchanged fixture metadata;
- package-owned registry/verified-byte loader and `backends/minilm.py`;
- `runtime/engine.py`: explicit profiles, pre-encoding calibration resolution,
  backend metadata, and token usage;
- `runtime/workflows.py`: exact five-label workflow revision;
- calibration models/IO/factory: closed identity variant and atomic creator;
- request JSON duplicate-key rejection;
- CLI commands, optional extras, docs, and focused tests.

## Acceptance criteria

1. Existing fixture requests, artifacts, responses, tests, and canonical bytes
   remain unchanged without optional dependencies.
2. Tests prove the exact workflow object/order/digests, stable model-revision
   derivation, runtime-ABI binding, CPU/MPS incompatibility, and direct backend
   rejection of noncanonical labels.
3. Tests mutate every external artifact between open/read/load and exercise
   symlink, hardlink, owner/mode, extra/missing-file, size, digest, malformed
   safetensor, wrong key/shape/dtype/non-finite, and manifest attacks. All fail
   before inference without path/content leakage.
4. Framing tests cover segment count, declared-length overflow, truncation,
   trailing bytes, invalid UTF-8/NFC, noncanonical JSON, recursive duplicate
   keys, state shape, text caps, and post-truncation token counting.
5. Every calibration model/dataset/request axis mismatch—including CPU versus
   MPS—fails before a tokenizer/device/head call. Missing backend metadata and
   missing non-fixture profiles fail stably.
6. Identity output is byte-reproducible, validates its derived ID and real UTC
   timestamp, never overwrites, and rejects any nonzero count, metric, bound,
   status, temperature, or relabelled method. Existing temperature artifacts
   parse and serialize byte-identically.
7. Fakes and the live stack prove one encode/one head call, no cache, exact
   tokenizer and logits conformance with the Phase 3B/3C implementation, MPS
   synchronization, explicit CPU behavior, and no fallback.
8. Clean default/wheel/sdist tests prove lazy imports, identical registry bytes,
   CLI help, no socket use, and absence of private artifacts.
9. With existing ignored Phase 3B-S v3 assets, a live MPS smoke runs
   `describe-backend`, creates an identity artifact, and runs `decide` end to
   end. The result remains `fixture_only`; it is not reported as accuracy.
10. Ruff, formatter, strict mypy, full pytest, manifest validation, build,
    distribution inspection, and `git diff --check` all pass.

## Live validation

The authorized local smoke uses the already verified MiniLM snapshot and
ignored Phase 3B-S v3 manifest/checkpoint. It first runs:

```bash
uv run saracura describe-backend \
  --backend minilm-routing \
  --encoder-snapshot <verified-snapshot> \
  --training-manifest <training-manifest.json> \
  --checkpoint <checkpoint.safetensors> \
  --device mps
```

The emitted immutable revision is placed into a new ignored request. Then:

```bash
uv run saracura create-identity-calibration \
  --backend minilm-routing \
  --request <ignored-request.json> \
  --output <ignored-identity-calibration.json> \
  --created-at 2026-09-22T00:00:00Z \
  --encoder-snapshot <verified-snapshot> \
  --training-manifest <training-manifest.json> \
  --checkpoint <checkpoint.safetensors> \
  --device mps

uv run saracura decide \
  --backend minilm-routing \
  --request <ignored-request.json> \
  --calibration <ignored-identity-calibration.json> \
  --encoder-snapshot <verified-snapshot> \
  --training-manifest <training-manifest.json> \
  --checkpoint <checkpoint.safetensors> \
  --device mps --timing
```

No external API, credential, cost, deployment, or production mutation is
involved.

## Rollout and rollback

This is a package/API increment with no production deployment. Merge only after
default and optional paths pass independently and the live smoke proves the
same tokenizer/logits as Phase 3C. Rollback is a Git revert. External ignored
weights and generated artifacts are not deleted or rewritten by rollback.

## Next gates

Phase 4B obtains an independently authored human PT-BR packet, freezes fit and
blind-test splits before labels are exposed, fits temperature only on fit,
evaluates once on untouched test, and publishes a new immutable dataset profile
and calibration artifact. Until then, MiniLM answers remain `fixture_only`.

Phase 4C may then add batched state encoding, multiple reviewed heads, and
persistent schema caches for high decision volume. Any numerical change gets a
new execution identity; calibration evidence is reused only when conformance
proves the logits path unchanged.
