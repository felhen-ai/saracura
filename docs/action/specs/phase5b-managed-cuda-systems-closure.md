---
title: Phase 5B managed CUDA systems closure
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase5b-managed-cuda-systems-closure.md
globalRef: qmd://saracura/docs/action/specs/phase5b-managed-cuda-systems-closure.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-28
sourceRefs:
  - github:#35
  - benchmarks/results/phase5b-kev4b-managed-cuda-preliminary.json
related:
  - docs/action/specs/phase5b-kev4b-acquisition-and-mac-evaluation.md
  - docs/action/specs/phase5-open-model-benchmark-and-public-readiness.md
  - docs/decisions/0002-open-model-runtime-and-readiness.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 5B managed CUDA systems closure

## Outcome

Saracura will add a checkout-only, host-neutral Linux/CUDA evaluation lane for
the immutable Kev-4B candidate and use it to close the systems evidence that
the preliminary RTX run intentionally omitted. The lane records bounded host
RAM, process RSS, GPU memory, swap, request limits and the real oversize-state
outcome while preserving the existing public TypeSafe-style wire contract.

This increment can promote Kev from `reviewed_acquisition` to `continue` or
`conditional` only in the candidate-research sense: either disposition may
advance to Phase 5C quality evaluation, while `conditional` records a mandatory
Saracura adapter guard before runtime registration. It does not complete
public-readiness Gate A, which also requires
the frozen bilingual development-quality lane. It does not establish
calibration, production readiness or automation authorization.

## Verified starting point

- The immutable acquisition descriptor and preliminary managed-CUDA report
  are on `main` through PR #35.
- The preliminary run completed 240 measured PT-BR/English requests with zero
  invalid outputs but explicitly excluded memory disposition and the formal
  oversize/state evidence.
- The first real oversize canary is invalid evidence: its generated questions
  used Saracura's list-shaped `questions`, singular `instruction` and list
  `criteria` directly, while the pinned Kev wire schema requires a question-id
  mapping, plural `instructions` and a criterion-id mapping. HTTP 422 therefore
  did not isolate the size boundary. P1 must encode the public fixture into the
  real wire shape, prove a valid in-bounds control, then derive each oversize
  payload by changing only state or one question's instructions. The old
  fake-loopback fixture that returned 200 for oversized state is likewise not
  evidence of the pinned upstream behavior.
- The pinned source defines `SERVE_MAX_STATE=65536` tokens and
  `SERVE_MAX_BRANCH=73728` tokens and converts encoder `ValueError` into HTTP
  422. It invokes the encoder without strict state checking, however, and the
  pinned encoder slices state tokens to the limit while exposing no truncation
  flag in the HTTP response. The corrected state probe therefore uses 100,000
  distinct JSON scalar items rather than 100,000 repeated characters, which
  may tokenize below the source limit. A corrected live canary then returned
  HTTP 200 with 79 input tokens for the valid Q=1 control, HTTP 422 for the
  100,000-item branch, HTTP 200 with 65,607 reported input tokens for the
  100,000-item state, and HTTP 401 for the unauthenticated direct-upstream
  control. This confirms the source-predicted silent state truncation while
  isolating every tested boundary.
- A second host canary confirmed that the service unit main PID is the sole Kev
  Python PID and the same PID is attributed approximately 9 GiB by
  `nvidia-smi`; process-to-VRAM attribution is viable without child summation
  for this frozen runtime.
- The managed GPU host already provides an external, clinically preemptible
  lease and restores its protected workload independently of Saracura. That
  private control plane is not copied into this public repository.

## Architecture boundary

The public runner is a benchmark client and sampler, not an HTTP server,
runtime backend, scheduler or GPU arbiter. It never opens, renews or closes a
GPU lease. An operator must establish a bounded external lease, start the
immutable candidate and supply only these explicit local inputs:

- a loopback admission-gateway URL;
- a mandatory loopback direct-upstream URL for the no-authentication probe;
- the expected absolute checkpoint path, used only for in-process identity
  comparison and never emitted;
- the candidate service PID;
- the already observed cold-load duration;
- the acquisition and managed-runtime descriptor paths; and
- a post-run, closed-schema operator receipt that reports only whether the
  external lease preempted the experiment and whether the protected workload
  was restored.

Only literal `127.0.0.1` endpoint hosts are accepted. The service PID
must exist, be owned by the invoking user, and remain the same process for the
entire run. The runner refuses root, arbitrary network endpoints, redirects,
inherited or divergent proxy configuration, caller-supplied provider
credentials and request-triggered downloads. The candidate's exact closed
environment includes only the fixed loopback proxy sink described below; it is
not an egress path. The admission gateway injects its host-only upstream bearer; the
runner sends no Authorization header to that gateway. The direct-upstream
probe also sends no bearer and must receive HTTP 401. A missing URL, transport
failure or any other status produces `blocked_evidence` or `reject_local` as
defined below, never an implicit pass. No bearer is accepted through argv,
environment, file or report in this frozen lane.

The benchmark package remains below `benchmarks/` and outside wheel and sdist.
The default environment does not gain Torch, CUDA, NVML or model dependencies.
Linux sampling uses `/proc`, the standard library and the fixed allowlisted
`/usr/bin/nvidia-smi` executable. Missing mandatory evidence fails closed; it
is never converted to a passing zero.

## Runtime identity and authority

The acquisition descriptor remains authoritative for source, checkpoint and
base ledgers, their immutable revisions, the upstream `uv.lock` digest and
shared limits. Its Darwin/MLX backend, port and physical-footprint fields do not
describe the CUDA lane. The managed-runtime descriptor is separately bound by
SHA-256 in the protocol and is committed as the host-neutral closed manifest
`benchmarks/manifests/phase5-kev4b-managed-cuda.v1.json`. It must declare
exactly Torch, bf16, literal
`127.0.0.1`, the direct-upstream port, the acquisition-descriptor digest and
the same source/checkpoint/base identities. It contains no hostname, operator,
service name, absolute host path, credential or private control-plane detail.
The offline validator rejects unknown fields and any identity not equal to the
pinned acquisition manifest. The protocol binds the exact bytes of both
descriptors.

Before sending a request, the runner reads `/proc/<pid>/cmdline` and
`/proc/<pid>/environ` for the same-user process. It compares in memory and
never writes the raw values. The command must be the capsule Python running
`-m kev.serve`, with `--run` and `--fallback` both equal to the expected
checkpoint, and the expected host and direct-upstream port. Its initial exec
environment is a closed 30-key map. `KEV_API_KEY` must be present and non-empty,
but its value is replaced in memory by a fixed redacted marker and is never
retained. The other fixed literal entries are `LANG=C.UTF-8`,
`LC_ALL=C.UTF-8`,
`PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HTTP_PROXY=http://127.0.0.1:9`, `HTTPS_PROXY=http://127.0.0.1:9`,
`NO_PROXY=127.0.0.1,localhost`, `KEV_BACKEND=torch`, `KEV_DTYPE=bf16`,
`KEV_MERGE=0`, `KEV_FUSED=0`, `KEV_CUDA_GRAPHS=0`, `KEV_DATE_FACTS=0`,
`KEV_LORA_SCALE=1`, `KEV_PREFIX_CACHE=4`, `KEV_PREFIX_MIN_TOKENS=0` and
`KEV_PREFIX_MAX_TOKENS=65536`. The capsule-derived entries are exactly
`HOME=<capsule>/derived`, `XDG_CACHE_HOME=<capsule>/cache`,
`HF_HOME=<capsule>/hf-home`,
`HUGGINGFACE_HUB_CACHE=<capsule>/hf-home/hub`,
`TRANSFORMERS_CACHE=<capsule>/cache`, `TORCH_HOME=<capsule>/cache`,
`TRITON_CACHE_DIR=<capsule>/cache`, `TMPDIR=<capsule>/tmp`,
`PYTHONPYCACHEPREFIX=<capsule>/cache`,
`KEV_BENCHMARK_OUTPUT=<capsule>/reports` and
`PYTHONPATH=<capsule>/payload/source/kev`.

The environment framing must contain exactly one final NUL terminator. The
parser removes only that terminator, then rejects any empty interior entry, an
empty key, a duplicate key or an entry without `=` before constructing the
map. This prevents a later duplicate from hiding the value that the interpreter
would resolve first. Invalid UTF-8 is an explicit identity failure rather than
an uncaught exception. Missing, extra or divergent keys fail closed. The
equality describes the immutable environment passed at exec; it does not claim
to observe later in-process mutations of an environment mapping.

The lexical executable entry, exact module source, checkpoint and fallback
must belong to the same content-addressed capsule root. The module is exactly
`<capsule>/payload/source/kev/kev/serve.py`, the process working directory is
exactly the capsule root, and no top-level root entry may be named `kev` or
start with `kev.` so the working directory cannot shadow the pinned
`PYTHONPATH` module through a package, source, bytecode or extension module.
Only the executable entry has the narrow system-Python resolution exception
defined below; module, checkpoint and fallback still resolve below the capsule.
The two public descriptors remain in the exact
clean Saracura archive and are bound to that runtime by their matching immutable
identities and digests; they are not copied into the private capsule. A
divergent command, environment, identity, lock binding or path fails identity.

The runner also proves request attribution. It maps the direct-upstream LISTEN
socket inode from `/proc/net/tcp` to an fd owned by the exact candidate PID,
then records CPU ticks immediately before and after a valid in-bounds request
through the admission gateway. The PID must remain the socket owner and its CPU
ticks must increase. A different owner, ambiguous socket or no attributable
work produces `blocked_evidence`; it cannot produce a capacity claim.

The wire adapter is explicit and closed. Each public `ChoiceQuestion` becomes
one entry under `questions[question.id]` with `type: choice`, `instructions`
equal to the public instruction and `criteria` mapped from criterion id to
description. The request includes `model: kev-latest`. A successful raw Kev
response must contain exactly `model`, `answers`, `usage` and `latency_ms`; its
model alias, question ids, option ids, finite probability simplex and usage
counters are validated before metrics are accepted. This managed-CUDA package
does not reuse the historical list-shaped fake response as evidence.

## Frozen protocol

The versioned protocol manifest is
`benchmarks/manifests/phase5-managed-cuda-systems.v1.json`. Its closed schema
binds the candidate, acquisition descriptor digest, fixture digest, workload
counts, probe sizes, report schema and thresholds. The initial thresholds are:

- Q=1 p95 at most 5 seconds;
- Q=10 p95 at most 15 seconds;
- Q=50 p95 at most 45 seconds;
- no request above 60 seconds;
- repeat probability delta at most `0.000001`;
- together-versus-separate delta at most `0.04`;
- cold load at most 600 seconds, recorded as operator-attested evidence from
  the official external open operation;
- process peak RSS at most 22 GiB, preserving the stricter acquisition limit;
- peak candidate GPU memory at most 16 GiB on a device with at least 24 GiB;
- candidate `VmSwap` growth at most 1 GiB; host-level swap growth is supporting
  evidence only;
- zero invalid outputs and zero accepted unauthenticated direct-upstream
  requests; and
- HTTP 422 for invalid contract and oversized branch, with a structurally valid
  in-bounds control returning HTTP 200 first. The oversized-state outcome is
  recorded exactly. For this pinned source, only a contract-valid 200 with
  `usage.input_tokens` at least 65,536 and below 73,728 is accepted as
  `accepted_with_pinned_source_truncation` and requires a future Saracura
  adapter guard. An unexpected 422 or other 4xx is protocol drift and produces
  `blocked_evidence`; 5xx/transport is preemption-ambiguous and also blocks;
  a 200 with invalid schema or counters is a positive contract failure and
  produces `reject_local`. The branch
  probe retains one valid choice question and replaces only its `instructions`
  with 100,000 distinct JSON scalar items; the state probe replaces only
  `state` with 100,000 distinct JSON scalar items. Each unambiguously crosses
  the respective pinned token limit without conflating question-count or schema
  validation.

The protocol executes three untimed warmups per locale/workload cell followed
by 20 new-state and 20 cached-state requests for Q=1, Q=10 and Q=50 in both
PT-BR and English. Latency thresholds are evaluated separately for every
`locale x question_count x state_kind` cell (`n=20`), not over an aggregate.
It also runs exact-repeat, together-versus-separate and option-order probes.
Invalid-contract and branch-oversize probes run after the measured matrix and
before its final primary memory sample. The diagnostic oversized-state probe
runs last in a separate evidence window: its latency and resource peaks are
reported but are not evaluated against the primary 60-second, RSS, GPU-memory
or swap thresholds because the required adapter guard forbids that workload.
Failure to obtain its exact outcome still blocks evidence. Each
oversize probe starts from the same valid Q=1 control, which must return 200;
only the probed dimension is enlarged.

The protocol manifest is authoritative for CUDA-only GPU-memory limits and may
tighten, never loosen, a shared acquisition threshold. The acquisition
descriptor remains authoritative for shared cold-load, latency, RSS and swap
ceilings; the protocol uses the equal or stricter value above. Any future
divergence requires a new reviewed schema rather than an in-place threshold
change.

The `/proc` monitor targets a 250 ms sampling interval from before the first
warmup through both the primary and diagnostic windows and fails evidence on a gap above one second. A
separate `nvidia-smi` sampler targets one second and fails evidence on a gap
above three seconds; its command duration and timestamps are recorded so the
sampler cannot silently perturb the latency claim. The monitors record maximum
`VmRSS`, `VmHWM` and `VmSwap` for the exact PID, maximum GPU memory attributed
to that PID, host swap before and after, every monotonic timestamp, all compute
PIDs observed on the same GPU and whether service identity survived. Any
foreign compute PID, PID disappearance, ownership/start-time change, malformed
metric, unavailable sampler, non-finite value or negative delta produces
`blocked_evidence`. Candidate `VmSwap` growth above the threshold is a positive
`reject_local`; host swap growth without matching candidate `VmSwap` growth is
ambiguous and produces `blocked_evidence`.

## Evidence and sanitization

The private raw receipt is atomically written to an absolute operator-selected
path that must be outside the repository root. It may contain the PID, absolute
checkpoint path and exact host samples and must not enter a PR. The public report is separately built from
a closed Pydantic model and contains only:

- immutable source/model/base and protocol digests;
- Saracura commit;
- generic Linux architecture and rounded RAM/VRAM buckets;
- aggregate and per-locale/workload timing;
- bounded RSS, GPU-memory and swap deltas;
- status-only functional, identity, authorization and oversize evidence; and
- one deterministic disposition with explicit scope exclusions.

The `KEV_API_KEY` value is discarded immediately after the non-empty check and
is never inserted into an in-memory report or receipt model. The public report rejects usernames, hostnames, IP addresses, absolute paths,
PIDs, bearer values, service/unit names, environment values, request text and
raw `/proc` or `nvidia-smi` output. Sanitization also rejects every exact
sensitive value observed in the private receipt. It records state oversize as
exactly one of `rejected_without_truncation` or
`accepted_with_pinned_source_truncation`; the latter is valid only for HTTP 200
with a schema-valid response from the exact pinned source and forces
`conditional` even when all other criteria pass.

## Disposition

Disposition precedence is fail closed:

1. `blocked_evidence` when the post-run operator receipt reports preemption,
   any foreign compute PID appears, a 5xx/503/transport failure occurs, a
   mandatory host measurement cannot be collected or sanitized without
   ambiguity, or the candidate PID disappears/changes. This precedence applies
   before any permanent candidate disposition is derived;
2. `reject_local` when identity, functional, security, matrix, invalid/branch
   oversize or a
   hard candidate-attributed memory/latency threshold positively fails with
   only contract-valid responses while the same candidate process remains
   proven alive and the operator receipt reports no preemption; or
3. `conditional` when every criterion except strict upstream state rejection
   passes and the exact pinned source returns a contract-valid 200 for the
   oversized-state probe; or
4. `continue` only under a future reviewed protocol/source binding that expects
   and observes strict state rejection. This v1 protocol cannot emit
   `continue`: an unexpected 422 from its exact pinned source is drift and
   therefore `blocked_evidence`, not a stronger claim.

`continue` and `conditional` authorize only the next reviewed Phase 5C quality
experiment. `conditional` additionally requires a later, separately reviewed
Saracura adapter to reject states whose canonical UTF-8 JSON exceeds 32 KiB
before invoking Kev; it cannot authorize runtime registration by itself. The
public-readiness manifest remains unchanged with Gate A `not_met` until Phase
5C supplies its separate bilingual quality evidence.

## Delivery phases

### P1 — public runner

Add both closed manifests (protocol and public managed-runtime descriptor), the
Linux/CUDA runner, report model, deterministic disposition, two-step CLI and
offline/fake-loopback tests. The execution step writes private provisional
evidence; the finalize step consumes the closed post-run operator receipt,
applies preemption precedence and alone may emit the sanitized public report.
Add a source-compatible fake loopback for the new package and revise the
historical client documentation so its list-shaped fake is not presented as
current pinned-runtime evidence; do not make the CUDA lane depend on that
historical response model. The new probes must use the valid-control-derived
contract and distinguish strict rejection from pinned-source truncation. Tests
must cover PID reuse/disappearance, socket ownership and CPU attribution, ownership,
malformed proc data, missing and malformed `nvidia-smi`, swap growth,
foreign GPU PIDs, 5xx/transport failures, operator preemption override,
oversize-state acceptance, secret/path sanitization, atomic no-clobber output
and every disposition branch. The P1 archive environment is provisioned with
`uv sync --frozen --offline`; the runner records the `uv.lock` digest and a
digest of its own benchmark package for comparison with the clean `git
archive`. P1 also adds the closed validator for
`phase5-open-model-candidates.v3`, binding its `supersedes_manifest_sha256` to
the exact v2 bytes. For Kev `continue`, v3 requires the exact public systems
report digest, a report disposition of `continue`, the protocol digest and an
unchanged public-readiness manifest whose Gate A remains `not_met`. P1 ships
the same bindings for `conditional` plus the exact
`state_size_guard_required=true` mitigation marker. `reject_local` also binds
the exact report, protocol, predecessor and unchanged readiness manifest, but
sets `allowed_claims` to an empty list and cannot retain
`candidate_for_evaluation`. P1 ships the schema and tests with nullable result
fields; P2 supplies only the reviewed data instance. The offline manifest
validator also validates the known committed systems-report path directly, so
a `blocked_evidence` report cannot bypass schema and sanitization checks merely
because no v3 successor is emitted.

P1 ends only after PR checks are green, the PR is merged and `main` is green.

#### Live identity correction after P1

The first P2 attempt from merged P1 bytes stopped before issuing measured
requests because `Path(argv[0]).resolve()` followed the venv `bin/python`
symlink to its system-Python target before locating the capsule. Runtime
identity instead anchors the content-addressed capsule in a normalized absolute
lexical `argv[0]` entry. `.` and `..` components, empty or relative entries and
more than one 64-lowercase-hex ancestor fail closed. The capsule root must equal
its own resolved path, and every component from it through `environment/bin`
must be a real root-owned directory that is not group- or world-writable.

The lexical entry is exactly `environment/bin/python`. It may be a root-owned,
non-group/world-writable regular executable inside that capsule, or the single
launcher-compatible symlink exception at that same path. The symlink entry may
resolve through at most eight hops, such as
`python -> python3 -> /usr/bin/python3 -> python3.X`, only while
each hop stays below the real capsule `environment/bin`, `/usr/bin` or
`/usr/local/bin`. Each absolute or relative hop target is normalized against
its containing directory before containment is evaluated; ambiguous, looping
or escaping chains fail closed. The final target must be a root-owned regular executable
below `/usr/bin` or `/usr/local/bin` and must not be group- or world-writable.
For both regular and symlink entries, the final target must equal the live
`/proc/<pid>/exe` target by normalized path and device/inode identity. A
deleted or replaced executable fails closed. No module,
checkpoint, fallback or other capsule path receives this escape.

Production defaults for allowed system prefixes are fixed. Narrow internal
test seams may substitute path metadata, symlink reads, proc-exe evidence and
approved roots only to make every branch falsifiable; they do not become CLI or
public API inputs. Regression tests use a real two-hop venv-style symlink and
assert a distinct failure reason for relative or non-normalized entry, missing
or multiple capsule roots, symlinked/writable/non-root capsule ancestors, wrong
entry name, dangling/looping/escaping hops, unapproved prefix, non-root,
group/world-writable, non-regular or non-executable target, and proc-exe
mismatch. The blocked first attempt stays private and does not publish a
systems report or candidate-manifest successor. P2 is retried only from a new
exact clean `main` archive after this correction is reviewed, merged and green.

The live capsule was checked read-only before implementation: `/srv`,
`/srv/models` and `/srv/models/saracura` are root-owned `0755`; the exact
content-addressed root, `environment` and `environment/bin` are root-owned
`0555`; its `python` entry points to `/usr/bin/python3.12`; `/usr` and
`/usr/bin` are root-owned `0755`; and the final Python is a root-owned regular
`0755` executable. This proves the deployed topology satisfies the corrected
ancestor and final-target invariants before another lease is attempted.

#### Live launcher identity correction after the second private attempt

The next P2 retry from merged correction bytes also stopped before the first
measured request. The executable gate passed, but the runner then looked for
the simplified `<capsule>/kev/serve.py` fixture path. The immutable capsule
instead contains the source at
`<capsule>/payload/source/kev/kev/serve.py`, with the source directory bound by
the launcher's exact `PYTHONPATH`. Read-only host evidence confirmed that the
descriptor-aligned path is a root-owned `0444` regular file below a root-owned
`0555` source directory and that the simplified path is absent.

The same comparison found sequential blockers that would otherwise require
more live retries: the launcher deliberately emits the two Python path entries,
the fixed loopback proxy sink and the remaining cache/report paths enumerated
above, while the runner still rejected every Python- or proxy-named entry and
unknown `KEV_*` entries. Runtime identity therefore validates the complete
30-key launcher map at once. A versioned test fixture records that complete
map and a parity test checks the runner expectation against it. Regression
fixtures reproduce the real module layout and exact environment; they reject
each missing or changed key, every extra key, duplicate/interior-empty/
malformed framing, invalid UTF-8, cwd shadowing and module escape. Both
benchmark-defect attempts remain private and produce no
systems report or candidate-manifest successor. P2 restarts only from another
new clean `main` archive after this correction is reviewed, merged and green.

#### Live response-contract correction after the third private attempt

The next P2 retry passed executable, module, environment, socket and ownership
identity and reached the gateway attribution control, but the client rejected
the first valid Kev Choice answer as `answer schema is invalid`. The immutable
source contract emits each Choice answer with exactly four fields:
`type="choice"`, `choice`, `confidence` and `probabilities`. The runner fixture
and validator had incorrectly modeled only the latter two fields. This is a
benchmark-contract defect, not candidate evidence, so the attempt remains
private and produces no systems report or candidate-manifest successor.

The corrected validator accepts exactly those four fields, requires the type
literal `choice`, validates `confidence` as a finite numeric value in `[0,1]`,
and retains the existing exact question/option identity, probability simplex,
choice membership, usage and latency checks. Missing, extra, mistyped,
non-finite or out-of-range fields fail closed. The real-contract positive
fixture is pinned to the immutable `to_answers` Choice shape rather than to a
historical fake.

Candidate rejection during the gateway attribution control is retained while
the runner completes the post-control identity proof. It becomes
`reject_local` with `candidate_invalid_response` only after stable PID
starttime, positive CPU delta, same owner and same LISTEN socket prove that the
invalid answer came from the candidate. PID or socket drift, disappearance or
missing CPU attribution takes precedence and remains `blocked_evidence` with
`identity_invalid_or_changed`. The selected result is then atomically
persisted instead of escaping without a provisional artifact. P2 restarts only
from a new clean `main` archive after this correction is independently
reviewed, merged and green.

#### Live failure-report binding correction after the fourth P2 attempt

The fourth P2 attempt used the reviewed response contract, attributed the gateway control to the stable candidate process and atomically recorded a candidate-invalid response because the returned probabilities did not sum to one. The official operator then closed the lease without preemption and proved that the protected clinical workload was restored. Finalization therefore produced the closed, sanitized `phase5b-managed-cuda-failure-report.v1` with disposition `reject_local`.

The committed-manifest validator nevertheless attempted to parse every bound report only as `phase5b-managed-cuda-systems-report.v1`. That contradicts the existing P1/P2 contract: an attributed early `reject_local` is intentionally a failure report and must still bind a v3 successor, while `blocked_evidence` must not publish one. This is a public binding-validator defect, not a reason to fabricate the uncollected 240-request systems fields or to rerun the candidate.

The validator must discriminate the two already closed public report schemas. The full systems-report path keeps its current provenance checks unchanged. The failure-report path is eligible for a bound v3 successor only when its classification and disposition are both `reject_local`, its evidence code is exactly `candidate_invalid_response`, operator preemption is false and protected-workload restoration is true. Its exact bytes remain bound by `systems_report_digest`; the manifest must also bind the exact protocol, v2 predecessor and unchanged readiness manifest. `allowed_claims` and `mitigation_markers` remain empty. Any `blocked_evidence`, preempted, unrestored, malformed or unknown report stays ineligible for v3. Tests must cover both report schemas, every failure-report eligibility gate and the unchanged prohibition on `blocked_evidence` successors before the two P2 data artifacts are committed. Negative preemption, restoration and evidence-code cases must mutate raw JSON on disk so tests prove whether the closed report model or the binding gate rejects each inconsistency; an unknown report schema must also fail at the binding boundary.

### P2 — live systems evidence

From the exact clean reviewed `main` commit containing P1 and any prerequisite
runner correction, establish the external managed-CUDA
lease, run the host-local benchmark, close the lease on every path and verify
the protected workload is restored. Keep raw evidence private. Add only the
sanitized report, its manifest binding and the candidate-manifest successor.

A runner preflight failure before any measured request is not a candidate P2
outcome when operator evidence attributes it to a defect in the benchmark
itself rather than to the candidate runtime. Preserve that attempt privately,
close and restore the protected workload, correct and re-review the runner, and
restart P2 from the new exact clean `main` archive. This exception cannot
reclassify a candidate-attributed identity mismatch from reviewed runner bytes,
which retains the normal disposition rules below.

If the run returns `continue` or `conditional`, the successor applies that
exact disposition while retaining only `candidate_for_evaluation`; conditional
also binds the required state-size guard. A positive `reject_local` result
changes the candidate to `reject_local` and binds that report. `blocked_evidence` preserves
`reviewed_acquisition` because the experiment is invalid rather than negative;
P2 publishes the sanitized blocked report but no v3 candidate manifest.
Gate A remains `not_met` in every P2 outcome.

## Acceptance criteria

1. The new protocol and managed-runtime manifests have closed,
   offline-validated schemas and bind the exact acquisition descriptor, public
   fixture, immutable runtime identities and equal-or-stricter thresholds.
2. The runner accepts only literal-IPv4 loopback endpoints and a same-user
   stable PID, verifies the closed command/environment runtime identity without
   retaining its secret, and performs no download or lease action.
3. The measured bilingual 240-request matrix and all repeat, isolation,
   permutation, invalid and oversize probes are deterministic and fail closed;
   each oversize probe follows a valid 200 control and changes only its probed
   dimension.
4. Linux RSS/HWM/VmSwap, GPU memory and host swap are sampled concurrently;
   the direct LISTEN socket and gateway-served CPU work are attributed to the
   same PID, and missing or ambiguous evidence cannot produce `continue`.
5. The public report is atomically created, closed-schema, content-free and
   contains no private host, path, PID, credential, request or raw telemetry.
6. Default install, wheel and sdist remain lightweight and free of weights,
   candidate source, benchmark artifacts and CUDA dependencies.
7. Offline tests cover every security, capacity, sanitization and disposition
   branch without network or model bytes.
8. P1 also delivers the v3 candidate schema/validator that cryptographically
   binds a later Kev disposition to the exact systems report, protocol, v2
   predecessor and unchanged readiness manifest, and cannot encode
   `conditional` without the mandatory adapter-guard marker.
9. P1 and every prerequisite runner correction are merged and green before P2
   executes from the resulting exact clean `main` SHA; the operator creates a
   `git archive` of that SHA, records its digest and runs those exact bytes
   host-locally. A benchmark-defect preflight attempt remains private evidence,
   not a candidate disposition.
10. P2 closes the external lease and proves protected-workload restoration even
   on benchmark failure or interruption; a preemption receipt overrides all
   provisional failures to `blocked_evidence` before publication.
11. Candidate promotion never changes public-readiness Gate A, calibration,
   production or automation claims.
12. Offline runtime-identity regression tests use a real two-hop venv-style
    Python symlink and assert distinct rejection reasons for every unsafe
    lexical entry, ancestor, hop, final target and `/proc/<pid>/exe` mismatch.
13. Runtime-identity regression tests reproduce the immutable
    `payload/source/kev/kev/serve.py` layout and the full closed 30-key launcher
    environment; they reject missing, extra, changed, duplicate, empty,
    malformed or invalid-UTF-8 entries, invalid NUL framing, divergent
    capsule-derived paths, cwd shadowing and module escape before measured
    work; a versioned fixture asserts parity with the 30-key launcher map.
14. Response-contract tests reproduce the immutable Kev Choice answer with
    exactly `type`, `choice`, `confidence` and `probabilities`; they reject
    missing, extra, mistyped, non-finite and out-of-range fields, and a
    candidate rejection during attribution creates a closed private
    `reject_local` provisional only after post-control attribution succeeds;
    identity drift or absent CPU attribution instead creates
    `blocked_evidence` before returning failure.
15. The v3 binding validator discriminates the two closed public report schemas. A failure report can publish a successor only when classification and disposition are `reject_local`, evidence code is `candidate_invalid_response`, the operator was not preempted and the protected workload was restored; raw-JSON regression tests cover every gate, malformed input, unknown schemas and the continued prohibition on a `blocked_evidence` successor.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests benchmarks
uv run pytest -q
uv run python -m benchmarks.validate_manifests
uv run python benchmarks/validate_default_environment.py
uv build
uv run python benchmarks/inspect_wheel.py dist/*.whl dist/*.tar.gz
```

The live command is intentionally documented only after P1 fixes its final CLI
shape. It must use loopback endpoints and write raw evidence outside Git.

## Rollback

- Before merge: discard only the isolated feature branch/worktree changes.
- After P1 merge: leave the unused checkout-only runner in place or revert its
  PR; installed Saracura behavior is unchanged.
- During P2: stop the benchmark client, close the external lease through its
  official operator and verify protected-workload restoration. Never edit the
  arbiter database, service profile or immutable capsule manually.
- After a bad public result: revert only the result/candidate-manifest PR. The
  immutable acquisition and historical preliminary report remain intact.
