---
title: Phase 4C.3c TypeSafe response-contract recovery
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase4c3c-typesafe-response-contract-recovery.md
globalRef: qmd://saracura/docs/action/specs/phase4c3c-typesafe-response-contract-recovery.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-22
sourceRefs:
  - https://docs.typesafe.ai/api
  - https://docs.typesafe.ai/primitives/choice
  - https://docs.typesafe.ai/sdk/python/api/types/responses
  - https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py
related:
  - AGENTS.md
  - SECURITY.md
  - docs/action/specs/phase4c-universal-backend-bakeoff.md
  - docs/action/specs/phase4c3-controlled-remote-comparison.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 4C.3c TypeSafe response-contract recovery

## Context

The first controlled Phase 4C.3b TypeSafe run produced a valid, sealed partial
artifact. Requests 0 through 9 settled successfully, covering 19 decisions.
Request 10, the first question with 20 Choice options, returned HTTP 200 but
failed the local response validator and was sealed as `response_invalid`.

The artifact intentionally retained no provider response body, selected label,
probabilities, confidence, prompt, state, or exception text. That redaction is
correct and remains binding. It also means the exact failed predicate cannot be
recovered from the existing artifact.

The live TypeSafe contract now describes Choice probabilities as summing
"approximately" to one. The official Python SDK validates each probability as
a float but does not enforce an exact distribution sum or re-check the selected
maximum. Phase 4C.3 instead required an absolute sum error at most `1e-6`.
Failure at the first 20-option response is therefore consistent with cumulative
wire rounding, but this is only a hypothesis until a bounded structural
diagnostic is produced by a new response.

The existing artifact manifest binds the exact Phase 4C.3 `plan.py` and
`runner.py` source bytes. Modifying either file would make independent
validation of that predecessor fail. Recovery must therefore be additive and
must not modify any Phase 4C.3 source, plan, operator documentation, claim, or
artifact.

The predecessor manifest did not bind the imported Phase 4C.2 materializer or
the wire digest of each request. It is therefore impossible to retroactively
prove the exact content sent for ordinals 0 through 9 from the artifact alone.
The recovery reports this limitation. It freezes all 57 currently reviewed wire
digests before implementation and uses those digests as the executable
reference for every continuation request.

## Goal

Add a checkout-only, fail-closed Phase 4C.3c continuation runner that:

1. independently validates the sealed Phase 4C.3 partial artifact and all seven
   of its governed predecessor inputs;
2. starts at the first unsettled request, ordinal 10, and can dispatch only the
   remaining ordinals 10 through 56 once each;
3. classifies any response-contract failure with a bounded structural category
   without retaining provider content;
4. accepts only a narrowly defined approximate probability sum suitable for
   systems-only evidence;
5. combines the ten predecessor settlements with new settlements into a new
   immutable 57-request result; and
6. remains inside the existing cumulative USD 0.25 authorization.

This is Phase 4C.3c recovery, not Phase 4C.4. Phase 4C.4 remains the architecture
decision record and begins only after the intended evidence lane is complete or
explicitly declared insufficient.

## Non-goals

This phase does not retry or replace any settled request, erase or amend the
original failure, mutate the original journal, claim quality or calibration,
publish confidence, retain raw answers, train a model, register a runtime
backend, change the fixed TypeSafe endpoint or model, or increase the USD 0.25
ceiling.

It does not use the TypeSafe SDK at runtime, because the reviewed direct HTTPS
transport is sufficient and the default installation must remain dependency
light. The SDK and live documentation are contract evidence only.

It does not interpret the approximate sum as calibrated probability evidence.
The probabilities are checked only to reject malformed responses and are then
discarded. A complete result remains systems-compatibility evidence.

## Frozen recovery plan

Create `benchmarks/manifests/phase4c3c-resume-plan.v1.json` with a closed schema
that binds:

- the exact bytes of `universal-bakeoff-plan.v3.json`;
- the literal provider, endpoint, and `jev-1.13.0` model from Phase 4C.3;
- `plan_id=phase4c3c-typesafe-response-contract-recovery`;
- `evidence_lane=synthetic_research`;
- `first_request_ordinal=10`, `last_request_ordinal=56`, and
  `continuation_request_count=47`;
- exactly one attempt and a 60-second timeout;
- `response_body_ceiling_bytes=1048576`;
- canonical reservation cost `0.0032` per request, computed input-token rate `0.042` per
  million, and cumulative ceiling `0.25`;
- `probability_sum_absolute_tolerance=0.01`;
- SHA-256 of `benchmarks/universal_local/plan.py`,
  `benchmarks/universal_local/registry.py`,
  `benchmarks/universal_local/inputs.py`, plans v1 and v2, registry v2, and the
  full 40-hex Git SHA-1 object ID of the source commit whose Phase 4C.3
  `plan.py` and `runner.py` hashes equal the sealed predecessor manifest;
- `required_partial_contract` with exactly schema
  `phase4c3-result.v1`, status `partial`, failure `response_invalid`, 10
  completed requests, and 19 completed decisions;
- a closed 57-entry `wire_sha256_by_ordinal` map generated from the reviewed
  materializer, with keys `0` through `56` and no gaps;
- the exact bounded diagnostic-category set in this spec; and
- `quality_labels=false` plus
  `conclusion_authority=systems_compatibility_only`.

The plan validator hashes the actual Phase 4C.3 plan, rejects aliases, unknown
fields, duplicate keys, reordered or expanded ordinal ranges, additional
providers, larger budgets, changed attempts, a changed source digest, a missing
wire digest, materialized wire bytes that do not match their frozen ordinal, or
a looser tolerance.

It also uses Git object reads to verify that the pinned source commit exists and
that the materializer sources at that commit have the same hashes as the
current prohibited sources. This does not add information to the old artifact,
but it strengthens the operational evidence that the continuation rematerializes
the same reviewed matrix. The pinned commit must be reachable from `main`;
`a461a5eba12a2777e73fb1e0ccb259bebf8d4cb5` is the governed Phase 4C.3 merge,
not an ephemeral feature-branch commit.

`benchmarks.validate_manifests` routes the new schema to this full validator.
Because that command also runs in the `universal-local` CI job, this phase adds
`fetch-depth: 0` to that job's checkout. The proof is never silently weakened in
a shallow clone; absence of the pinned object fails closed.

## Immutable predecessor contract

The continuation CLI accepts the same seven explicit predecessor inputs as
Phase 4C.3 plus `--phase4c3-partial-artifact-dir`. Before any output, claim, or
network action it:

1. runs the unchanged Phase 4C.3 preflight over the seven inputs;
2. invokes the unchanged Phase 4C.3 artifact validator against the partial
   artifact;
3. requires exact schema `phase4c3-result.v1`, `status=partial`,
   `failure_category=response_invalid`, 10 completed requests, and 19 completed
   decisions;
4. requires the journal to contain settlements for ordinals 0 through 9,
   represented as exactly ten ordered `reserve`/`settle` pairs, followed by one
   reservation and one `response_invalid` failure for ordinal 10, with no later
   row;
5. recomputes the predecessor journal charge and requires it to match the
   partial result; and
6. binds the exact predecessor artifact-manifest SHA-256 into the claim,
   continuation evidence, result, and new artifact manifest.

The repository recovery plan intentionally does not contain the digest or cost
of one ignored local artifact. It freezes the required partial contract above;
production preflight derives the actual manifest digest and charge, then binds
them through the single-use claim. Tests can therefore construct synthetic
partials without a production digest override or mutable plan path.

For ordinals 0 through 9, the validator checks that the predecessor samples'
scenario, state ordinal, question offset/count, and option-count bounds match
the frozen materialization. This proves structural alignment but does not
retroactively prove the request text; the report states that limitation. For
ordinals 10 through 56, the runner compares the complete wire SHA-256 to the
frozen per-ordinal digest immediately before every dispatch.

The implementation is a new package under
`benchmarks/universal_remote_resume/`. It may import the Phase 4C.3 validators
and request materializer, but must not edit files under
`benchmarks/universal_remote/`, `universal-bakeoff-plan.v3.json`, or the Phase
4C.3 spec and operator runbook. This preserves validation of the source-bound
partial artifact.

It also must not modify `benchmarks/universal_local/plan.py`,
`benchmarks/universal_local/registry.py`,
`benchmarks/universal_local/inputs.py`,
`benchmarks/manifests/universal-bakeoff-plan.v1.json`,
`benchmarks/manifests/universal-bakeoff-plan.v2.json`, or
`benchmarks/manifests/decision-backend-candidates.v2.json`. The new manifest
binds all of those sources in addition to every new Phase 4C.3c source.

The ignored local predecessor artifact remains outside Git. Tests construct
synthetic partial artifacts through fake transports; they do not copy the live
artifact into fixtures.

## Response validation and bounded diagnostics

The new validator preserves the closed Phase 4C.3 response contract:

- root fields are exactly `model`, `answers`, and `usage`;
- resolved model is exactly `jev-1.13.0`;
- answer keys exactly match the request question keys;
- each answer has exactly `type`, `choice`, `probabilities`, and `confidence`;
- answer type is `choice` and the selected label is one of the declared labels;
- probability keys exactly equal the declared criteria keys;
- every probability and confidence is a finite JSON number in `[0,1]` and is
  not a boolean;
- usage fields are exactly `input_tokens` and `output_tokens`, both
  non-negative integers; and
- the selected label has a maximal returned probability.

The only relaxed predicate is distribution normalization. The sum of returned
probabilities must be within an absolute `0.01` of one. This tolerance is fixed,
not configurable, and does not grow with option count. It is appropriate only
because this lane discards the values and draws no quality or calibration
conclusion. A response outside the bound still fails closed. Future work that
uses probabilities must define and validate its own stricter numerical
contract.

The strict response parser decodes JSON floating-point tokens directly to
`Decimal` and preserves their textual value only in process memory. Each
finite Decimal is converted to an exact `Fraction` for summation, and the
validator accepts the inclusive rational boundary
`absolute_error <= Fraction(1, 100)`. It neither round-trips through binary
`float` nor depends on the ambient Decimal context. The limit is not claimed to
reflect a provider serialization
precision: with three-decimal rounding, 20 options can accumulate `0.01`, while
77 options could accumulate `0.0385`. The fixed limit deliberately does not
expand for ordinal 56: a larger drift there is a valid structural failure, not
permission to widen the contract. Conversely, the earlier five-option
responses passed `1e-6`, so the rounding hypothesis is not strong. Ordinal 10
is a diagnostic probe as well as the first missing sample; it may instead
identify key truncation, choice/maximality drift, or another closed failure.
The `0.01` tolerance is acceptable here only because no probability-derived
metric survives reduction.

Before Fraction conversion, each probability and confidence Decimal must have
at most 1,000 coefficient digits and an absolute exponent at most 1,000, checked
through `Decimal.as_tuple()` without constructing a large integer. Exceeding
either bound is `probability_invalid` or `confidence_invalid`. This accepts the
more-than-28-significant-digit boundary tests while preventing a compact value
such as `1e-999999999` from consuming unbounded CPU or memory.

The reduced settlement adds one boolean,
`probability_sum_tolerance_used`, which is true when the absolute error exceeds
the old `1e-6` threshold but remains within `0.01`. It retains neither the sum,
the error, the number of decimal places, nor any probability value. If ordinal
10 settles with this boolean true, the new evidence supports the rounding
hypothesis without exposing the response.

Validation failures carry a private typed exception whose public diagnostic is
exactly one of:

- `json_invalid`;
- `root_shape`;
- `answer_key_mismatch`;
- `usage_shape`;
- `usage_invalid`;
- `answer_shape`;
- `answer_type_mismatch`;
- `choice_unknown`;
- `probability_key_mismatch`;
- `probability_invalid`;
- `probability_sum_out_of_tolerance`;
- `choice_not_maximal`;
- `confidence_invalid`;
- `unexpected_response_type`;
- `validator_internal_error`; or
- `response_too_large`.

`model_drift` remains its existing top-level failure category. Every other
category above is stored only as `response_invalid_detail` alongside the
top-level `response_invalid`. No key name, index, count beyond already approved
request metadata, value, body excerpt, exception text, label, or digest of
provider content is retained. HTTP and transport failures have no response
detail.

Diagnostics use this deterministic precedence: response-size ceiling, strict
JSON decoding, root shape, model identity, answer-key equality,
usage shape, usage values, then each answer in request insertion order with
answer shape, answer type, selected-choice membership, probability-key
equality, probability numeric validity, confidence validity, probability-sum
tolerance, and selected-choice maximality. Invalid UTF-8, duplicate JSON keys,
JSON `NaN`/`Infinity`, and malformed JSON are `json_invalid`; a decoded numeric
overflow such as `1e400` is `probability_invalid` or `confidence_invalid`
according to its field.

Strict decoding does not require RFC 8785 or any provider key order, spacing,
or number spelling. Tests accept semantically valid responses with reordered
keys, whitespace, `0.10`, and `1e-3`. JSON integers that exceed the interpreter's
safe parse limit fail decoding as `json_invalid`. Every field is type-checked
before membership, ordering, conversion, or arithmetic so lists, objects,
booleans, and other unexpected types cannot raise unclassified errors.
`unexpected_response_type` covers a decoded field with a type outside its
contract when no more specific type category above applies.
`validator_internal_error` is the final fail-closed category for any unforeseen
exception after HTTP 200; it retains no exception text or body and is tested by
an injected validator fault.

Wrong decoded types map deterministically as follows:

| Field | Wrong decoded type | Category |
|---|---|---|
| response root | anything except object | `root_shape` |
| `model` | anything except string | `unexpected_response_type` |
| `answers` | anything except object | `unexpected_response_type` |
| `usage` | anything except object | `unexpected_response_type` |
| one answer | anything except object | `unexpected_response_type` |
| answer `type` | anything except string | `unexpected_response_type` |
| answer `choice` | anything except string, including array/object | `unexpected_response_type` |
| `probabilities` | anything except object | `unexpected_response_type` |
| one probability | boolean, string, array, object, null, or non-finite number | `probability_invalid` |
| `confidence` | boolean, string, array, object, null, or non-finite number | `confidence_invalid` |
| either usage value | anything except non-negative non-boolean integer | `usage_invalid` |

For correctly typed containers, wrong field sets map to `root_shape`,
`answer_key_mismatch`, `usage_shape`, `answer_shape`, or
`probability_key_mismatch` according to the validation stage. A string model
different from `jev-1.13.0` is `model_drift`; a string answer type different
from `choice` is `answer_type_mismatch`; and a string choice outside criteria
is `choice_unknown`.

Container type checks occur immediately on entering their named validation
stage, before any key-set comparison: `answers` before answer-key equality,
`usage` before usage shape, each answer before answer shape, and probabilities
before probability-key equality.

Usage integers above JSON's interoperable safe-integer domain `2^53-1` are also
`usage_invalid`, before cost computation or journal canonicalization. This
prevents an unrepresentable settlement from leaving only an understated open
reservation.

Malformed nesting that raises `RecursionError`, the interpreter integer-digit
limit's `ValueError`, and all JSON decoder exceptions are `json_invalid`.

The transport reads at most 1,048,577 bytes for every status. A successful HTTP
response larger than 1,048,576 bytes is discarded and classified as
`response_too_large`. Non-200 bodies are bounded by the same read limit, never
parsed, and discarded before the existing status classification.

The 60-second urllib timeout remains a blocking network-operation timeout, not
an independently enforced wall-clock deadline for a peer that continuously
dribbles bytes. Measured total latency may therefore exceed 60 seconds. This is
a documented inherited transport limitation, not permission to retry.

The response body exists only in process memory long enough to parse, validate,
and reduce it. Tests scan every artifact file for forbidden response fields and
sentinel provider content.

## Continuation journal and state machine

Create a new append-only canonical hash-chain journal. It is not an extension
or copy of the old journal. Its lineage is the SHA-256 of the predecessor
artifact manifest and its first permitted ordinal is 10.

Journal rows use schema `phase4c3c-journal-row.v1`. Every row has exactly
`sequence`, `kind`, `ordinal`, `lineage`, `previous_sha256`, and `row_sha256`,
plus the kind-specific fields below:

- `reserve`: canonical `reservation_cost_usd="0.0032"` and `wire_sha256`;
- `settle` and `settled_over_limit`: `sample` and `computed_cost_usd`;
- `failure`: `category` and `response_invalid_detail`; and
- `halt`: `category=budget_uncertain` and
  `response_invalid_detail=null`.

Exactly one `reserve` precedes at most one `settle`, `settled_over_limit`, or
`failure` row for each sequential ordinal 10 through 56. A `failure` after a
reservation stores a bounded top-level category; its detail is non-null only
for `response_invalid`. A `halt` is permitted only before reserving the next
expected ordinal when the dynamic budget formula fails. The first terminal
failure, over-limit settlement, or halt stops the run. There is no retry, skip,
fallback, or second settlement. An open reservation is permanently charged at
the full reservation amount.

Written `failure` categories are exactly `http_error`, `rate_limited`,
`provider_unavailable`, `timeout`, `transport_error`, `response_invalid`, and
`model_drift`. `budget_uncertain` exists only in `halt`; `usage_overflow` is
derived only from `settled_over_limit`; and `process_interrupted` is never a
journal row.

The `wire_sha256` in every reservation must equal both the frozen plan entry and
the bytes dispatched next. This makes the request identity independently
visible without retaining or hashing any provider response.

Dispatch sends the already materialized `meta["wire"]` bytes that were checked
and reserved; it must not reserialize `request`. Fake transports are subject to
the same 1,048,577-byte bounded-response contract as the real transport.

The continuation settlement sample has exactly the 14 Phase 4C.3 sample fields
plus `probability_sum_tolerance_used`. The value is the bounded boolean derived
by response validation.

The continuation result reconstructs a composite view only after independently
validating both journals. Predecessor settlements for ordinals 0 through 9 are
read from the immutable Phase 4C.3 journal. New settlements must begin at 10.
Duplicate, missing, reordered, or overlapping ordinals fail validation.

A composite sample uses schema `phase4c3c-composite-sample.v1`. It contains the
14 Phase 4C.3 sample fields plus exactly:

- `measurement_source`, either `phase4c3_predecessor` or
  `phase4c3c_continuation`;
- `response_validation_contract`, either `strict_sum_1e-6` or
  `approx_sum_0.01`; and
- `probability_sum_tolerance_used`.

For predecessor samples, the original 14 field values are copied exactly,
`measurement_source=phase4c3_predecessor`,
`response_validation_contract=strict_sum_1e-6`, and the tolerance boolean is
the derivable value `false`. For continuation samples, the source and contract
identify Phase 4C.3c and the boolean comes from the new validator. The original
journal remains the byte-authoritative representation of the old samples; the
composite transformation is deterministic and separately validated.

The claim schema is `phase4c3c-claim.v1` and contains exactly
`schema_version`, `lineage`, and `binding`. The binding contains exactly the
keys `recovery_plan`, `phase4c3_partial`, `plan_v3`, `prior_ledger`, `training`,
`packet`, `phase3c`, `phase3d`, `laya`, and `mdeberta`.
The claim `lineage` is exactly the prior-ledger SHA-256; the continuation
journal lineage remains exactly the partial artifact-manifest SHA-256.

The `predecessor_partial` object in both result shapes contains exactly
`artifact_manifest_sha256`, `result_sha256`, `journal_sha256`, `claim_sha256`,
`completed_request_count`, `completed_decision_count`, and `charged_cost_usd`.
The fixed expected counts are 10 and 19; the charge is recomputed from the old
journal. The `predecessors` object in both result shapes and in the new manifest
equals the claim binding exactly, including `recovery_plan` and
`phase4c3_partial`.

A `phase4c3c-result.v1` complete result contains exactly `schema_version`,
`status`, `systems_samples`, `summary`, `continuation_summary`, `per_scenario`,
`local_comparison`, `remote_charge`, `predecessor_partial`, `predecessors`,
`failure_category`, and `response_invalid_detail`. It contains all 57 composite
samples in ordinal order. `summary` and `per_scenario` cover the composite 57;
`continuation_summary` covers only the 47 newly measured requests. Failure and
detail are null.

A `phase4c3c-result.v1` partial result contains exactly `schema_version`,
`status`, `completed_request_count`, `completed_decision_count`,
`failure_category`, `response_invalid_detail`, `remote_charge`,
`predecessor_partial`, and `predecessors`. Counts are cumulative across both
journals. It does not emit samples or full-run summaries.

In both shapes `remote_charge` contains exactly
`earlier_governed_use_usd`, `predecessor_partial_usd`,
`continuation_journal_usd`, and `cumulative_authorized_use_usd`, all canonical
Decimal strings. The cumulative value is exactly the sum of the other three,
so it includes the earlier governed use of USD `0.051104882` rather than only
the two TypeSafe comparison sessions. The artifact manifest schema is
`phase4c3c-artifact-manifest.v1` and contains exactly
`schema_version`, `claim_sha256`, `files`, `predecessors`, `sources`, and
`self_sha256`; its three committed files are `result.json`, `report.md`, and
`cost-journal.jsonl`.

Permitted `failure_category` values are exactly `http_error`, `rate_limited`,
`provider_unavailable`, `timeout`, `transport_error`, `response_invalid`,
`model_drift`, `usage_overflow`, `budget_uncertain`, and
`process_interrupted`. `process_interrupted` is never written as a journal row.
Credential-free reconstruction derives it for every non-complete journal with
no explicit terminal failure, over-limit settlement, or halt: an empty journal
after claim creation, a journal ending in a settlement before the next reserve,
or a journal with an unmatched reservation.
`response_invalid_detail` is non-null exactly when the failure category is
`response_invalid` and must be one of the bounded diagnostics above.

Artifact redaction uses the closed schema, not an ambiguous substring ban over
fixed enum names. Every string value in result and journal must be one of: a
declared enum, schema version, exact scenario ID, exact model ID, canonical
decimal string, or lowercase 64-character digest. Report bytes are reproduced
from validated reduced fields only. Tests inject sentinel provider strings and
answer structures and prove they cannot survive reduction or publication.

The manifest has its own closed string contract: fixed schema and file names,
fixed repository-relative source keys, the exact Git object ID, and lowercase
64-character digests. It never accepts a local absolute path or a provider-
derived string.

The report must distinguish reused predecessor measurements from new
continuation measurements and retain every Phase 4C.3 interpretation limit.
It must never call the run a retry of successful requests. It states that
ordinal 10 returned HTTP 200 in the failed Phase 4C.3 session and is submitted
again as the first continuation request. A complete result therefore contains
57 samples from 58 provider executions. A partial report claims a second
provider execution only when ordinal 10 settled, exceeded usage, or produced an
HTTP 200 response-contract/model failure; otherwise it states only that the
continuation attempted dispatch and provider receipt is unknown. The
ordinal-10 latency may be affected by provider caching, and the two network
sessions are not a single uninterrupted timing run. It shows both composite
and continuation-only summaries and names the strict versus approximate
response contracts for their respective ordinal ranges.

## Cumulative budget and new single-use claim

The validated budget lineage before Phase 4C.3 is recomputed from the seven
governed inputs exactly as before. The continuation then adds the conservative
charge reconstructed from the sealed Phase 4C.3 journal.

For the approved partial artifact, the values are:

- earlier validated use: USD `0.051104882`;
- Phase 4C.3 partial conservative charge: USD `0.00350702`;
- use before continuation: USD `0.054611902`;
- maximum 47-request continuation envelope: USD `0.15040`;
- maximum cumulative use after reservation: USD `0.205011902`;
- minimum remaining headroom: USD `0.044988098`.

These figures are explanatory. Code derives them from validated inputs and
journals and fails if they do not match the governed state. It never trusts a
copied constant for prior use.

Before claim creation, the runner proves that validated prior use plus all 47
new reservations is at most USD 0.25. Before each dispatch it also checks the
dynamic formula
`validated_earlier_use + phase4c3_partial_charge + continuation_journal_charge + 0.0032 <= 0.25`.
If it fails before a reservation, the runner writes the terminal `halt` row and
does not dispatch. Settled cost is computed with `Decimal` from returned input
tokens. Usage above 64,000 is retained only as reduced usage and cost, journaled
as `settled_over_limit`, and stops execution.

The full-envelope proof makes `halt` defense in depth under the governed live
inputs. Tests inject a larger validated prior-use value to exercise it without
network access through a private `_run_prevalidated` seam; the public CLI and
preflight proof remain unchanged and cannot accept a budget override.

The USD 0.25 ceiling is a conservative pre-dispatch authorization enforced by
reservations. A provider-contract breach reporting more than 64,000 input
tokens always stops the run; above approximately 76,190 input tokens its
computed cost can also exceed the USD 0.0032 reservation and, in the extreme,
the cumulative ceiling. That reduced actual usage is recorded and execution
stops immediately. The report states this distinction and never describes the
ceiling as a provider billing guarantee.

The fresh claim lives under the shared Git common directory at
`saracura-local-claims/phase4c3c/<prior-ledger-sha256>.json`. The key is the
same immutable budget-lineage root used by the Phase 4C.3 entitlement, under a
separate namespace. Its closed value binds the recovery plan, actual
predecessor partial manifest, and the same seven input manifest digests. It is
atomically create-if-absent, mode `0600`, beneath `0700` directories. A
byte-identical existing claim is consumed, not resumable. Relocating inputs,
copying the partial artifact, or constructing a second structurally valid
partial with a different manifest cannot create another entitlement for that
budget lineage. The Phase 4C.3 claim is read-only evidence and is never reused
or deleted.

Because validation of the predecessor artifact also requires its original
Phase 4C.3 claim, continuation intentionally runs from the same Git clone and
Git common directory. Another clone fails closed even when given copied
artifacts. The operator runbook states this requirement.

## CLI and operator flow

The new module exposes:

```bash
uv run python -m benchmarks.universal_remote_resume validate-plan

uv run python -m benchmarks.universal_remote_resume run \
  --prior-cost-ledger <ledger> \
  --training-manifest <training-manifest> \
  --phase3c-artifact-dir <phase3c-dir> \
  --packet <packet> \
  --phase3d-artifact-dir <phase3d-dir> \
  --laya-mps-artifact-dir <laya-dir> \
  --mdeberta-mps-artifact-dir <mdeberta-dir> \
  --phase4c3-partial-artifact-dir <sealed-partial-dir> \
  --output-dir <new-directory> \
  --allow-network --budget-usd 0.25
```

`finalize-partial` and `validate-artifact` accept the same paths but no network,
budget, or credential flags. Finalization fills only byte-identical missing
publication files and never resumes dispatch.

The additive CLI may call the private Phase 4C.3 preflight only with a complete
namespace whose `output_dir` is the continuation destination, so the inherited
Git-ignore/outside-checkout safety check applies to the new output. It
independently repeats and returns the earlier-use calculation because Phase
4C.3 does not expose that local value.

The live operator command injects only `TYPESAFE_API_KEY` into the child through
the approved 1Password wrapper and reference. It creates no `.env` file and
does not print, copy, persist, or accept the secret in argv.

Immediately before the live command, the operator checks the TypeSafe account
usage UI without copying account identifiers or billing details into the
artifact. Any unexplained provider use outside the governed ledgers is a stop
condition. This external check distinguishes the locally computed ceiling from
actual provider billing; the artifact continues to label its amounts computed,
not billed.

Normal creation order matches Phase 4C.3: exclusively create and fsync the
`0700` output directory and empty `0600` journal, then atomically create the
single-use claim. A claim conflict removes only that invocation's still-empty
journal and output directory. After any reservation, no automatic cleanup is
allowed. Finalization publishes result, report, and then manifest with atomic
create-if-absent semantics.

## Files and ownership

Expected implementation scope:

- `benchmarks/manifests/phase4c3c-resume-plan.v1.json`;
- `benchmarks/universal_remote_resume/__init__.py`;
- `benchmarks/universal_remote_resume/__main__.py`;
- `benchmarks/universal_remote_resume/plan.py`;
- `benchmarks/universal_remote_resume/runner.py`;
- `tests/test_universal_remote_resume.py`;
- `docs/action/phase4c3c-operator.md`; and
- `benchmarks/validate_manifests.py` for the new schema route;
- `.github/workflows/ci.yml` only to give `universal-local` a full-history
  checkout; and
- manifest/frontmatter indexes only where repository validation requires them.

Forbidden modifications include the Phase 4C.3 package, plan, spec, runbook,
the Phase 4C.2 materializer and registry sources and manifests named above,
existing claims, and existing artifacts.

## Acceptance criteria

1. The recovery plan validates as a closed, digest-bound revision and the
   existing Phase 4C.3 plan and sources are byte-unchanged.
2. Offline preflight validates all seven governed inputs and the source-bound
   partial artifact before creating output, claim, or making a network call.
3. The runner can dispatch only ordinals 10 through 56, once each and in order;
   it cannot resend ordinals 0 through 9.
4. Fake-transport tests prove the exact 20- and 77-option wire requests remain
   unchanged by comparing all 57 materialized requests with frozen per-ordinal
   digests, and response bodies never reach artifacts.
5. Response-contract tests exercise every bounded diagnostic category, accept
   a sum error above `1e-6` and at most `0.01` with the boolean marker, and reject
   an error above `0.01`. Exact decimal sum `0.99` is accepted and `0.9899` is
   rejected without binary-float or Decimal-context ambiguity, including inputs
   with more than 28 significant digits.
6. Tests prove selected-choice maximality, key equality, numeric finiteness,
   model identity, and usage limits remain fail-closed.
7. Claim tests prove exclusivity across concurrent worktrees and relocated
   byte-identical inputs without touching the Phase 4C.3 claim. A second valid
   partial with a mutated, re-chained journal and different manifest still
   collides on the same budget-lineage claim before network use.
8. Budget tests derive the predecessor charge, reserve exactly 47 possible
   requests, reject any cumulative total above USD 0.25, and charge open
   reservations conservatively.
9. Composite reconstruction accepts exactly the ten predecessor settlements
   followed by 47 continuation settlements and rejects gaps, overlap, mutation,
   or a different partial artifact.
10. Crash-boundary tests cover output, claim, reserve, settlement, result,
    report, and manifest publication. Credential-free finalization and
    validation perform no network access. Empty, post-settlement, and
    unmatched-reservation journals all reconstruct as `process_interrupted`.
11. The complete fake run reports 57 requests and 183 decisions while clearly
    identifying ten reused predecessor measurements and 47 new measurements.
12. `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`,
    `uv run pytest -q`, repository manifest/frontmatter gates, default-environment
    checks, and `uv build` all pass.
13. Wheel inspection proves the new benchmark package remains checkout-only and
    the default installation imports no provider SDK.
14. The independent reviewer verifies no response data, prompt data, secret,
    local path, or private artifact was added to Git.
15. Tests prove the diagnostic precedence, one-MiB response ceiling, custom
    artifact redaction contract, and `halt` transition. Fixed enum names such as
    `confidence_invalid` remain allowed while injected provider sentinel text,
    answer payloads, and unrecognized strings are rejected.
16. Tests prove valid non-canonical JSON key order, whitespace, and number
    spellings are accepted; malformed, duplicate-key, non-finite, overlarge
    integer, non-hashable, and injected internal-validator failures reduce to
    their exact bounded categories without exception or body leakage.
17. Prior-use derivation independently repeats the Phase 3C and Phase 3D
    validators and is tested for exact Decimal equivalence with the Phase 4C.3
    preflight formula; it does not trust or import an unreturned local variable.
18. Manifest routing validates the new plan in both full-history CI jobs; the
    `universal-local` checkout uses `fetch-depth: 0`, and a missing pinned Git
    object fails closed. Usage above `2^53-1` is rejected as `usage_invalid`
    before settlement.
19. Decimal digit/exponent limits reject compact computational bombs before
    Fraction conversion while accepting legitimate high-precision boundary
    fixtures.

## Rollout, stop conditions, and rollback

Implementation and offline review merge before any continuation run. There is
no deploy or migration.

The live continuation stops immediately on any failure, model drift, usage
overflow, budget uncertainty, claim conflict, artifact mismatch, or structural
response category. If ordinal 10 fails outside the new normalization tolerance,
the bounded detail is the only new diagnostic; another network attempt requires
a new reviewed spec and authorization.

Rollback is a Git revert of the additive Phase 4C.3c tooling. Claims and sealed
artifacts remain immutable evidence and are not removed by rollback.

## Next gate

If the continuation artifact is complete, or is a sealed terminal partial that is explicitly declared `insufficient_evidence`, Phase 4C.4 may produce the [architecture decision record](../../decisions/0001-two-tier-decision-architecture.md). For a terminal partial, the operator must have stopped further recovery and the failure and completed coverage must remain preserved; missing cells stay `insufficient_evidence`, the artifact is not relabeled complete, and it cannot support a candidate-quality recommendation or another provider call. Because the benchmark is synthetic and unlabeled, production selection remains `insufficient_evidence`; the ADR may select the next labeled PT-BR experiment based on systems fit, capacity, cost, and operational complexity.
