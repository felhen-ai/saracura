---
title: Gemini refusal-aware internal V3 diagnostic
kind: spec
area: research
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/action/specs/gemini-refusal-diagnostic.md
globalRef: qmd://saracura/docs/action/specs/gemini-refusal-diagnostic.md
reviewCadenceDays: 30
lastReviewedAt: 2026-11-12
sourceRefs:
  - https://aistudio.google.com/docs/decisions-api
related:
  - research/evaluation-guide.md
  - research/v3_spec.md
  - docs/action/specs/gemini-decision-v3.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Refusal-aware diagnostic on V3 inputs

The user authorizes an internal evaluation that counts vendor content refusals as incorrect decisions and continues. This is an explicit diagnostic policy, not a relaxation of official submission validators. Exact V3 inputs, option order and scoring weights remain fixed. Existing fail-closed bridge behavior remains unchanged; the new explicit runner distinguishes content refusal from operational failure directly.

## C1

Add research/gemini_diagnostic.py, a stdlib opt-in direct API diagnostic runner using existing gemini_bridge parse_request/translate/https_transport/remap_response. Existing bridge default latch, official evaluators, validators, data and runtime unchanged. API key only from GEMINI_API_KEY env. No retries, remote fallback, private data or downloads of weights.

## C2

Create frozen 5196-item V3 plan from existing benchmark_v3/camara_v3 loaders: 599 Camara32 full inputs and 893 full legal inputs, five unchanged tasks preserving exact original ptbr_bench.question RNG replay across ALL seven legacy tasks. Camara own seed42. Bind dataset revision/digests and exact provider payload hash per item. Text/instructions and shuffled choice order unchanged; no gold in API payload. Explicit text transform: five unchanged legacy tasks use text[:3500], legal uses entire text, Camara32 uses entire V3 text. Source V2 wire bytes are json.dumps({model,state:text[:3500],questions:{q:q}}).encode() with default ensure_ascii=True and spaces exactly as ptbr_bench.call. Provider bytes are existing Bridge serialization ensure_ascii=False,separators=(comma,colon),allow_nan=False UTF-8; add golden equality test.

## C3

Only HTTP400 with parsed JSON error.code in closed documented set safety/prohibited_content/blocklist/image_safety/content_blocked is model refusal. Journal outcome refusal, prediction null and safe reason/code/digests, count as incorrect and continue next item without retry. Any 401/403/429/5xx/unknown400, transport, malformed success/usage, model identity drift or journal failure is operational error, append sanitized evidence then stop; not a model-quality error. Never emit raw provider errors or secret/request text.

## C4

Persist append-only per-item journal (0600), exclusive manifest/report, fsync writes, no overwrites. Record run identity, source hashes, original benchmark/request and provider payload digests, input chars, question order, prediction or refusal, confidence/probabilities sanitized position arrays, UTC time, upstream latency excluding quota sleep, numeric usage or unknown, source provenance. Resume validates manifest/plan and every prior row unique ID, ordered prefix, exact request hashes and prediction type/key before skipping resolved answered/refusal rows. Operational failure remains separate attempt evidence; require explicit new/resume attempt for rerun and do not silently overwrite it. Mutable API alias unknown revision and multi-attempt reuse disclosed; no official submission eligibility.

## C5

CLI import-legacy validates frozen original full-mode legacy report (partial failures permitted), dataset digests/revision/gold, and each successful prediction against original full-run bridge journal within report start/finish UTC range, original wire-body hash, provider payload hash and sanitized decoded answer positions/noul. Use the first chronological full-run matching evidence, never best/latest or smoke. Import ONLY answered items whose exact provider payload matches current V3 plan; ignore previous refusal and latched-null rows. The formerly refused FACTCK item is attempted once explicitly under this new diagnostic policy and classified by the new parser; this removes the separate old refusal-evidence importer. Write immutable seed with source hashes and records digest, check records against plan and declared seed digest during run/resume; reject mutation/duplicate/mismatched payload or invalid prediction. Declare old Camara24 answers incompatible with V3 and exclude them from V3 scoring. No fabricated official report.

## C6

Report each task coverage answered/N, refusal counts/reasons, correct counts and balanced accuracy with ALL rows in denominator (refusals wrong). For Camara implement same official balanced-valid-topic weights/chance over immutable accepted labels, treating null/refusal as no hit; verify no-refusal equality with camara_v3.score_predictions on synthetic fixtures. Seven-task diagnostic normalized aggregate uses same V3 chance correction/clipping/equal task weighting. Headline refusal-inclusive score plus answered-only accuracy clearly conditional, unranked internal diagnostic; incomplete/operationally failed runs get no full aggregate. Report planned/attempted/reused/answered/refused/operational counts distinctly. Never claim sample of 715 represents seven tasks. Match official public return scales: Camara macro_valid_topic and chance are 0-100 points; normalized =100*min(1,max(0,(macro_fraction-chance_fraction)/(1-chance_fraction))). Legacy adjusted =100*(BA-1/K)/(1-1/K), normalized100*max(0,signed). Average seven normalized POINT values. Test complete seven-task no-refusal equivalence against existing formulas.

## C7

Quota-aware sequential pacing default free-tier 200RPM/50000TPM, use min interval >=0.4 and rolling 61s token window conservative reserve len(provider payload bytes)+2048 <=49000 then replace with provider returned total_input_tokens. Unknown usage retains conservative reservation. No network if request reserve exceeds budget; unknown consumption/error stays in journal; no auto retry. Totals include imported/new counters separately, per-request latency measured only during HTTP. Injectable fake transport/clock/sleep for offline tests. Cost unknown unless confirmed tier; no false zero-token usage. Mandatory offline plan preflight computes all payload reservations before network; any oversized item produces an explicit unrunnable_quota plan error and zero API calls, never drops/truncates it. Current frozen plan verified all5196 fit: largest reservation48421 (full legal), limit49000. Synthetic oversized test required.

## C8

Synthetic offline tests meaningfully cover content refusal continues/wrong denominator, quota/auth/transport/invalid responses fatal, exact order and full input, Camara scorer equivalence+refusal penalty, seed compatibility/incompatibility/mutation and refusal binding, resume duplicates/order/hash rejection, no repeated completed calls, pacing and sanitized logs/no credentials. Ruff, format, mypy src tests benchmarks and focused pytest pass. Document commands plan/import/run/resume/report as needed, keep public docs free of private paths/item IDs/real outputs. Orchestrator owns actual credentials, live run, final QA and integration. Content parser tests: numeric/uppercase/missing error.code and oversized body are operational failures, not refusals. Error.message never persisted; extra fields ignored rather than trusted/copied.
