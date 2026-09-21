---
title: Phase 2C data, license, and LGPD gate
kind: spec
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: active
canonical: docs/action/specs/phase2c-data-governance-gate.md
globalRef: qmd://saracura/docs/action/specs/phase2c-data-governance-gate.md
reviewCadenceDays: 14
lastReviewedAt: 2026-09-21
sourceRefs:
  - https://www.gov.br/mj/pt-br/assuntos/sua-protecao/sedigi/Lei13709.pdf
  - https://www.gov.br/anpd/pt-br/centrais-de-conteudo/documentos-tecnicos-orientativos/estudo_tecnico_sobre_anonimizacao_de_dados_na_lgpd___analise_juridica.pdf
  - https://www.amazon.science/publications/massive-a-1m-example-multilingual-natural-language-understanding-dataset-with-51-typologically-diverse-languages
  - https://huggingface.co/datasets/AmazonScience/massive/tree/cf8448dca459d700b8a250b270e57f6d1444bf3e/pt-PT
  - https://huggingface.co/datasets/Magurofg/massive-pt-br/tree/907f905b4b237c34ba805f942bfbb6d92ec9c81f
related:
  - docs/action/specs/phase2b-encoder-candidate-gate.md
  - SECURITY.md
  - CONTRIBUTING.md
supersedes: []
supersededBy: []
sensitivity: public
---

# Phase 2C data, license, and LGPD gate

## Context

Phase 2B proved that two multilingual encoders can be acquired and measured
locally without weakening the supply-chain boundary. It did not authorize data
ingestion or training. Phase 2C makes that boundary executable before a small
`choice` head is trained.

The first narrow workflow is `support-routing.v1`, with five mutually exclusive
choices: `billing`, `technical_support`, `account_access`,
`subscription_cancellation`, and `order_delivery`. The beachhead locale is
`pt-BR`; English is a measured secondary locale, not a substitute for PT-BR
evidence.

This document records an engineering policy, not legal advice. A source being
public or carrying a dataset-card license does not by itself prove that every
record is lawful, non-personal, accurate, or suitable for model training.

## Outcome

The repository gains a versioned, fail-closed source-policy registry and an
offline validator. It records which source categories may be considered for a
future artifact and which controls would be required. It does not approve any
dataset bytes. Concrete artifact approval is a separate Phase 3 manifest that
must bind exact bytes, completed controls, reviewers, and the source-policy
registry revision. No dataset is downloaded, generated, ingested, approved, or
published in this phase.

## Legal and privacy baseline

The LGPD defines personal data by identifiability and defines anonymization by
the loss of direct or indirect association using reasonable available means.
Article 12 keeps data within scope when anonymization can be reversed using the
controller's own means or reasonable efforts. Therefore:

- `public` is not a privacy classification;
- deletion of obvious names is not proof of anonymization;
- pseudonymized customer or employee text remains personal data;
- synthetic text is treated as potentially personal if prompted from, copied
  from, or linkable to a natural person;
- source text with personal or sensitive data is blocked from the public lane;
- any future private-data program requires its own purpose, legal basis,
  controller/operator mapping, minimization, retention, access, deletion,
  incident, and impact-assessment review outside this repository.

The public Saracura lane accepts only defined artifact records whose completed
review concludes `no_personal_data`; a source category cannot claim this before
records exist. Apache-2.0 or CC BY 4.0 does not prove privacy rights or the chain
of rights for every contributed record.
Automated scanning is a gate, not proof: ambiguous findings require human
review, and a failed or incomplete review blocks promotion.

## Source matrix

| Source policy | Locale | Rights/provenance | Planned uses | Policy state | Artifact approved now | Reason and required controls |
| --- | --- | --- | --- | --- | --- | --- |
| Saracura human-original cases | pt-BR, en | future original human-authored text; author identity and contribution grant required | train, dev, calibration, blind test | conditionally_allowed | no | no real people, companies, accounts, transactions, addresses, contact data, secrets, copied tickets, or model-authored text; author and reviewer must be distinct humans; two-human review for blind test |
| Saracura deterministic derivatives | pt-BR, en | future derivation only from approved human-original atoms; generator revision and seed required | train only | conditionally_allowed | no | grouped by global scenario family; exact/near duplicate detection; no template, atom, translation pair, or family crosses splits; never used for dev/calibration/test evidence |
| Provider/model-generated or model-assisted cases | unspecified | provider, model, terms, prompt, input, and output lineage not selected | none | quarantined | no | human review does not change model-generated origin; requires a later policy revision and cannot enter the human-original lane |
| Amazon MASSIVE official `pt-PT` | pt-PT | Amazon Science, CC BY 4.0, upstream revision `cf8448dca459d700b8a250b270e57f6d1444bf3e` | separate external intent-classification control | conditionally_allowed | no | 60-class assistant-intent taxonomy is incompatible with the five support queues; preserve native task/splits, attribute, mark modifications, and never map it into PT-BR support evidence without a new reviewed spec |
| Community `massive-pt-br` localization | pt-BR | recent CC BY 4.0 derivative whose card reports LLM localization from MASSIVE `pt-PT` | none | quarantined | no | audit immutable files/digests, teacher identity and terms, prompts, rejected rows, split preservation, leakage, shortcuts, attribution, taxonomy mismatch, and Brazilian human review before any use |
| Felhen/AIOS operational traces | pt-BR, en | private operational and potentially confidential/personal data | none | blocked | no | must not enter the standalone OSS repository, public artifacts, teacher prompts, training, calibration, or evaluation |
| Customer support tickets or CRM exports | any | controller-specific personal/confidential data | none | blocked | no | de-identification alone is insufficient; a separate private governance program would be required and is unnecessary for the first public model |
| Web crawls, social posts, or datasets without record-level provenance and compatible rights | any | unknown or mixed | none | blocked | no | public availability is not permission; source-level labels cannot cure unknown record-level rights or privacy |

`conditionally_allowed` means only that a future concrete artifact may request
approval under a later spec. Every Phase 2C row has
`artifact_approved_for_use=false`; no row authorizes acquisition or use.

## Closed registry contract

Add `benchmarks/manifests/training-data-source-policies.v1.json`, validated by
strict Pydantic models. The registry is policy metadata only and contains no
examples or artifact approvals. Root fields are exactly `schema_version` and
`sources`; schema is `training-data-source-policies.v1` and `sources` has exactly
the eight matrix entries.

Each source object has the following exact fields:

| Field | Type and bound |
| --- | --- |
| `id` | regex `^[a-z][a-z0-9-]{2,63}$`, unique |
| `name` | string, 3–120 characters |
| `source_type` | `human_original`, `deterministic_derivative`, `model_generated`, `third_party_dataset`, `private_operational`, `customer_export`, or `unvetted_public` |
| `origin` | `first_party`, `third_party`, `private_felhen`, `customer_controller`, or `unknown` |
| `policy_state` | `conditionally_allowed`, `quarantined`, or `blocked` |
| `artifact_approved_for_use` | literal `false` in Phase 2C |
| `planned_uses` | unique subset of `train`, `dev`, `calibration`, `blind_test`, `external_control`; empty for quarantined/blocked |
| `locales` | 1–8 unique values from `pt-BR`, `en`, `pt-PT`, `any`, `unspecified`; `any`/`unspecified` must be alone |
| `license` | `Apache-2.0`, `CC-BY-4.0`, or `unknown`; first-party policy uses `Apache-2.0`, while `origin` separately carries authorship |
| `source_url` | HTTPS URL without userinfo/query/fragment for defined third-party sources; otherwise `null` |
| `license_url` | HTTPS URL without userinfo/query/fragment when license is known; otherwise `null` |
| `upstream_revision` | 40 lowercase hex characters for defined third-party sources; otherwise `null` |
| `record_identity_state` | `not_created`, `not_acquired`, or `unavailable` |
| `privacy_review_state` | literal `not_reviewed` in Phase 2C |
| `rights_review_state` | `policy_only` or `unknown` |
| `authoring_mode` | `human_original`, `deterministic_derivative`, `model_generated_or_assisted`, or `not_applicable` |
| `teacher_lineage_state` | `not_used`, `required_missing`, or `not_applicable` |
| `redistribution_state` | `policy_allowed`, `requires_artifact_review`, or `prohibited` |
| `attribution` | string 0–500 characters; non-empty for CC BY 4.0 |
| `required_controls` | 1–24 unique closed control IDs |
| `reviewed_at` | ISO date |
| `reviewer_role` | literal `saracura-maintainers` |
| `rationale` | string, 20–1,000 characters |

The closed control IDs are `artifact_manifest`, `byte_hashes`,
`record_provenance`, `human_author_identity`, `independent_human_review`,
`provider_terms`, `teacher_lineage`, `privacy_record_review`,
`rights_record_review`, `attribution_notice`, `modification_notice`,
`native_split_preservation`, `taxonomy_separation`, `grouped_split`,
`cross_locale_family_link`, `duplicate_scan`, `contamination_scan`,
`shortcut_audit`, and `takedown_lineage`.

Semantic truth table:

- all entries have `artifact_approved_for_use=false`, non-empty controls, and no
  artifact/content digest because no bytes are approved;
- `quarantined` and `blocked` require empty `planned_uses`;
- `blocked` requires `license=unknown` unless the block is privacy/confidentiality
  based, and always requires `redistribution_state=prohibited`;
- the two first-party policies alone may plan `train`; deterministic derivatives
  may plan only `train` and require `authoring_mode=deterministic_derivative`;
- model-generated/assisted sources require `teacher_lineage_state=required_missing`
  and remain quarantined;
- only the official MASSIVE policy may plan `external_control`; it requires
  `pt-PT`, CC BY 4.0, the pinned revision, attribution/modification controls, and
  `taxonomy_separation`;
- no policy may plan both `external_control` and a Saracura workflow split;
- `pt-PT` can never authorize PT-BR evidence; locale claims live in the later
  artifact manifest and must equal reviewed record bytes;
- private operational, customer, and unvetted-public policies are fixed blocked
  IDs with no uses and cannot be reinterpreted by editing fields.

The registry has exactly these ID-bound identities. `planned_uses` may be a
subset of the listed maximum but may never exceed it. `required_controls` must
be a superset of the per-ID minimum; controls may be added but none of the
minimums may be removed.

Locale list, source URL, license URL, upstream revision, record-identity state,
rights-review state, teacher-lineage state, and redistribution state are also
fixed per ID to the initial reviewed manifest. In particular, official MASSIVE
is fixed to revision `cf8448dca459d700b8a250b270e57f6d1444bf3e`, and the
community localization is fixed to revision
`907f905b4b237c34ba805f942bfbb6d92ec9c81f`. A different URL, branch-like path,
revision, locale, or state requires a new reviewed policy revision; a syntactically
valid SHA-1 is not interchangeable with the reviewed revision.

| ID | Fixed identity | State and maximum uses | Minimum required controls |
| --- | --- | --- | --- |
| `saracura-human-original` | `human_original`, `first_party`, `human_original`, Apache-2.0 | `conditionally_allowed`; train, dev, calibration, blind_test | `artifact_manifest`, `byte_hashes`, `record_provenance`, `human_author_identity`, `independent_human_review`, `privacy_record_review`, `rights_record_review`, `grouped_split`, `cross_locale_family_link`, `duplicate_scan`, `contamination_scan`, `shortcut_audit`, `takedown_lineage` |
| `saracura-deterministic-derivatives` | `deterministic_derivative`, `first_party`, `deterministic_derivative`, Apache-2.0 | `conditionally_allowed`; train only | `artifact_manifest`, `byte_hashes`, `record_provenance`, `human_author_identity`, `privacy_record_review`, `rights_record_review`, `grouped_split`, `cross_locale_family_link`, `duplicate_scan`, `contamination_scan`, `shortcut_audit`, `takedown_lineage` |
| `provider-model-generated` | `model_generated`, `unknown`, `model_generated_or_assisted`, unknown license | `quarantined`; no uses | `artifact_manifest`, `byte_hashes`, `record_provenance`, `provider_terms`, `teacher_lineage`, `privacy_record_review`, `rights_record_review`, `grouped_split`, `cross_locale_family_link`, `duplicate_scan`, `contamination_scan`, `shortcut_audit`, `takedown_lineage` |
| `amazon-massive-ptpt` | `third_party_dataset`, `third_party`, `not_applicable`, CC-BY-4.0 | `conditionally_allowed`; external_control only | `artifact_manifest`, `byte_hashes`, `record_provenance`, `privacy_record_review`, `rights_record_review`, `attribution_notice`, `modification_notice`, `native_split_preservation`, `taxonomy_separation`, `duplicate_scan`, `contamination_scan`, `shortcut_audit`, `takedown_lineage` |
| `community-massive-ptbr` | `third_party_dataset`, `third_party`, `model_generated_or_assisted`, CC-BY-4.0 | `quarantined`; no uses | `artifact_manifest`, `byte_hashes`, `record_provenance`, `provider_terms`, `teacher_lineage`, `independent_human_review`, `privacy_record_review`, `rights_record_review`, `attribution_notice`, `modification_notice`, `native_split_preservation`, `taxonomy_separation`, `cross_locale_family_link`, `duplicate_scan`, `contamination_scan`, `shortcut_audit`, `takedown_lineage` |
| `felhen-aios-operational` | `private_operational`, `private_felhen`, `not_applicable`, unknown license | `blocked`; no uses | `privacy_record_review`, `rights_record_review`, `takedown_lineage` |
| `customer-support-exports` | `customer_export`, `customer_controller`, `not_applicable`, unknown license | `blocked`; no uses | `privacy_record_review`, `rights_record_review`, `takedown_lineage` |
| `unvetted-public-text` | `unvetted_public`, `unknown`, `not_applicable`, unknown license | `blocked`; no uses | `record_provenance`, `privacy_record_review`, `rights_record_review` |

The validator tests every fixed identity and removes each minimum control in
turn. It must reject a renamed/retyped fixed source, broadened uses, a missing
minimum control, or a model-generated/assisted source presented as
`human_original`.

Reject unknown fields, duplicate JSON keys/IDs, every violation of this table,
credential-bearing or mutable URLs, and mutable third-party revisions. Parse the
bundled file directly with no generic dataset loader. The only command is
`python -m benchmarks.data_policy_gate validate-registry`; it accepts no URL,
path, source, revision, override, or network flag. Public errors are bounded
codes and never echo URLs or local paths. Validation performs no network call.

The registry records upstream declarations and internal policy disposition; it
does not replace legal review. A future `training-data-artifact.v1` contract,
not this registry, will carry exact byte digests, record identities, completed
privacy/rights evidence, reviewer identities, and the exact policy-registry
digest that authorized consideration.

## First-party dataset design for the next phase

Phase 3 may create a separate `support-routing.v1` dataset only after this gate
is merged. The task is assignment to the single primary queue that should own
the next action, not identification of every topic mentioned. The annotation
guide must freeze priority and tie-break rules before examples are authored. It
must include `ambiguous` and `out_of_scope` review outcomes; those records are
excluded from the five-class head and reported, never forced into a convenient
label. Its contract must require:

- immutable record ID, locale, text, primary-queue label, global
  scenario-family ID shared across translations, authoring mode, source atom
  IDs, generator revision/seed when applicable, author identity, reviewer
  identity, reviewer status, and content digest;
- no customer, employee, partner, or production-derived text;
- fictional values only, with denylisted identifiers and pattern scanning;
- labels from the closed five-choice taxonomy;
- closed authoring modes `human_original`, `deterministic_derivative`, and
  `model_generated_or_assisted`; Codex or any other agent output is always
  model-generated/assisted, even after human review;
- distinct human author and reviewer roles; two agents do not satisfy
  two-person review, and absent human evidence keeps the packet experimental;
- NFC normalization and duplicate detection before split assignment;
- grouped splitting by scenario family before generation, so paraphrases and
  deterministic variants or cross-locale translations never cross
  train/dev/calibration/test boundaries;
- a human-authored blind test set that shares no templates or source atoms with
  training and is never sent to a teacher model;
- independent-state counts as the statistical unit; variants and multiple
  questions from one state do not inflate sample size;
- two independent annotations for dev/calibration/blind-test, followed by
  adjudication under the frozen primary-queue rules;
- append-only lineage and a dataset card stating limitations, removals, and
  known coverage gaps.

The initial target is a research packet, not a production corpus:

| Split | Planning target: independent PT-BR states | Planning target per label | Allowed authoring |
| --- | ---: | ---: | --- |
| train | 1,000 | 200 | human-original plus deterministic derivatives from approved atoms |
| dev | 500 | 100 | independently human-original; no train templates, atoms, or families |
| calibration | 500 | 100 | independently human-original; untouched until model/protocol freeze |
| blind test | 500 | 100 | independently human-original and two-human reviewed; one final access |

English remains a separate exploratory slice and may not be pooled into PT-BR
denominators. A smaller dataset may exercise plumbing but cannot support quality,
calibration, or automation claims. The numbers above are planning targets, not
statistical authorization. Phase 3 must freeze a primary metric, practical
margin, grouped interval method, rare-event treatment, target distribution, and
power/sample-size analysis before a headline claim.

Encoder, head, hyperparameters, thresholds, and error-driven revisions are
selected only on train/dev. Then the candidate, baselines, serializer, protocol,
and metric code are frozen. Calibration is fitted once on the calibration split.
The blind test is evaluated once; its errors cannot drive model selection,
example rewriting, or another reported test run. Every holdout access is logged.

## Contamination, leakage, and quality gates

Before a dataset can move from metadata to use:

1. pin source bytes and SHA-256 before parsing;
2. preserve the upstream split and record IDs for external sources;
3. canonicalize only through a versioned transform and retain raw-byte digests;
4. scan exact duplicates, normalized duplicates, and near duplicates across
   splits, reporting thresholds and counts rather than silently deleting;
5. group semantically related states and cross-locale translations by one global
   family before splitting;
6. scan for personal data, secrets, URLs, handles, email, phone, taxpayer,
   payment, account, address, and production identifiers;
7. review class balance, lexical shortcuts, label ambiguity, negation,
   misspellings, regional wording, and code-switching;
8. freeze label guidelines and adjudicate disagreements without allowing the
   model under test to label its own evaluation set;
9. make takedown traceable from source/record to every derived artifact and
   require retraining when removal changes training bytes;
10. compare against Saracura fixtures/probes and pertinent known public corpora,
    recording thresholds, source revisions, overlaps, and uncertainty;
11. state that unknown overlap with encoder pretraining cannot be excluded;
    “no overlap found” must never be reported as “uncontaminated”;
12. record every exclusion and transformation in an append-only audit report.

No PII detector score, anonymization function, teacher assertion, or dataset-card
license automatically promotes a source.

## Implementation scope

Phase 2C delivers only:

1. this reviewed spec;
2. the closed source-policy registry with all eight matrix rows and no artifact
   approval;
3. an offline registry validator routed by `validate_manifests.py`;
4. tests for schema closure, duplicate keys/IDs, exact enums/bounds, URL
   credentials, immutable revisions, the semantic truth table, privacy/rights
   fail-closed rules, locale bounds, attribution, and network-free validation;
5. English and PT-BR documentation of the gate and the next approved command;
6. CI execution in the lightweight default environment.

Out of scope: dataset download, example authoring, LLM generation, external API
calls, PII processing, training, calibration, model selection, weight or dataset
publication, AIOS adapters, paid compute, and quality claims.

## Acceptance criteria

1. The registry represents every row in the source matrix, validates offline,
   and approves no artifact bytes.
2. Only future first-party human-original data and deterministic derivatives may
   be considered for PT-BR training; no source or artifact is silently approved.
3. MASSIVE `pt-PT` is constrained to an external control role and cannot satisfy
   PT-BR evidence.
4. Private operational and customer data fail closed for every use.
5. Unknown privacy, rights, license, provenance, revision, or teacher lineage
   cannot be promoted.
6. Tests prove contradictory or broadened policies fail validation, including
   deterministic derivatives promoted to dev/calibration/test, `pt-PT` promoted
   to PT-BR evidence, a model-assisted case marked human-original, and private
   data assigned any use.
7. Default installation imports no dataset loader. Registry validation and tests
   perform no network; dependency installation is outside that assertion.
8. Documentation states that the matrix is an engineering gate, not legal
   advice or proof of dataset quality.
9. Ruff, format, mypy, pytest, manifest validation, default-environment check,
   build, package inspection, secret scan, and `git diff --check` pass.

## Rollback

Phase 2C contains metadata and policy only. Rollback is a Git revert. Because no
records are acquired and no model is trained, there is no dataset cache, derived
weight, or remote artifact to delete.

## Next gate

After Phase 2C, a separate Phase 3 spec may author, review, and freeze a concrete
first-party `support-routing.v1` research packet. Its artifact manifest must bind
exact bytes and record IDs to the exact Phase 2C registry digest and prove every
completed control before any training process can read it. Phase 3 then trains
the shared-encoder small `choice` head using grouped train/dev selection,
one-time calibration, and one-time blind testing. External sources remain
separate controls until their own acquisition and artifact-review specs pass;
MASSIVE is optional inventory and is not required to complete Phase 2C or 3.
