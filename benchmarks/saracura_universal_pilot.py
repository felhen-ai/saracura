"""Isolated Phase 4E.2b acceptance-protocol pilot.

The corpus acceptance default stays in ``saracura_universal_corpus``. This module
owns the pilot plan, the pilot acceptance parameter, and the durable spend root.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from benchmarks import saracura_universal_corpus as corpus
from benchmarks.data_policy_registry import (
    Phase4EUniversalSyntheticExceptionV2,
    bundled_registry_v2_path,
    load_bundled_registry_v2,
)
from benchmarks.io import atomic_create
from benchmarks.saracura_universal_policy import AUTHOR_MODEL, REVIEWER_MODEL

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks/manifests/phase4e-protocol-pilot-policy.v1.json"
RECOVERY_POLICY_V2_PATH = ROOT / "benchmarks/manifests/phase4e-protocol-pilot-recovery.v2.json"
RECOVERY_POLICY_V3_PATH = ROOT / "benchmarks/manifests/phase4e-protocol-pilot-recovery.v3.json"
RECOVERY_POLICY_V4_PATH = ROOT / "benchmarks/manifests/phase4e-protocol-pilot-recovery.v4.json"
RECOVERY_POLICY_V5_PATH = ROOT / "benchmarks/manifests/phase4e-protocol-pilot-recovery.v5.json"
RECOVERY_POLICY_V6_PATH = ROOT / "benchmarks/manifests/phase4e-protocol-pilot-recovery.v6.json"
RECOVERY_POLICY_PATH = ROOT / "benchmarks/manifests/phase4e-protocol-pilot-recovery.v7.json"
BASELINE_PATH = ROOT / "benchmarks/manifests/phase4e-spend-baseline.v1.json"
PILOT_SEED = "saracura-phase4e-protocol-pilot-v1"
PILOT_PLAN_SCHEMA = "phase4e-protocol-pilot-plan.v7"
PILOT_SPLIT = "protocol_pilot"
AUTHOR_STAGE = "pilot_author"
REVIEWER_STAGE = "pilot_reviewer"
PILOT_AUTHOR_MODEL = "openai/gpt-4.1-mini"
PILOT_REVIEWER_MODEL = "google/gemini-3.8-flash"
PILOT_PROVIDER_POLICY = {
    "allow_fallbacks": True,
    "require_parameters": True,
    "data_collection": "deny",
    "zdr": True,
}
AUTHOR_MAX_OUTPUT_TOKENS = 1024
REVIEWER_MAX_OUTPUT_TOKENS = 192
_BOUNDARY = "\U0001f9ea"
_BOUNDARY_STATE_CODEPOINTS = 180
_AUTHOR_LINEAGE = (
    "author_response_sha256",
    "author_reservation_id",
    "author_request_id",
)
_REVIEWER_VIEW_FIELDS = frozenset(
    {"task_id", "locale", "domain", "instruction", "state", "criteria"}
)
_BASELINE_KEYS = (
    "schema_version",
    "as_of",
    "directory_count",
    "entry_count",
    "settled_count",
    "open_reservation_count",
    "provider_reported_cost_usd",
    "conservative_debit_usd",
    "open_reservation_debit_usd",
    "cumulative_authorization_usd",
    "pilot_complete_plan_bound_usd",
)
_PERSISTED_DIAGNOSTIC_KEYS = (
    "reviewed",
    "scenario_disagreement",
    "criterion_role_disagreement",
    "local_privacy",
    "reviewer_privacy_flags",
    "author_response_failures",
    "reviewer_response_failures",
    "retry_recoveries",
)


def _canonical(value: Any) -> bytes:
    return corpus._canonical(value)


def _sha(value: bytes) -> str:
    return corpus._sha(value)


def _money(value: Decimal) -> str:
    return format(value, "f")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    except (OSError, TypeError, ValueError) as error:
        raise corpus.CorpusError("invalid pilot JSON") from error
    if not isinstance(payload, dict):
        raise corpus.CorpusError("invalid pilot JSON")
    return payload


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = child
    return value


def _pilot_reviewer_system() -> str:
    return (
        "Return a JSON object whose sole top-level key is reviews, one review with status, "
        "chosen criterion ID, reason codes, the quality flags natural_language, "
        "generic_or_invented, exclusive_options, and private_or_sensitive, plus independent "
        "semantic_equivalence_attestation. Accepted reviews require reason_codes=[]. "
        + corpus.SCENARIO_CODEBOOK
        + " "
        + corpus.CRITERION_ROLE_CODEBOOK
        + " Infer roles from rule, facts, and options; do not write role labels into criteria. "
        "Choose the single scenario that directly explains the decision, ignoring merely "
        "incidental wording. Attest one scenario, distinct ordered roles, and "
        "selected_role=matches_rule at the chosen position. Select one criterion or reject. "
        "Evaluate generic_or_invented and private_or_sensitive only from instruction, "
        "state.summary, and "
        "criteria[].description. Ignore task_id, criterion IDs, locale, domain, and JSON field "
        "names for both judgments; they are trusted routing metadata. Treat generic_or_invented "
        "as an observable content-safety flag, not a claim about hidden author intent. A name "
        "without "
        "an observable real-world anchor must be treated as generic or invented: do not search, "
        "speculate about name collisions, require external provenance, or require an explicit "
        "synthetic declaration. Set generic_or_invented=true when the evaluated content has no "
        "URL, account "
        "handle, credential, exact private address, official registration, known public-entity "
        "claim, private record, or claim tied to a real record. Set generic_or_invented=false "
        "only when the evaluated content visibly contains one of those real-world anchors, and "
        "then use reason code real_world_anchor. "
        "exclusive_options=true only if exactly one criterion is best. Check language, "
        "sufficiency, privacy, and sensitive patterns. You do not receive answer, author "
        "attestation, family, pair, gold position, sibling, or split metadata."
    )


def validate_spend_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    """Reject any drift from the committed cumulative research totals."""

    payload = _read_object(path)
    if tuple(payload) != _BASELINE_KEYS:
        raise corpus.CorpusError("spend baseline")
    expected = {
        "schema_version": "phase4e-spend-baseline.v1",
        "as_of": "2026-09-24",
        "directory_count": 46,
        "entry_count": 2117,
        "settled_count": 2099,
        "open_reservation_count": 18,
        "provider_reported_cost_usd": "0.81263107",
        "conservative_debit_usd": "5.12153043",
        "open_reservation_debit_usd": "0.01971612",
        "cumulative_authorization_usd": "17.00",
        "pilot_complete_plan_bound_usd": "1.50",
    }
    if payload != expected:
        raise corpus.CorpusError("spend baseline")
    Decimal(payload["provider_reported_cost_usd"])
    Decimal(payload["conservative_debit_usd"])
    Decimal(payload["open_reservation_debit_usd"])
    if Decimal(payload["open_reservation_debit_usd"]) > Decimal(payload["conservative_debit_usd"]):
        raise corpus.CorpusError("spend baseline")
    if Decimal(payload["conservative_debit_usd"]) + Decimal(
        payload["pilot_complete_plan_bound_usd"]
    ) > Decimal(payload["cumulative_authorization_usd"]):
        raise corpus.CorpusError("spend baseline")
    return payload


def validate_pilot_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Validate the closed pilot manifest and its v2 registry binding."""

    policy = _read_object(path)
    expected = {
        "schema_version",
        "id",
        "seed",
        "task_count",
        "pair_count",
        "split",
        "plan_schema_version",
        "source_policy_exception_id",
        "source_policy_registry",
        "provider",
        "stages",
        "total_cap_usd",
        "cumulative_authorization_usd",
        "automatic_retries",
        "thresholds",
        "authorizations",
        "domain_scenario_map",
    }
    if set(policy) != expected:
        raise corpus.CorpusError("pilot policy shape")
    if (
        policy["schema_version"] != "phase4e-protocol-pilot-policy.v1"
        or policy["id"] != "phase4e-protocol-pilot"
        or policy["seed"] != PILOT_SEED
        or policy["task_count"] != 140
        or policy["pair_count"] != 70
        or policy["split"] != PILOT_SPLIT
        or policy["plan_schema_version"] != "phase4e-protocol-pilot-plan.v1"
        or policy["source_policy_exception_id"] != "phase4e_saracura_universal_synthetic"
        or policy["total_cap_usd"] != "1.50"
        or policy["cumulative_authorization_usd"] != "17.00"
        or policy["automatic_retries"] != 0
    ):
        raise corpus.CorpusError("pilot policy identity")
    registry = policy["source_policy_registry"]
    if not isinstance(registry, dict) or set(registry) != {"schema_version", "path", "sha256"}:
        raise corpus.CorpusError("pilot policy registry")
    v2_path = bundled_registry_v2_path()
    digest = hashlib.sha256(v2_path.read_bytes()).hexdigest()
    if (
        registry["schema_version"] != "training-data-source-policies.v2"
        or registry["path"] != "benchmarks/manifests/training-data-source-policies.v2.json"
        or registry["sha256"] != digest
    ):
        raise corpus.CorpusError("pilot policy registry digest")
    loaded = load_bundled_registry_v2()
    pilot_exception = cast(Phase4EUniversalSyntheticExceptionV2, loaded.exceptions[1])
    if (
        pilot_exception.id != policy["source_policy_exception_id"]
        or pilot_exception.protocol_pilot.seed != PILOT_SEED
        or pilot_exception.protocol_pilot.task_count != 140
        or pilot_exception.protocol_pilot.split != PILOT_SPLIT
        or pilot_exception.protocol_pilot.training_authorized is not False
        or pilot_exception.protocol_pilot.evaluation_only is not True
    ):
        raise corpus.CorpusError("pilot policy registry binding")
    provider = policy["provider"]
    if provider != {
        "host": "openrouter.ai",
        "author_model": AUTHOR_MODEL,
        "reviewer_model": REVIEWER_MODEL,
        "author_provider": "DeepInfra",
        "reviewer_provider": "CoreWeave",
        "allow_fallbacks": False,
        "require_parameters": True,
    }:
        raise corpus.CorpusError("pilot provider policy")
    stages = policy["stages"]
    if not isinstance(stages, dict) or tuple(stages) != (AUTHOR_STAGE, REVIEWER_STAGE):
        raise corpus.CorpusError("pilot stage policy")
    if stages[AUTHOR_STAGE] != {
        "cap_usd": "0.50",
        "input_usd_per_million": "0.30",
        "output_usd_per_million": "2.50",
    } or stages[REVIEWER_STAGE] != {
        "cap_usd": "1.00",
        "input_usd_per_million": "0.71",
        "output_usd_per_million": "0.71",
    }:
        raise corpus.CorpusError("pilot stage policy")
    if policy["thresholds"] != {
        "minimum_accepted": 98,
        "minimum_accepted_per_locale": 45,
        "locale_members": 70,
        "minimum_accepted_per_option_count": 11,
        "option_count_members": 20,
        "maximum_reviewer_privacy_flags": 7,
        "maximum_local_privacy_violations": 0,
    }:
        raise corpus.CorpusError("pilot thresholds")
    if policy["authorizations"] != {
        "evaluation_only": True,
        "training_authorized": False,
        "calibration_authorized": False,
        "publication_authorized": False,
        "quality_claims_allowed": False,
        "runtime_authorized": False,
    }:
        raise corpus.CorpusError("pilot authorizations")
    _validate_domain_map(policy["domain_scenario_map"])
    return policy


def validate_pilot_recovery_policy(path: Path = RECOVERY_POLICY_PATH) -> dict[str, Any]:
    """Validate the small execution-only overlay without re-declaring v1 data policy."""

    payload = _read_object(path)
    expected = {
        "schema_version",
        "id",
        "recovery_of",
        "plan_schema_version",
        "cost_mode",
        "cost_report_interval_usd",
        "maximum_attempts_per_request",
        "review_protocol",
        "models",
        "provider_policy",
    }
    if set(payload) != expected or payload != {
        "schema_version": "phase4e-protocol-pilot-recovery.v7",
        "id": "phase4e-protocol-pilot-recovery",
        "recovery_of": "phase4e-protocol-pilot-policy.v1",
        "plan_schema_version": PILOT_PLAN_SCHEMA,
        "cost_mode": "report_only",
        "cost_report_interval_usd": "10.00",
        "maximum_attempts_per_request": 5,
        "review_protocol": "genericity-schema-metadata.v1",
        "models": {
            AUTHOR_STAGE: {
                "id": PILOT_AUTHOR_MODEL,
                "input_usd_per_million": "0.40",
                "output_usd_per_million": "1.60",
            },
            REVIEWER_STAGE: {
                "id": PILOT_REVIEWER_MODEL,
                "input_usd_per_million": "0.75",
                "output_usd_per_million": "3.75",
            },
        },
        "provider_policy": PILOT_PROVIDER_POLICY,
    }:
        raise corpus.CorpusError("pilot recovery policy")
    return payload


def _validate_domain_map(mapping: object) -> dict[str, list[str]]:
    if not isinstance(mapping, dict) or tuple(mapping) != corpus.DOMAINS:
        raise corpus.CorpusError("pilot domain map")
    validated: dict[str, list[str]] = {}
    for domain in corpus.DOMAINS:
        choices = mapping[domain]
        if (
            not isinstance(choices, list)
            or not choices
            or not all(isinstance(item, str) for item in choices)
            or len(set(choices)) != len(choices)
            or any(item not in corpus.SCENARIO_CODES for item in choices)
        ):
            raise corpus.CorpusError("pilot domain map")
        validated[domain] = list(choices)
    return validated


def _scenario_for(
    seed: str, pair_id: str, domain: str, mapping: Mapping[str, Sequence[str]]
) -> str:
    choices = mapping[domain]
    return choices[corpus._seeded(seed, "pilot-scenario", pair_id) % len(choices)]


def _roles_for(seed: str, pair_id: str, option_count: int) -> list[str]:
    distractors = sorted(
        corpus.CRITERION_ROLES[1:],
        key=lambda role: corpus._seeded(seed, "pilot-role", pair_id, role),
    )[: option_count - 1]
    return ["matches_rule", *distractors]


def _pilot_slots(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    seed = cast(str, policy["seed"])
    mapping = _validate_domain_map(policy["domain_scenario_map"])
    option_seen: Counter[int] = Counter()
    slots: list[dict[str, Any]] = []
    serial = 0
    for pair_index in range(70):
        option_count = corpus.OPTION_COUNTS[pair_index % len(corpus.OPTION_COUNTS)]
        domain = corpus.DOMAINS[pair_index % len(corpus.DOMAINS)]
        within = option_seen[option_count]
        option_seen[option_count] += 1
        gold = within % option_count
        pair_id = "family-" + _sha(f"{seed}\0pilot-pair\0{pair_index}".encode())
        axes = {
            key: values[corpus._seeded(seed, "pilot-axis", pair_id, key) % 2]
            for key, values in corpus.AXES.items()
        }
        scenario = _scenario_for(seed, pair_id, domain, mapping)
        roles = _roles_for(seed, pair_id, option_count)
        for locale in corpus.LOCALES:
            task_id = "task-" + _sha(f"{seed}\0pilot-task\0{serial}".encode())
            slots.append(
                {
                    "slot": serial,
                    "task_id": task_id,
                    "family_id": pair_id,
                    "pair_id": pair_id,
                    "split": PILOT_SPLIT,
                    "locale": locale,
                    "domain": domain,
                    "axes": axes,
                    "option_count": option_count,
                    "gold_position": gold,
                    "semantic_target": {"scenario": scenario, "criterion_roles": list(roles)},
                }
            )
            serial += 1
    if serial != 140 or any(count != 10 for count in option_seen.values()):
        raise corpus.CorpusError("pilot allocation")
    return slots


_PREFLIGHT_CACHE: dict[str, dict[str, Decimal]] = {}


def preflight_bounds(slots: Sequence[Mapping[str, Any]]) -> dict[str, Decimal]:
    """Exact canonical-body worst case for the complete 140-task plan."""

    key = _sha(
        _canonical(
            {
                "slots": [dict(slot) for slot in slots],
                "author_model": PILOT_AUTHOR_MODEL,
                "reviewer_model": PILOT_REVIEWER_MODEL,
                "provider_policy": PILOT_PROVIDER_POLICY,
                "reviewer_system": _pilot_reviewer_system(),
            }
        )
    )
    cached = _PREFLIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    author_total = Decimal()
    reviewer_total = Decimal()
    batches = _pair_batches(slots)
    for batch in batches:
        author_body = corpus.provider_request_bytes(
            stage=AUTHOR_STAGE,
            model=PILOT_AUTHOR_MODEL,
            messages=corpus.author_messages(batch),
            response_schema=corpus.author_schema(batch),
            max_output_tokens=AUTHOR_MAX_OUTPUT_TOKENS,
            author_stage=AUTHOR_STAGE,
            author_model=PILOT_AUTHOR_MODEL,
            provider_override=PILOT_PROVIDER_POLICY,
            author_reasoning_effort=None,
        )
        author_total += corpus.request_worst_case(
            AUTHOR_STAGE,
            author_body,
            AUTHOR_MAX_OUTPUT_TOKENS,
            prices=corpus.PILOT_LEDGER_POLICY.prices,
        )
        for slot in batch:
            reviewer_body = corpus.provider_request_bytes(
                stage=REVIEWER_STAGE,
                model=PILOT_REVIEWER_MODEL,
                messages=pilot_reviewer_messages(_boundary_row(slot)),
                response_schema=pilot_reviewer_schema(1),
                max_output_tokens=REVIEWER_MAX_OUTPUT_TOKENS,
                author_stage=AUTHOR_STAGE,
                author_model=PILOT_AUTHOR_MODEL,
                provider_override=PILOT_PROVIDER_POLICY,
                author_reasoning_effort=None,
            )
            reviewer_total += corpus.request_worst_case(
                REVIEWER_STAGE,
                reviewer_body,
                REVIEWER_MAX_OUTPUT_TOKENS,
                prices=corpus.PILOT_LEDGER_POLICY.prices,
            )
    totals = {AUTHOR_STAGE: author_total, REVIEWER_STAGE: reviewer_total}
    _PREFLIGHT_CACHE[key] = totals
    return totals


def _pair_batches(slots: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for slot in slots:
        pair_id = slot.get("pair_id")
        if not isinstance(pair_id, str):
            raise corpus.CorpusError("pilot pair")
        if pair_id not in grouped:
            order.append(pair_id)
        grouped[pair_id].append(slot)
    return [grouped[pair_id] for pair_id in order]


def _boundary_row(slot: Mapping[str, Any]) -> dict[str, Any]:
    option_count = cast(int, slot["option_count"])
    return {
        "task_id": slot["task_id"],
        "locale": slot["locale"],
        "domain": slot["domain"],
        "instruction": _BOUNDARY * 120,
        "state": {"summary": _BOUNDARY * _BOUNDARY_STATE_CODEPOINTS},
        "criteria": [
            {"id": f"criterion-{index}", "description": _BOUNDARY * 120}
            for index in range(option_count)
        ],
    }


def pilot_reviewer_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Answer-blind reviewer object. Family and pair identity are omitted."""

    criteria = row.get("criteria")
    if not isinstance(criteria, list):
        raise corpus.CorpusError("reviewer transport criteria")
    view = {
        "task_id": row.get("task_id"),
        "locale": row.get("locale"),
        "domain": row.get("domain"),
        "instruction": row.get("instruction"),
        "state": row.get("state"),
        "criteria": [
            {"id": criterion.get("id"), "description": criterion.get("description")}
            if isinstance(criterion, Mapping)
            else {}
            for criterion in criteria
        ],
    }
    _validate_pilot_reviewer_view(view)
    return view


def _validate_pilot_reviewer_view(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _REVIEWER_VIEW_FIELDS:
        raise corpus.CorpusError("reviewer transport fields")
    forbidden = {
        "family_id",
        "pair_id",
        "split",
        "gold_position",
        "selected_criterion_id",
        "semantic_target",
        "semantic_equivalence_attestation",
        "axes",
        "review",
    }
    if forbidden & set(value):
        raise corpus.CorpusError("reviewer transport fields")
    if any(
        not isinstance(value[field], str)
        for field in ("task_id", "locale", "domain", "instruction")
    ) or not isinstance(value["state"], dict):
        raise corpus.CorpusError("reviewer transport types")
    corpus._validate_reviewer_json(value["state"])
    criteria = value["criteria"]
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 8:
        raise corpus.CorpusError("reviewer transport criteria")
    for criterion in criteria:
        if (
            not isinstance(criterion, dict)
            or set(criterion) != {"id", "description"}
            or not isinstance(criterion["id"], str)
            or not isinstance(criterion["description"], str)
        ):
            raise corpus.CorpusError("reviewer transport criterion")


def pilot_reviewer_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    safe = pilot_reviewer_view(row)
    payload = _canonical({"tasks": [safe]}).decode("utf-8")
    return [
        {"role": "system", "content": _pilot_reviewer_system()},
        {"role": "user", "content": payload},
    ]


def pilot_reviewer_schema(batch_size: int) -> dict[str, Any]:
    """Use an observable provider field while preserving the internal review contract."""

    schema = corpus.reviewer_schema(batch_size)
    reviews = cast(dict[str, Any], cast(dict[str, Any], schema["properties"])["reviews"])
    item = cast(dict[str, Any], reviews["items"])
    properties = cast(dict[str, Any], item["properties"])
    properties.pop("fictional")
    properties["generic_or_invented"] = {
        "title": "Generic Or Invented",
        "description": (
            "True when evaluated content uses generic or invented entities and contains no "
            "observable real-world anchor; false only when a real-world anchor is visible."
        ),
        "type": "boolean",
    }
    reason_codes = cast(dict[str, Any], properties["reason_codes"])
    reason_items = cast(dict[str, Any], reason_codes["items"])
    reason_items["enum"] = [
        "real_world_anchor" if code == "fictional" else code
        for code in cast(list[str], reason_items["enum"])
    ]
    required = cast(list[str], item["required"])
    item["required"] = [
        "generic_or_invented" if field == "fictional" else field for field in required
    ]
    return schema


def bind_pilot_reviewer_record(record: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    """Map the pilot wire-only genericity field into the stable internal field."""

    if "fictional" in record or type(record.get("generic_or_invented")) is not bool:
        raise corpus.CorpusError("review record genericity schema")
    normalized = dict(record)
    normalized["fictional"] = normalized.pop("generic_or_invented")
    reason_codes = normalized.get("reason_codes")
    if not isinstance(reason_codes, list) or "fictional" in reason_codes:
        raise corpus.CorpusError("review record genericity schema")
    normalized["reason_codes"] = [
        "fictional" if code == "real_world_anchor" else code for code in reason_codes
    ]
    return corpus.bind_reviewer_record(normalized, task_id)


def pilot_acceptance(row: Mapping[str, Any], review: corpus.ReviewerRecord) -> str | None:
    """Pilot acceptance keeps answer and quality gates and ignores semantic mismatch."""

    return corpus.acceptance_reason(row, review, reject_semantic_disagreement=False)


def pilot_pair_resolution(
    rows: Sequence[Mapping[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Do not apply the corpus exact-attestation pair rule."""

    del rows
    return accepted, rejected


def apply_pair_author_target_rule(
    slots: Sequence[Mapping[str, Any]],
    usable: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """An author target mismatch rejects both members of the planned pair."""

    del slots
    if not any(row.get("reason") == "semantic_target_mismatch" for row in rejected):
        return usable, rejected
    extra: list[dict[str, Any]] = []
    for row in usable:
        extra.append(
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "reason": "paired_author_target_mismatch",
                **{key: row[key] for key in _AUTHOR_LINEAGE if key in row},
            }
        )
    return [], [*rejected, *extra]


def build_plan() -> dict[str, Any]:
    """Return the immutable 140-task recovery plan with report-only cost estimates."""

    policy = validate_pilot_policy()
    recovery = validate_pilot_recovery_policy()
    slots = _pilot_slots(policy)
    totals = preflight_bounds(slots)
    author = totals[AUTHOR_STAGE]
    reviewer = totals[REVIEWER_STAGE]
    registry_sha = hashlib.sha256(bundled_registry_v2_path().read_bytes()).hexdigest()
    return {
        "schema_version": PILOT_PLAN_SCHEMA,
        "seed": PILOT_SEED,
        "source_policy_registry_sha256": registry_sha,
        "policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        "recovery_policy_sha256": hashlib.sha256(RECOVERY_POLICY_PATH.read_bytes()).hexdigest(),
        "reviewer_system_sha256": _sha(_pilot_reviewer_system().encode("utf-8")),
        "models": {
            AUTHOR_STAGE: PILOT_AUTHOR_MODEL,
            REVIEWER_STAGE: PILOT_REVIEWER_MODEL,
        },
        "provider_policy_sha256": _sha(_canonical(PILOT_PROVIDER_POLICY)),
        "domain_scenario_map_sha256": _sha(_canonical(policy["domain_scenario_map"])),
        "slots": slots,
        "cost_estimate": {
            "pilot_author_usd": _money(author),
            "pilot_reviewer_usd": _money(reviewer),
            "total_usd": _money(author + reviewer),
            "mode": recovery["cost_mode"],
            "report_interval_usd": recovery["cost_report_interval_usd"],
        },
    }


def validate_pilot_plan(value: Mapping[str, Any]) -> None:
    if _canonical(value) != _canonical(build_plan()):
        raise corpus.CorpusError("immutable pilot plan mismatch")


@lru_cache(maxsize=1)
def pilot_task_ids() -> frozenset[str]:
    return frozenset(cast(str, slot["task_id"]) for slot in build_plan()["slots"])


def write_pilot_plan(output: Path) -> Path:
    _require_component(output, "pilot-plan-v7", file_path=True)
    atomic_create(output, _canonical(build_plan()) + b"\n")
    os.chmod(output, 0o600)
    return output


def _require_component(path: Path, component: str, *, file_path: bool = False) -> None:
    resolved = path.resolve()
    directory = resolved.parent if file_path else resolved
    if directory.name != component:
        raise corpus.CorpusError("pilot artifact path")


def _outside(path: Path, root: Path) -> None:
    child = path.resolve()
    parent = root.resolve()
    if child == parent or parent in child.parents:
        raise corpus.CorpusError("pilot work directory must stay outside the artifact root")


@dataclass(frozen=True)
class SpendScan:
    directory_count: int
    entry_count: int
    settled_count: int
    open_reservation_count: int
    uncertain_count: int
    overspent_count: int
    provider_reported_cost_usd: Decimal
    conservative_debit_usd: Decimal
    open_reservation_debit_usd: Decimal

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": "phase4e-research-spend-index.v1",
            "directory_count": self.directory_count,
            "entry_count": self.entry_count,
            "settled_count": self.settled_count,
            "open_reservation_count": self.open_reservation_count,
            "uncertain_count": self.uncertain_count,
            "overspent_count": self.overspent_count,
            "provider_reported_cost_usd": _money(self.provider_reported_cost_usd),
            "conservative_debit_usd": _money(self.conservative_debit_usd),
            "open_reservation_debit_usd": _money(self.open_reservation_debit_usd),
        }


def scan_research_ledgers(root: Path) -> SpendScan:
    """Read every append-only ledger chain. Open debits count in full."""

    if not root.is_dir():
        raise corpus.CorpusError("artifact root")
    directories = _ledger_directories(root)
    provider = Decimal()
    debit = Decimal()
    open_debit = Decimal()
    entry_count = 0
    settled = 0
    opened = 0
    uncertain = 0
    overspent = 0
    for directory in directories:
        ledger = _validate_ledger_chain(directory)
        for entry in ledger.entries:
            amount = Decimal(entry["debit_usd"])
            provider += Decimal(entry["provider_cost_usd"])
            debit += amount
            entry_count += 1
            status = entry["status"]
            if status == "reserved":
                opened += 1
                open_debit += amount
            elif status == "settled":
                settled += 1
            elif status == "uncertain":
                uncertain += 1
            elif status == "overspent":
                overspent += 1
    return SpendScan(
        directory_count=len(directories),
        entry_count=entry_count,
        settled_count=settled,
        open_reservation_count=opened,
        uncertain_count=uncertain,
        overspent_count=overspent,
        provider_reported_cost_usd=provider,
        conservative_debit_usd=debit,
        open_reservation_debit_usd=open_debit,
    )


def _numbered_files(root: Path, prefix: str) -> list[Path]:
    files = sorted(path for path in root.glob(f"{prefix}-*.json") if path.is_file())
    if any(path.is_symlink() for path in files) or any(
        path.name != f"{prefix}-{index:04d}.json" for index, path in enumerate(files)
    ):
        raise corpus.CorpusError(f"{prefix} chain")
    return files


def _valid_ledger_transition(previous: corpus.BudgetLedger, current: corpus.BudgetLedger) -> bool:
    if previous.policy != current.policy or len(previous.entries) > len(current.entries):
        return False
    mutable = {
        "request_id",
        "provider_cost_usd",
        "debit_usd",
        "response_sha256",
        "status",
    }
    for earlier, later in zip(
        previous.entries, current.entries[: len(previous.entries)], strict=True
    ):
        if earlier == later:
            continue
        if (
            earlier["status"] != "reserved"
            or later["status"] not in {"settled", "overspent", "uncertain"}
            or any(earlier[key] != later[key] for key in set(earlier) - mutable)
            or Decimal(later["debit_usd"]) < Decimal(earlier["debit_usd"])
        ):
            return False
    return previous.provider_journal == current.provider_journal[: len(previous.provider_journal)]


def _validate_ledger_chain(directory: Path) -> corpus.BudgetLedger:
    snapshots = _numbered_files(directory, "ledger")
    if not snapshots:
        raise corpus.CorpusError("research ledger")
    ledgers: list[corpus.BudgetLedger] = []
    try:
        for path in snapshots:
            ledgers.append(corpus.BudgetLedger.from_json(json.loads(path.read_bytes())))
    except (OSError, ValueError, corpus.CorpusError) as error:
        raise corpus.CorpusError("research ledger") from error
    if any(
        not _valid_ledger_transition(previous, current) for previous, current in pairwise(ledgers)
    ):
        raise corpus.CorpusError("research ledger chain")
    return ledgers[-1]


def _ledger_directories(root: Path) -> list[Path]:
    found: set[Path] = set()
    for path in root.rglob("ledger-*.json"):
        if path.is_symlink() or path.parent.is_symlink():
            raise corpus.CorpusError("research ledger rejects symlinks")
        if path.parent.name != "ledger":
            continue
        found.add(path.parent.resolve())
    return sorted(found)


def _research_files(source: Path) -> list[Path]:
    if not source.is_dir():
        raise corpus.CorpusError("research import source")
    files: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise corpus.CorpusError("research import rejects symlinks")
        relative = path.relative_to(source)
        if _is_research_artifact(relative):
            files.append(path)
    if not files:
        raise corpus.CorpusError("research import source")
    return sorted(files)


def _is_research_artifact(relative: Path) -> bool:
    parent = relative.parent.name
    name = relative.name
    if (
        parent == "ledger"
        and name.endswith(".json")
        and (name.startswith("ledger-") or name.startswith("wilson-"))
    ):
        return True
    if parent == "resolved" and name.endswith(".json"):
        return True
    return parent == "diagnostics" and name.startswith("call-") and name.endswith(".json")


def import_research_artifacts(source: Path, destination: Path) -> str:
    """Copy historical research artifacts create-only, then seal the catalog."""

    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for path in _research_files(source):
        relative = path.relative_to(source)
        if ".." in relative.parts:
            raise corpus.CorpusError("research import path")
        payload = path.read_bytes()
        dest = destination / relative
        if dest.exists():
            if dest.is_symlink() or dest.read_bytes() != payload:
                raise FileExistsError(f"benchmark artifact already exists: {dest.name}")
            continue
        dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_create(dest, payload)
        os.chmod(dest, 0o600)
        if dest.read_bytes() != payload:
            raise corpus.CorpusError("research import digest")
    inventory_sha, _scan = record_research_catalog(destination)
    return inventory_sha


def _inventory_payload(root: Path) -> dict[str, Any]:
    files = []
    for path in _research_files(root):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema_version": "phase4e-research-inventory.v1",
        "files": sorted(files, key=lambda item: cast(str, item["path"])),
    }


def _latest_numbered(root: Path, prefix: str) -> Path | None:
    found = _numbered_files(root, prefix)
    return found[-1] if found else None


def _append_numbered(root: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    existing = _numbered_files(root, prefix)
    path = root / f"{prefix}-{len(existing):04d}.json"
    atomic_create(path, _canonical(payload) + b"\n")
    os.chmod(path, 0o600)
    return path


def verify_inventory(root: Path) -> str:
    paths = _numbered_files(root, "inventory")
    if not paths:
        raise corpus.CorpusError("research inventory missing")
    previous: dict[str, str] = {}
    latest_payload: dict[str, Any] | None = None
    for path in paths:
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise corpus.CorpusError("research inventory") from error
        files = payload.get("files") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "files"}
            or payload.get("schema_version") != "phase4e-research-inventory.v1"
            or not isinstance(files, list)
        ):
            raise corpus.CorpusError("research inventory")
        current: dict[str, str] = {}
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise corpus.CorpusError("research inventory")
            relative = item["path"]
            digest = item["sha256"]
            if (
                not isinstance(relative, str)
                or not isinstance(digest, str)
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or relative in current
                or not len(digest) == 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise corpus.CorpusError("research inventory")
            current[relative] = digest
        if any(current.get(name) != digest for name, digest in previous.items()):
            raise corpus.CorpusError("research inventory chain")
        previous = current
        latest_payload = payload
    actual_payload = _inventory_payload(root)
    if latest_payload != actual_payload:
        raise corpus.CorpusError("research inventory does not match artifact root")
    return hashlib.sha256(paths[-1].read_bytes()).hexdigest()


def _spend_scan_from_json(payload: object) -> SpendScan:
    if not isinstance(payload, dict):
        raise corpus.CorpusError("research spend index")
    try:
        scan = SpendScan(
            directory_count=payload["directory_count"],
            entry_count=payload["entry_count"],
            settled_count=payload["settled_count"],
            open_reservation_count=payload["open_reservation_count"],
            uncertain_count=payload["uncertain_count"],
            overspent_count=payload["overspent_count"],
            provider_reported_cost_usd=Decimal(payload["provider_reported_cost_usd"]),
            conservative_debit_usd=Decimal(payload["conservative_debit_usd"]),
            open_reservation_debit_usd=Decimal(payload["open_reservation_debit_usd"]),
        )
    except (ArithmeticError, KeyError, TypeError) as error:
        raise corpus.CorpusError("research spend index") from error
    if payload != scan.as_json() or any(
        type(getattr(scan, key)) is not int or getattr(scan, key) < 0
        for key in (
            "directory_count",
            "entry_count",
            "settled_count",
            "open_reservation_count",
            "uncertain_count",
            "overspent_count",
        )
    ):
        raise corpus.CorpusError("research spend index")
    return scan


def _validate_spend_index_chain(root: Path) -> SpendScan:
    paths = _numbered_files(root, "spend-index")
    if not paths:
        raise corpus.CorpusError("research spend index missing")
    scans: list[SpendScan] = []
    try:
        for path in paths:
            scans.append(_spend_scan_from_json(json.loads(path.read_bytes())))
    except (OSError, ValueError) as error:
        raise corpus.CorpusError("research spend index") from error
    monotonic = (
        "directory_count",
        "entry_count",
        "settled_count",
        "uncertain_count",
        "overspent_count",
        "provider_reported_cost_usd",
        "conservative_debit_usd",
    )
    if any(
        any(getattr(current, key) < getattr(previous, key) for key in monotonic)
        for previous, current in pairwise(scans)
    ):
        raise corpus.CorpusError("research spend index chain")
    return scans[-1]


def record_research_catalog(root: Path) -> tuple[str, SpendScan]:
    scan = scan_research_ledgers(root)
    payload = _inventory_payload(root)
    current = _latest_numbered(root, "inventory")
    if current is None or json.loads(current.read_bytes()) != json.loads(_canonical(payload)):
        current = _append_numbered(root, "inventory", payload)
    inventory_sha = hashlib.sha256(current.read_bytes()).hexdigest()
    index_payload = scan.as_json()
    latest = _latest_numbered(root, "spend-index")
    if latest is None or json.loads(latest.read_bytes()) != json.loads(_canonical(index_payload)):
        _append_numbered(root, "spend-index", index_payload)
    return inventory_sha, scan


def require_pilot_research_history(root: Path) -> tuple[SpendScan, str]:
    """Validate append-only history; financial totals are telemetry, never a gate."""

    inventory_sha = verify_inventory(root)
    scan = scan_research_ledgers(root)
    baseline = validate_spend_baseline()
    if (
        scan.directory_count < baseline["directory_count"]
        or scan.entry_count < baseline["entry_count"]
        or scan.settled_count < baseline["settled_count"]
        or scan.provider_reported_cost_usd < Decimal(baseline["provider_reported_cost_usd"])
        or scan.conservative_debit_usd < Decimal(baseline["conservative_debit_usd"])
    ):
        raise corpus.CorpusError("cumulative spend baseline")
    indexed = _validate_spend_index_chain(root)
    if indexed != scan or indexed.conservative_debit_usd < Decimal(
        baseline["conservative_debit_usd"]
    ):
        raise corpus.CorpusError("cumulative spend baseline")
    return scan, inventory_sha


def canonical_research_ledger_root() -> Path:
    """Return the platform state root. Tests and the CLI pass ``--artifact-root`` explicitly."""

    try:
        from platformdirs import user_state_path  # type: ignore[import-not-found]
    except ImportError as error:
        raise corpus.CorpusError("durable research root requires platformdirs") from error
    return Path(user_state_path("saracura")) / "phase4e" / "research-ledgers"


def _semantic_kind(row: Mapping[str, Any], review: Mapping[str, Any]) -> str | None:
    try:
        author = corpus.SemanticEquivalenceAttestation.model_validate(
            row["semantic_equivalence_attestation"]
        )
        reviewer = corpus.SemanticEquivalenceAttestation.model_validate(
            review["semantic_equivalence_attestation"]
        )
        selected = next(
            index
            for index, criterion in enumerate(row["criteria"])
            if criterion["id"] == review["selected_criterion_id"]
        )
    except (KeyError, StopIteration, TypeError, ValueError):
        return None
    return corpus._attestation_disagreement_reason(
        author,
        reviewer,
        option_count=len(row["criteria"]),
        selected_position=selected,
    )


def build_pilot_report(
    *,
    plan: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    ledger: corpus.BudgetLedger,
    diagnostics: Mapping[str, int],
    operational: bool,
    artifact_root: Path,
    inventory_sha256: str,
    historical: SpendScan,
) -> dict[str, Any]:
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    accepted_ids = {cast(str, row["task_id"]) for row in accepted}
    rejected_ids = {cast(str, row["task_id"]) for row in rejected}
    unresolved = [slot for slot in slots if slot["task_id"] not in accepted_ids | rejected_ids]
    locale_accepted = {
        locale: sum(row.get("locale") == locale for row in accepted) for locale in corpus.LOCALES
    }
    option_accepted = {str(count): 0 for count in corpus.OPTION_COUNTS}
    for row in accepted:
        count = _option_count(row)
        if count in corpus.OPTION_COUNTS:
            option_accepted[str(count)] += 1
    cells = {}
    for locale in corpus.LOCALES:
        for count in corpus.OPTION_COUNTS:
            members = [
                slot for slot in slots if slot["locale"] == locale and slot["option_count"] == count
            ]
            cells[f"{locale}|{count}"] = {
                "accepted": sum(slot["task_id"] in accepted_ids for slot in members),
                "total": len(members),
            }
    pairs: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        pair_id = row.get("pair_id")
        member_locale = row.get("locale")
        if isinstance(pair_id, str) and isinstance(member_locale, str):
            pairs[pair_id].add(member_locale)
    complete_pairs = sum(locales == set(corpus.LOCALES) for locales in pairs.values())
    resolved_count = len(accepted) + len(rejected)
    global_wilson = (
        corpus.wilson_lower_bound(len(accepted), resolved_count) if resolved_count else None
    )
    pair_wilson = corpus.wilson_lower_bound(complete_pairs, 70)
    thresholds = validate_pilot_policy()["thresholds"]
    quality_pass = (
        not operational
        and not unresolved
        and len(accepted) >= thresholds["minimum_accepted"]
        and all(
            locale_accepted[locale] >= thresholds["minimum_accepted_per_locale"]
            for locale in corpus.LOCALES
        )
        and all(
            option_accepted[str(count)] >= thresholds["minimum_accepted_per_option_count"]
            for count in corpus.OPTION_COUNTS
        )
        and diagnostics["local_privacy"] <= thresholds["maximum_local_privacy_violations"]
        and diagnostics["reviewer_privacy_flags"] <= thresholds["maximum_reviewer_privacy_flags"]
    )
    if operational or unresolved or diagnostics["missing_journal"] or diagnostics["uncertain"]:
        decision = "INCONCLUSIVE"
    elif quality_pass:
        decision = "PASS"
    else:
        decision = "FAIL"
    this_provider = sum(
        (Decimal(entry["provider_cost_usd"]) for entry in ledger.entries), Decimal()
    )
    this_debit = sum((Decimal(entry["debit_usd"]) for entry in ledger.entries), Decimal())
    return {
        "schema_version": "phase4e-protocol-pilot-report.v7",
        "decision": decision,
        "seed": plan["seed"],
        "plan_sha256": _sha(_canonical(plan)),
        "task_count": 140,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "unresolved_count": len(unresolved),
        "locale_accepted": locale_accepted,
        "option_count_accepted": option_accepted,
        "cells": cells,
        "complete_pairs": complete_pairs,
        "pair_count": 70,
        "global_wilson_lower_bound": global_wilson,
        "complete_pair_wilson_lower_bound": pair_wilson,
        "semantic_diagnostics": {
            "reviewed": diagnostics["reviewed"],
            "scenario_disagreement": diagnostics["scenario_disagreement"],
            "criterion_role_disagreement": diagnostics["criterion_role_disagreement"],
        },
        "response_diagnostics": {
            "author_failures": diagnostics["author_response_failures"],
            "reviewer_failures": diagnostics["reviewer_response_failures"],
            "retry_recoveries": diagnostics["retry_recoveries"],
            "orphaned_settled_calls": diagnostics["orphaned_settled_calls"],
        },
        "local_privacy_violations": diagnostics["local_privacy"],
        "reviewer_privacy_flags": diagnostics["reviewer_privacy_flags"],
        "operational_failures": diagnostics["operational_failures"],
        "provider_reported_cost_usd": _money(this_provider),
        "conservative_debit_usd": _money(this_debit),
        "cumulative_provider_reported_cost_usd": _money(
            historical.provider_reported_cost_usd + this_provider
        ),
        "cumulative_conservative_debit_usd": _money(historical.conservative_debit_usd + this_debit),
        "cost_estimate": plan["cost_estimate"],
        "artifact_root": str(artifact_root.resolve()),
        "inventory_sha256": inventory_sha256,
        "ledger_sha256": _sha(_canonical(ledger.as_json(final=True))),
    }


def _option_count(row: Mapping[str, Any]) -> int | None:
    declared = row.get("option_count")
    if type(declared) is int:
        return declared
    criteria = row.get("criteria")
    if isinstance(criteria, list):
        return len(criteria)
    return None


def _blank_diagnostics() -> dict[str, int]:
    return {
        "reviewed": 0,
        "scenario_disagreement": 0,
        "criterion_role_disagreement": 0,
        "local_privacy": 0,
        "reviewer_privacy_flags": 0,
        "author_response_failures": 0,
        "reviewer_response_failures": 0,
        "retry_recoveries": 0,
        "orphaned_settled_calls": 0,
        "operational_failures": 0,
        "missing_journal": 0,
        "missing_diagnostics": 0,
        "uncertain": 0,
    }


def _diagnostics_path(work_dir: Path, slots: Sequence[Mapping[str, Any]]) -> Path:
    task_ids = sorted(cast(str, slot["task_id"]) for slot in slots)
    digest = _sha(_canonical(task_ids))
    return work_dir / "diagnostics" / f"call-{digest}.json"


def _store_pilot_diagnostics(
    work_dir: Path,
    slots: Sequence[Mapping[str, Any]],
    values: Mapping[str, int],
) -> None:
    task_ids = sorted(cast(str, slot["task_id"]) for slot in slots)
    if set(values) != set(_PERSISTED_DIAGNOSTIC_KEYS) or any(
        type(values[key]) is not int or values[key] < 0 for key in _PERSISTED_DIAGNOSTIC_KEYS
    ):
        raise corpus.CorpusError("pilot diagnostics")
    atomic_create(
        _diagnostics_path(work_dir, slots),
        _canonical(
            {
                "schema_version": "phase4e-protocol-pilot-diagnostics.v2",
                "task_ids": task_ids,
                "counts": {key: values[key] for key in _PERSISTED_DIAGNOSTIC_KEYS},
            }
        )
        + b"\n",
    )


def _load_pilot_diagnostics(
    work_dir: Path,
    plan: Mapping[str, Any],
    completed: set[str],
) -> tuple[dict[str, int], bool]:
    totals = {key: 0 for key in _PERSISTED_DIAGNOSTIC_KEYS}
    expected_paths: set[Path] = set()
    complete = True
    for batch in _pair_batches(cast(list[Mapping[str, Any]], plan["slots"])):
        task_ids = {cast(str, slot["task_id"]) for slot in batch}
        if not task_ids <= completed:
            continue
        path = _diagnostics_path(work_dir, batch)
        expected_paths.add(path)
        try:
            payload = _read_object(path)
        except corpus.CorpusError:
            complete = False
            continue
        counts = payload.get("counts")
        if (
            set(payload) != {"schema_version", "task_ids", "counts"}
            or payload["schema_version"] != "phase4e-protocol-pilot-diagnostics.v2"
            or payload["task_ids"] != sorted(task_ids)
            or not isinstance(counts, dict)
            or set(counts) != set(_PERSISTED_DIAGNOSTIC_KEYS)
            or any(
                type(counts[key]) is not int or counts[key] < 0
                for key in _PERSISTED_DIAGNOSTIC_KEYS
            )
        ):
            complete = False
            continue
        for key in _PERSISTED_DIAGNOSTIC_KEYS:
            totals[key] += counts[key]
    diagnostics_dir = work_dir / "diagnostics"
    actual_paths = set(diagnostics_dir.glob("call-*.json")) if diagnostics_dir.exists() else set()
    if actual_paths != expected_paths:
        complete = False
    return totals, complete


def run_pilot(
    plan_path: Path,
    snapshot: Path | None,
    work_dir: Path,
    report_dir: Path,
    artifact_root: Path,
    *,
    allow_network: bool,
    transport: corpus.Transport | None = None,
    counter: corpus.TokenCounter | None = None,
) -> Path:
    """Resolve all 140 pilot identities with report-only cost telemetry."""

    from benchmarks import phase4e_pipeline as pipeline

    _require_component(plan_path, "pilot-plan-v7", file_path=True)
    _require_component(work_dir, "pilot-work-v7")
    _require_component(report_dir, "pilot-report-v7")
    if not allow_network:
        raise corpus.CorpusError("pilot requires literal --allow-network")
    plan = _read_object(plan_path)
    validate_pilot_plan(plan)
    _outside(work_dir, artifact_root)
    _outside(report_dir, artifact_root)
    historical, _inventory_sha = require_pilot_research_history(artifact_root)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise corpus.CorpusError("missing provider credential")
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    recovery = validate_pilot_recovery_policy()
    totals = preflight_bounds(slots)
    recorded = plan["cost_estimate"]
    if totals[AUTHOR_STAGE] != Decimal(recorded["pilot_author_usd"]) or totals[
        REVIEWER_STAGE
    ] != Decimal(recorded["pilot_reviewer_usd"]):
        raise corpus.CorpusError("pilot preflight mismatch")
    if counter is None:
        if snapshot is None:
            raise corpus.CorpusError("pilot requires verified tokenizer")
        counter = corpus.VerifiedMiniLMTokenizerReceipt.create(snapshot)
    work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = pipeline._load_resolved(work_dir, plan)
    resumed = corpus.resume_ledger(work_dir / "ledger")
    ledger = resumed if resumed.entries else corpus.BudgetLedger(policy=corpus.PILOT_LEDGER_POLICY)
    if ledger.policy.schema_version != corpus.PILOT_LEDGER_POLICY.schema_version:
        raise corpus.CorpusError("ledger resume mismatch")
    chosen_transport = transport if transport is not None else pipeline._openrouter_transport()
    client = corpus.OpenRouterCorpusClient(
        transport=cast(corpus.Transport, chosen_transport),
        allow_network=True,
        ledger=ledger,
        ledger_directory=work_dir / "ledger",
        policy=corpus.PILOT_LEDGER_POLICY,
        author_stage=AUTHOR_STAGE,
        reviewer_stage=REVIEWER_STAGE,
        author_model=PILOT_AUTHOR_MODEL,
        reviewer_model=PILOT_REVIEWER_MODEL,
        provider_preferences_by_stage={
            AUTHOR_STAGE: PILOT_PROVIDER_POLICY,
            REVIEWER_STAGE: PILOT_PROVIDER_POLICY,
        },
        author_reasoning_effort=None,
    )
    maximum_attempts = cast(int, recovery["maximum_attempts_per_request"])
    report_interval = Decimal(cast(str, recovery["cost_report_interval_usd"]))
    next_cost_report = report_interval
    diagnostics = _blank_diagnostics()
    accepted = [
        cast(dict[str, Any], value["row"])
        for value in resolved.values()
        if value["status"] == "accepted"
    ]
    rejected = [
        cast(dict[str, Any], value["row"])
        for value in resolved.values()
        if value["status"] == "rejected"
    ]
    completed = set(resolved)
    restored_diagnostics, diagnostics_complete = _load_pilot_diagnostics(work_dir, plan, completed)
    for key, value in restored_diagnostics.items():
        diagnostics[key] = value
    operational = any(
        entry["status"] in {"uncertain", "reserved"} for entry in client.ledger.entries
    )
    if not diagnostics_complete:
        diagnostics["missing_diagnostics"] = 1
        operational = True
    if any(entry["status"] == "uncertain" for entry in client.ledger.entries):
        diagnostics["uncertain"] = 1
    try:
        client.ledger.require_complete_provider_journal()
    except corpus.CorpusError:
        diagnostics["missing_journal"] = 1
        operational = True
    try:
        pipeline._require_settled_task_resolution(client.ledger, resolved)
    except corpus.CorpusError:
        diagnostics["orphaned_settled_calls"] = 1
        operational = True
    if not operational:
        batches = pipeline._author_batches(plan)
        for batch in batches:
            batch_slots: list[Mapping[str, Any]] = list(batch)
            batch_ids = [cast(str, slot["task_id"]) for slot in batch_slots]
            if all(task_id in completed for task_id in batch_ids):
                continue
            if any(task_id in completed for task_id in batch_ids):
                raise corpus.CorpusError("partial author family resume")
            try:
                _run_pilot_batch(
                    pipeline,
                    client,
                    batch_slots,
                    counter,
                    api_key,
                    accepted,
                    rejected,
                    diagnostics,
                    work_dir,
                    maximum_attempts=maximum_attempts,
                )
            except corpus.CorpusError:
                if not any(entry["status"] == "uncertain" for entry in client.ledger.entries):
                    raise
                diagnostics["operational_failures"] += 1
                operational = True
                diagnostics["uncertain"] = 1
                break
            completed.update(batch_ids)
            run_cost = sum(
                (Decimal(entry["provider_cost_usd"]) for entry in client.ledger.entries),
                Decimal(),
            )
            while run_cost >= next_cost_report:
                print(
                    f"phase4e pilot provider spend crossed USD {_money(next_cost_report)}",
                    flush=True,
                )
                next_cost_report += report_interval
    all_ids = {cast(str, slot["task_id"]) for slot in slots}
    if completed != all_ids and not operational:
        # A quiet shortfall is still unresolved and therefore inconclusive.
        operational = True
    if operational:
        diagnostics["operational_failures"] = max(diagnostics["operational_failures"], 1)
    copied = _copy_work_tree(work_dir, artifact_root / work_dir.name)
    del copied
    inventory_sha, rescan = record_research_catalog(artifact_root)
    report = build_pilot_report(
        plan=plan,
        accepted=accepted,
        rejected=rejected,
        ledger=client.ledger,
        diagnostics=diagnostics,
        operational=operational
        or bool(
            diagnostics["uncertain"]
            or diagnostics["missing_journal"]
            or diagnostics["missing_diagnostics"]
        ),
        artifact_root=artifact_root,
        inventory_sha256=inventory_sha,
        historical=historical,
    )
    # The catalog rescan includes this run after the copy. Prefer that cumulative total.
    report["cumulative_provider_reported_cost_usd"] = _money(rescan.provider_reported_cost_usd)
    report["cumulative_conservative_debit_usd"] = _money(rescan.conservative_debit_usd)
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    atomic_create(report_path, _canonical(report) + b"\n")
    os.chmod(report_path, 0o600)
    ledger_path = report_dir / "ledger.json"
    atomic_create(ledger_path, _canonical(client.ledger.as_json(final=True)) + b"\n")
    os.chmod(ledger_path, 0o600)
    print(
        "phase4e pilot final provider spend USD " + report["provider_reported_cost_usd"],
        flush=True,
    )
    return report_path


def _run_pilot_batch(
    pipeline: Any,
    client: corpus.OpenRouterCorpusClient,
    slots: Sequence[Mapping[str, Any]],
    counter: corpus.TokenCounter,
    api_key: str,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    diagnostics: dict[str, int],
    work_dir: Path,
    *,
    maximum_attempts: int,
) -> None:
    diagnostics_before = {key: diagnostics[key] for key in _PERSISTED_DIAGNOSTIC_KEYS}
    messages = corpus.author_messages(slots)
    author_schema = corpus.author_schema(slots)
    task_ids = [cast(str, slot["task_id"]) for slot in slots]
    usable: list[dict[str, Any]] | None = None
    rejected_author: list[dict[str, Any]] | None = None
    author_lineage: dict[str, str] | None = None
    for attempt in range(maximum_attempts):
        try:
            provider_response = client._request(
                stage=AUTHOR_STAGE,
                model=PILOT_AUTHOR_MODEL,
                messages=messages,
                response_schema=author_schema,
                max_output_tokens=AUTHOR_MAX_OUTPUT_TOKENS,
                api_key=api_key,
                task_ids=task_ids,
            )
        except corpus.CorpusError:
            diagnostics["author_response_failures"] += 1
            if client.ledger.entries and client.ledger.entries[-1]["status"] == "uncertain":
                raise
            author_lineage = corpus.lineage_from_journal(
                client.last_journal(AUTHOR_STAGE), "author", expected_stage=AUTHOR_STAGE
            )
            break
        author_lineage = corpus.lineage_from_journal(
            client.last_journal(AUTHOR_STAGE), "author", expected_stage=AUTHOR_STAGE
        )
        try:
            author_response = pipeline._response_content(provider_response)
            candidate_usable, candidate_rejected = corpus.classify_author_rows(
                corpus.decode_author_response(author_response, slots), slots, counter
            )
            if candidate_rejected:
                diagnostics["author_response_failures"] += 1
                if attempt + 1 < maximum_attempts:
                    continue
                break
            usable, rejected_author = candidate_usable, candidate_rejected
            if attempt:
                diagnostics["retry_recoveries"] += 1
            break
        except corpus.CorpusError:
            diagnostics["author_response_failures"] += 1
            if attempt + 1 < maximum_attempts:
                continue
            break
    if usable is None or rejected_author is None:
        if author_lineage is None:
            raise corpus.CorpusError("author recovery invariant")
        stored = pipeline._store_settled_fallback_required(
            work_dir,
            slots,
            reason="author_response_failure",
            author_lineage=author_lineage,
            reviewer_lineages={},
        )
        _store_pilot_diagnostics(
            work_dir,
            slots,
            {key: diagnostics[key] - diagnostics_before[key] for key in _PERSISTED_DIAGNOSTIC_KEYS},
        )
        rejected.extend(stored)
        return
    if author_lineage is None:
        raise corpus.CorpusError("author recovery invariant")
    usable = [{**row, **author_lineage} for row in usable]
    rejected_author = [{**row, **author_lineage} for row in rejected_author]
    for row in [*usable, *rejected_author]:
        if "instruction" in row and corpus._privacy(row):
            diagnostics["local_privacy"] += 1
    usable, rejected_author = apply_pair_author_target_rule(slots, usable, rejected_author)
    reviews: list[dict[str, Any]] = []
    reviewed_rows: list[dict[str, Any]] = []
    reviewer_failures: list[dict[str, Any]] = []
    reviewer_lineages: dict[str, dict[str, str]] = {}
    for row in usable:
        review_body = pilot_reviewer_messages(row)
        task_id = cast(str, row["task_id"])
        review: dict[str, Any] | None = None
        for attempt in range(maximum_attempts):
            try:
                provider_response = client._request(
                    stage=REVIEWER_STAGE,
                    model=PILOT_REVIEWER_MODEL,
                    messages=review_body,
                    response_schema=pilot_reviewer_schema(1),
                    max_output_tokens=REVIEWER_MAX_OUTPUT_TOKENS,
                    api_key=api_key,
                    task_ids=[task_id],
                )
            except corpus.CorpusError:
                diagnostics["reviewer_response_failures"] += 1
                if client.ledger.entries and client.ledger.entries[-1]["status"] == "uncertain":
                    raise
                journal = client.last_journal(REVIEWER_STAGE)
                reviewer_lineages[task_id] = corpus.lineage_from_journal(
                    journal, "reviewer", expected_stage=REVIEWER_STAGE
                )
                break
            journal = client.last_journal(REVIEWER_STAGE)
            reviewer_lineages[task_id] = corpus.lineage_from_journal(
                journal, "reviewer", expected_stage=REVIEWER_STAGE
            )
            try:
                response = pipeline._response_content(provider_response)
                raw_reviews = response.get("reviews")
                if (
                    not isinstance(raw_reviews, list)
                    or len(raw_reviews) != 1
                    or not isinstance(raw_reviews[0], dict)
                ):
                    raise corpus.CorpusError("review response")
                review = bind_pilot_reviewer_record(cast(dict[str, Any], raw_reviews[0]), task_id)
                if attempt:
                    diagnostics["retry_recoveries"] += 1
                break
            except corpus.CorpusError:
                diagnostics["reviewer_response_failures"] += 1
                if attempt + 1 < maximum_attempts:
                    continue
                break
        if review is None:
            reviewer_failures.append(
                {
                    "task_id": task_id,
                    "split": row["split"],
                    "reason": "reviewer_response_failure",
                    **author_lineage,
                    **reviewer_lineages[task_id],
                }
            )
            continue
        reviews.append(review)
        reviewed_rows.append(row)
        diagnostics["reviewed"] += 1
        if review["private_or_sensitive"]:
            diagnostics["reviewer_privacy_flags"] += 1
        kind = _semantic_kind(row, review)
        if kind == "scenario_disagreement":
            diagnostics["scenario_disagreement"] += 1
        elif kind == "criterion_role_disagreement":
            diagnostics["criterion_role_disagreement"] += 1
    newly_accepted, newly_rejected = corpus.resolve_reviews(
        reviewed_rows,
        reviews,
        prior_rows=accepted,
        reviewer_lineages=reviewer_lineages,
        acceptance=pilot_acceptance,
        pair_resolution=pilot_pair_resolution,
    )
    outcomes = [
        *[(item, "rejected") for item in rejected_author],
        *[(item, "rejected") for item in reviewer_failures],
        *[(item, "accepted") for item in newly_accepted],
        *[(item, "rejected") for item in newly_rejected],
    ]
    pipeline._store_call_resolution(work_dir, slots, outcomes)
    _store_pilot_diagnostics(
        work_dir,
        slots,
        {key: diagnostics[key] - diagnostics_before[key] for key in _PERSISTED_DIAGNOSTIC_KEYS},
    )
    accepted.extend(newly_accepted)
    rejected.extend([*rejected_author, *reviewer_failures, *newly_rejected])


def _copy_work_tree(work_dir: Path, destination: Path) -> int:
    copied = 0
    if not work_dir.exists():
        return 0
    for path in work_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(work_dir)
        if not _is_research_artifact(relative):
            continue
        payload = path.read_bytes()
        dest = destination / relative
        if dest.exists():
            if dest.read_bytes() != payload:
                raise FileExistsError(f"benchmark artifact already exists: {dest.name}")
            continue
        dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_create(dest, payload)
        os.chmod(dest, 0o600)
        copied += 1
    return copied


__all__ = [
    "AUTHOR_STAGE",
    "PILOT_PLAN_SCHEMA",
    "PILOT_SEED",
    "REVIEWER_STAGE",
    "SpendScan",
    "apply_pair_author_target_rule",
    "build_plan",
    "canonical_research_ledger_root",
    "import_research_artifacts",
    "pilot_acceptance",
    "pilot_pair_resolution",
    "pilot_reviewer_messages",
    "pilot_reviewer_view",
    "pilot_task_ids",
    "preflight_bounds",
    "require_pilot_research_history",
    "run_pilot",
    "scan_research_ledgers",
    "validate_pilot_plan",
    "validate_pilot_policy",
    "validate_spend_baseline",
    "write_pilot_plan",
]
