"""Closed Phase 4E policy manifest validator; it performs no provider action."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from benchmarks.data_policy_registry import (
    Phase4EUniversalSyntheticException,
    Phase4EUniversalSyntheticExceptionV3,
    load_bundled_registry,
    load_bundled_registry_v3,
)
from saracura.serialization import canonical_json_bytes

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks/manifests/phase4e-saracura-universal-policy.v1.json"
POST_PILOT_POLICY_PATH = ROOT / "benchmarks/manifests/phase4e-saracura-universal-policy.v2.json"
POST_PILOT_AUTHOR_MODEL = "openai/gpt-4.1"
POST_PILOT_REVIEWER_MODEL = "openai/gpt-4.1-mini"
POST_PILOT_DOMAIN_SCENARIO_MAP_SHA256 = (
    "946715b68eb920cb8e5c83468997089f90267a77229082b0e2c41554efed7575"
)
AUTHOR_MODEL = "qwen/qwen3-30b-a3b"
REVIEWER_MODEL = "meta-llama/llama-3.3-70b-instruct"
AUTHOR_PROVIDER = "DeepInfra"
REVIEWER_PROVIDER = "CoreWeave"
STAGE_LIMITS = {
    "corpus_author": Decimal("5.00"),
    "corpus_reviewer": Decimal("10.00"),
    "comparison_author": Decimal("0.75"),
    "comparison_reviewer": Decimal("1.25"),
}
TOTAL_BUDGET = Decimal("17.00")


class Phase4EPolicyError(ValueError):
    """The planned-only universal policy failed a closed-manifest check."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = child
    return value


def validate_phase4e_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Validate every reviewed Phase 4E.1 policy axis without network or weights."""

    try:
        policy = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, TypeError, ValueError) as error:
        raise Phase4EPolicyError("Phase 4E policy is not valid JSON") from error
    expected_keys = {
        "schema_version",
        "id",
        "workflow_revision",
        "source_policy_exception_id",
        "base_encoder",
        "ranker",
        "training",
        "provider",
        "budget",
        "capacity",
        "planning",
        "authorizations",
    }
    if not isinstance(policy, dict) or set(policy) != expected_keys:
        raise Phase4EPolicyError("Phase 4E policy shape is closed")
    if (
        policy["schema_version"] != "phase4e-saracura-universal-policy.v1"
        or policy["id"] != "phase4e-saracura-universal-synthetic"
        or policy["workflow_revision"] != "phase4e-saracura-universal-synthetic.v1"
        or policy["source_policy_exception_id"] != "phase4e_saracura_universal_synthetic"
    ):
        raise Phase4EPolicyError("Phase 4E policy identity is invalid")
    if policy["base_encoder"] != {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "frozen": True,
        "dimensions": 384,
    }:
        raise Phase4EPolicyError("base encoder policy is invalid")
    if policy["ranker"] != {
        "architecture_revision": "saracura-universal-ranker.v0",
        "projection_dimensions": 192,
        "criterion_batch_max": 20,
        "dynamic_label_bias": False,
    }:
        raise Phase4EPolicyError("ranker policy is invalid")
    if policy["training"] != {
        "seed": 20260923,
        "optimizer": "AdamW",
        "learning_rate": 0.02,
        "weight_decay": 0.01,
        "batch_size": 16,
        "maximum_epochs": 40,
        "early_stopping_patience": 8,
        "cpu_threads": 1,
        "selection_metric": "stratified_macro_accuracy.v1",
        "dev_improvement_over_baseline": 0.03,
    }:
        raise Phase4EPolicyError("training policy is invalid")
    if policy["provider"] != {
        "host": "openrouter.ai",
        "author_model": AUTHOR_MODEL,
        "author_provider": AUTHOR_PROVIDER,
        "author_endpoint_id": "qwen/qwen3-30b-a3b-04-28",
        "author_provider_tag": "deepinfra/fp8",
        "author_quantization": "fp8",
        "author_context_length": 40960,
        "author_input_price_usd_per_million": 0.12,
        "author_output_price_usd_per_million": 0.5,
        "author_endpoint_checked_at": "2026-09-24",
        "reviewer_model": REVIEWER_MODEL,
        "reviewer_provider": REVIEWER_PROVIDER,
        "allow_fallbacks": False,
        "require_parameters": True,
        "zdr": True,
        "data_collection": "deny",
    }:
        raise Phase4EPolicyError("provider policy is invalid")
    if policy["budget"] != {
        "total_usd": 17.0,
        "author_usd": 5.0,
        "reviewer_usd": 10.0,
        "comparison_author_usd": 0.75,
        "comparison_reviewer_usd": 1.25,
    }:
        raise Phase4EPolicyError("budget policy is invalid")
    if policy["capacity"] != {
        "context_tokens": 128,
        "criterion_tokens": 96,
        "state_codepoints": 200,
        "state_utf8_bytes": 800,
        "instruction_codepoints": 120,
        "instruction_utf8_bytes": 480,
        "criterion_codepoints": 120,
        "criterion_utf8_bytes": 480,
        "questions_min": 1,
        "questions_max": 10,
        "criteria_min": 2,
        "criteria_max": 8,
    }:
        raise Phase4EPolicyError("capacity policy is invalid")
    if policy["planning"] != {
        "task_slots": 1600,
        "seed": "saracura-phase4e-universal-v1",
        "locales": ["pt-BR", "en"],
        "domains": [
            "email_triage",
            "customer_support",
            "finance",
            "accounting",
            "commerce",
            "operations",
            "scheduling",
            "document_routing",
            "browser_action",
            "security_triage",
            "content_moderation",
            "personal_productivity",
        ],
        "splits": {"synthetic_train": 1120, "synthetic_dev": 240, "synthetic_holdout": 240},
        "axes": {
            "explicitness": ["explicit", "implicit"],
            "negation": ["absent", "present"],
            "distractor_overlap": ["low", "high"],
            "urgency": ["normal", "urgent"],
        },
        "cross_locale_pairs": 300,
    }:
        raise Phase4EPolicyError("planning policy is invalid")
    if policy["authorizations"] != {
        "synthetic_generation_authorized": False,
        "synthetic_research_training_authorized": False,
        "real_checkpoint_present": False,
        "runtime_registration_authorized": False,
        "calibration_authorized": False,
        "publication_authorized": False,
        "quality_claims_allowed": False,
        "automation_authorized": False,
    }:
        raise Phase4EPolicyError("authorization policy is invalid")
    registry = load_bundled_registry()
    exception = registry.exceptions[1]
    if (
        not isinstance(exception, Phase4EUniversalSyntheticException)
        or exception.id != policy["source_policy_exception_id"]
    ):
        raise Phase4EPolicyError("source-policy exception is not bound")
    if (
        exception.author_model != AUTHOR_MODEL
        or exception.reviewer_model != REVIEWER_MODEL
        or policy["provider"]["author_model"] != exception.author_model
        or policy["provider"]["reviewer_model"] != exception.reviewer_model
    ):
        raise Phase4EPolicyError("source-policy model lineage is invalid")
    budget = policy["budget"]
    manifest_total = budget["total_usd"]
    stage_total = sum(STAGE_LIMITS.values())
    if (
        type(manifest_total) is not float
        or exception.spend_ceiling_usd != 17.0
        or manifest_total != exception.spend_ceiling_usd
        or stage_total != TOTAL_BUDGET
        or Decimal(str(manifest_total)) != stage_total
    ):
        raise Phase4EPolicyError("Phase 4E budget equivalence is invalid")
    return policy


def require_phase4e_authorization(
    action: Literal["synthetic_generation", "synthetic_research_training"],
) -> dict[str, Any]:
    """Fail closed while the protocol pilot keeps corpus and training at NO-GO."""

    policy = validate_phase4e_policy()
    key = {
        "synthetic_generation": "synthetic_generation_authorized",
        "synthetic_research_training": "synthetic_research_training_authorized",
    }[action]
    if policy["authorizations"].get(key) is not True:
        raise Phase4EPolicyError(f"{action} is not authorized before a reviewed pilot PASS")
    return policy


def validate_post_pilot_phase4e_policy(path: Path = POST_PILOT_POLICY_PATH) -> dict[str, Any]:
    """Validate the create-only v2 execution policy without altering v1."""

    try:
        policy = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, TypeError, ValueError) as error:
        raise Phase4EPolicyError("post-pilot policy is not valid JSON") from error
    expected = {
        "schema_version",
        "id",
        "workflow_revision",
        "source_policy_exception_id",
        "base_encoder",
        "ranker",
        "training",
        "provider",
        "cost",
        "capacity",
        "planning",
        "domain_scenario_map",
        "authorizations",
    }
    if not isinstance(policy, dict) or set(policy) != expected:
        raise Phase4EPolicyError("post-pilot policy shape is closed")
    if (
        policy["schema_version"] != "phase4e-saracura-universal-policy.v2"
        or policy["id"] != "phase4e-saracura-universal-synthetic"
        or policy["workflow_revision"] != "phase4e-saracura-universal-synthetic.v2"
        or policy["source_policy_exception_id"] != "phase4e_saracura_universal_synthetic"
    ):
        raise Phase4EPolicyError("post-pilot policy identity")
    # These contracts intentionally match the historical v1 planning/model lane.
    historical = validate_phase4e_policy()
    for key in ("base_encoder", "ranker", "training", "capacity", "planning"):
        if policy[key] != historical[key]:
            raise Phase4EPolicyError(f"post-pilot {key} drift")
    provider = policy["provider"]
    if not isinstance(provider, dict) or provider != {
        "host": "openrouter.ai",
        "author_model": POST_PILOT_AUTHOR_MODEL,
        "reviewer_model": POST_PILOT_REVIEWER_MODEL,
        "provider_policy": {
            "order": ["Azure"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        },
        "transport_timeout_seconds": 120,
        "author_request": {"max_output_tokens": 1024, "temperature": 0, "reasoning": None},
        "reviewer_request": {"max_output_tokens": 512, "temperature": 0, "reasoning": None},
        "author_system_sha256": "439eb0b348c3c179a7f1c6618a466b05fc92fab24d3638652be98fb8dc473c6c",
        "reviewer_system_sha256": (
            "f5e2f107e4b232099e364156c9dd8c596ab6bbeb4dd40bd27166806bc39627dc"
        ),
    }:
        raise Phase4EPolicyError("post-pilot provider policy")
    cost = policy["cost"]
    if cost != {
        "mode": "report_only",
        "report_interval_usd": "10.00",
        "automatic_retries": 4,
        "prices": {
            "corpus_author": {"input_usd_per_million": "2.00", "output_usd_per_million": "8.00"},
            "corpus_reviewer": {"input_usd_per_million": "0.40", "output_usd_per_million": "1.60"},
        },
    }:
        raise Phase4EPolicyError("post-pilot cost policy")
    mapping = policy["domain_scenario_map"]
    if not isinstance(mapping, dict) or tuple(mapping) != tuple(policy["planning"]["domains"]):
        raise Phase4EPolicyError("post-pilot domain map")
    if any(not isinstance(v, list) or not v or len(v) != len(set(v)) for v in mapping.values()):
        raise Phase4EPolicyError("post-pilot domain map")
    if (
        hashlib.sha256(canonical_json_bytes(mapping)).hexdigest()
        != POST_PILOT_DOMAIN_SCENARIO_MAP_SHA256
    ):
        raise Phase4EPolicyError("post-pilot domain map digest")
    if policy["authorizations"] != {
        "synthetic_generation_authorized": True,
        "synthetic_research_training_authorized": False,
        "real_checkpoint_present": False,
        "runtime_registration_authorized": False,
        "calibration_authorized": False,
        "publication_authorized": False,
        "quality_claims_allowed": False,
        "automation_authorized": False,
    }:
        raise Phase4EPolicyError("post-pilot authorizations")
    registry = load_bundled_registry_v3()
    exception = registry.exceptions[1]
    if not isinstance(exception, Phase4EUniversalSyntheticExceptionV3):
        raise Phase4EPolicyError("post-pilot registry binding")
    if (
        exception.id != policy["source_policy_exception_id"]
        or exception.author_model != POST_PILOT_AUTHOR_MODEL
        or exception.reviewer_model != POST_PILOT_REVIEWER_MODEL
        or exception.synthetic_generation_authorized is not True
    ):
        raise Phase4EPolicyError("post-pilot registry binding")
    return policy


def require_post_pilot_phase4e_authorization(
    action: Literal["synthetic_generation", "synthetic_research_training"],
) -> dict[str, Any]:
    policy = validate_post_pilot_phase4e_policy()
    key = f"{action}_authorized"
    if policy["authorizations"].get(key) is not True:
        raise Phase4EPolicyError(f"{action} is not authorized in post-pilot policy")
    return policy
