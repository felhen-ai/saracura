"""Closed Phase 4E policy manifest validator; it performs no provider action."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from benchmarks.data_policy_registry import (
    Phase4EUniversalSyntheticException,
    load_bundled_registry,
)

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks/manifests/phase4e-saracura-universal-policy.v1.json"
AUTHOR_MODEL = "google/gemini-2.5-flash"
REVIEWER_MODEL = "meta-llama/llama-3.3-70b-instruct"
AUTHOR_PROVIDER = "Google"
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
        "synthetic_generation_authorized": True,
        "synthetic_research_training_authorized": True,
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
