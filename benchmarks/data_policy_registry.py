"""Closed, metadata-only source policy registry for Phase 2C."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTROL_IDS = frozenset(
    {
        "artifact_manifest",
        "byte_hashes",
        "record_provenance",
        "human_author_identity",
        "independent_human_review",
        "provider_terms",
        "teacher_lineage",
        "privacy_record_review",
        "rights_record_review",
        "attribution_notice",
        "modification_notice",
        "native_split_preservation",
        "taxonomy_separation",
        "grouped_split",
        "cross_locale_family_link",
        "duplicate_scan",
        "contamination_scan",
        "shortcut_audit",
        "takedown_lineage",
    }
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[a-z][a-z0-9-]{2,63}$")

MINIMUMS: dict[str, frozenset[str]] = {
    "saracura-human-original": frozenset(
        {
            "artifact_manifest",
            "byte_hashes",
            "record_provenance",
            "human_author_identity",
            "independent_human_review",
            "privacy_record_review",
            "rights_record_review",
            "grouped_split",
            "cross_locale_family_link",
            "duplicate_scan",
            "contamination_scan",
            "shortcut_audit",
            "takedown_lineage",
        }
    ),
    "saracura-deterministic-derivatives": frozenset(
        {
            "artifact_manifest",
            "byte_hashes",
            "record_provenance",
            "human_author_identity",
            "privacy_record_review",
            "rights_record_review",
            "grouped_split",
            "cross_locale_family_link",
            "duplicate_scan",
            "contamination_scan",
            "shortcut_audit",
            "takedown_lineage",
        }
    ),
    "provider-model-generated": frozenset(
        {
            "artifact_manifest",
            "byte_hashes",
            "record_provenance",
            "provider_terms",
            "teacher_lineage",
            "privacy_record_review",
            "rights_record_review",
            "grouped_split",
            "cross_locale_family_link",
            "duplicate_scan",
            "contamination_scan",
            "shortcut_audit",
            "takedown_lineage",
        }
    ),
    "amazon-massive-ptpt": frozenset(
        {
            "artifact_manifest",
            "byte_hashes",
            "record_provenance",
            "privacy_record_review",
            "rights_record_review",
            "attribution_notice",
            "modification_notice",
            "native_split_preservation",
            "taxonomy_separation",
            "duplicate_scan",
            "contamination_scan",
            "shortcut_audit",
            "takedown_lineage",
        }
    ),
    "community-massive-ptbr": frozenset(
        {
            "artifact_manifest",
            "byte_hashes",
            "record_provenance",
            "provider_terms",
            "teacher_lineage",
            "independent_human_review",
            "privacy_record_review",
            "rights_record_review",
            "attribution_notice",
            "modification_notice",
            "native_split_preservation",
            "taxonomy_separation",
            "cross_locale_family_link",
            "duplicate_scan",
            "contamination_scan",
            "shortcut_audit",
            "takedown_lineage",
        }
    ),
    "felhen-aios-operational": frozenset(
        {"privacy_record_review", "rights_record_review", "takedown_lineage"}
    ),
    "customer-support-exports": frozenset(
        {"privacy_record_review", "rights_record_review", "takedown_lineage"}
    ),
    "unvetted-public-text": frozenset(
        {"record_provenance", "privacy_record_review", "rights_record_review"}
    ),
}

IDENTITY: dict[str, tuple[str, str, str, str, str, frozenset[str]]] = {
    "saracura-human-original": (
        "human_original",
        "first_party",
        "human_original",
        "Apache-2.0",
        "conditionally_allowed",
        frozenset({"train", "dev", "calibration", "blind_test"}),
    ),
    "saracura-deterministic-derivatives": (
        "deterministic_derivative",
        "first_party",
        "deterministic_derivative",
        "Apache-2.0",
        "conditionally_allowed",
        frozenset({"train"}),
    ),
    "provider-model-generated": (
        "model_generated",
        "unknown",
        "model_generated_or_assisted",
        "unknown",
        "quarantined",
        frozenset(),
    ),
    "amazon-massive-ptpt": (
        "third_party_dataset",
        "third_party",
        "not_applicable",
        "CC-BY-4.0",
        "conditionally_allowed",
        frozenset({"external_control"}),
    ),
    "community-massive-ptbr": (
        "third_party_dataset",
        "third_party",
        "model_generated_or_assisted",
        "CC-BY-4.0",
        "quarantined",
        frozenset(),
    ),
    "felhen-aios-operational": (
        "private_operational",
        "private_felhen",
        "not_applicable",
        "unknown",
        "blocked",
        frozenset(),
    ),
    "customer-support-exports": (
        "customer_export",
        "customer_controller",
        "not_applicable",
        "unknown",
        "blocked",
        frozenset(),
    ),
    "unvetted-public-text": (
        "unvetted_public",
        "unknown",
        "not_applicable",
        "unknown",
        "blocked",
        frozenset(),
    ),
}

FIXED_METADATA: dict[
    str,
    tuple[
        tuple[str, ...],
        str | None,
        str | None,
        str | None,
        str,
        str,
        str,
        str,
    ],
] = {
    "saracura-human-original": (
        ("pt-BR", "en"),
        None,
        "https://www.apache.org/licenses/LICENSE-2.0",
        None,
        "not_created",
        "policy_only",
        "not_used",
        "requires_artifact_review",
    ),
    "saracura-deterministic-derivatives": (
        ("pt-BR", "en"),
        None,
        "https://www.apache.org/licenses/LICENSE-2.0",
        None,
        "not_created",
        "policy_only",
        "not_used",
        "requires_artifact_review",
    ),
    "provider-model-generated": (
        ("unspecified",),
        None,
        None,
        None,
        "not_created",
        "unknown",
        "required_missing",
        "requires_artifact_review",
    ),
    "amazon-massive-ptpt": (
        ("pt-PT",),
        "https://huggingface.co/datasets/AmazonScience/massive",
        "https://creativecommons.org/licenses/by/4.0/",
        "cf8448dca459d700b8a250b270e57f6d1444bf3e",
        "not_acquired",
        "policy_only",
        "not_applicable",
        "requires_artifact_review",
    ),
    "community-massive-ptbr": (
        ("pt-BR",),
        "https://huggingface.co/datasets/Magurofg/massive-pt-br",
        "https://creativecommons.org/licenses/by/4.0/",
        "907f905b4b237c34ba805f942bfbb6d92ec9c81f",
        "not_acquired",
        "unknown",
        "required_missing",
        "requires_artifact_review",
    ),
    "felhen-aios-operational": (
        ("pt-BR", "en"),
        None,
        None,
        None,
        "unavailable",
        "unknown",
        "not_applicable",
        "prohibited",
    ),
    "customer-support-exports": (
        ("any",),
        None,
        None,
        None,
        "unavailable",
        "unknown",
        "not_applicable",
        "prohibited",
    ),
    "unvetted-public-text": (
        ("any",),
        None,
        None,
        None,
        "unavailable",
        "unknown",
        "not_applicable",
        "prohibited",
    ),
}


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=3, max_length=64, pattern=ID_RE.pattern)
    name: str = Field(min_length=3, max_length=120)
    source_type: Literal[
        "human_original",
        "deterministic_derivative",
        "model_generated",
        "third_party_dataset",
        "private_operational",
        "customer_export",
        "unvetted_public",
    ]
    origin: Literal[
        "first_party", "third_party", "private_felhen", "customer_controller", "unknown"
    ]
    policy_state: Literal["conditionally_allowed", "quarantined", "blocked"]
    artifact_approved_for_use: Literal[False]
    planned_uses: list[Literal["train", "dev", "calibration", "blind_test", "external_control"]] = (
        Field(max_length=5)
    )
    locales: list[Literal["pt-BR", "en", "pt-PT", "any", "unspecified"]] = Field(
        min_length=1, max_length=8
    )
    license: Literal["Apache-2.0", "CC-BY-4.0", "unknown"]
    source_url: str | None
    license_url: str | None
    upstream_revision: str | None
    record_identity_state: Literal["not_created", "not_acquired", "unavailable"]
    privacy_review_state: Literal["not_reviewed"]
    rights_review_state: Literal["policy_only", "unknown"]
    authoring_mode: Literal[
        "human_original",
        "deterministic_derivative",
        "model_generated_or_assisted",
        "not_applicable",
    ]
    teacher_lineage_state: Literal["not_used", "required_missing", "not_applicable"]
    redistribution_state: Literal["policy_allowed", "requires_artifact_review", "prohibited"]
    attribution: str = Field(max_length=500)
    required_controls: list[str] = Field(min_length=1, max_length=24)
    reviewed_at: date
    reviewer_role: Literal["saracura-maintainers"]
    rationale: str = Field(min_length=20, max_length=1000)

    @field_validator("planned_uses", "locales", "required_controls")
    @classmethod
    def unique_items(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate list item")
        return value

    @field_validator("source_url", "license_url")
    @classmethod
    def safe_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or re.search(r"/(?:main|master|latest)(?:/|$)", parsed.path.lower())
        ):
            raise ValueError("unsafe URL")
        return value

    @field_validator("upstream_revision")
    @classmethod
    def immutable_revision(cls, value: str | None) -> str | None:
        if value is not None and not HEX40.fullmatch(value):
            raise ValueError("revision must be immutable")
        return value

    @field_validator("reviewed_at", mode="before")
    @classmethod
    def parse_review_date(cls, value: Any) -> date:
        if not isinstance(value, str):
            raise ValueError("reviewed_at must be an ISO date")
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError("reviewed_at must be an ISO date") from None

    @model_validator(mode="after")
    def semantic_contract(self) -> Policy:
        expected = IDENTITY.get(self.id)
        if (
            expected is None
            or (self.source_type, self.origin, self.authoring_mode, self.license, self.policy_state)
            != expected[:5]
        ):
            raise ValueError("fixed identity mismatch")
        if not set(self.planned_uses).issubset(expected[5]):
            raise ValueError("planned uses exceed fixed maximum")
        metadata = FIXED_METADATA[self.id]
        if (
            tuple(self.locales),
            self.source_url,
            self.license_url,
            self.upstream_revision,
            self.record_identity_state,
            self.rights_review_state,
            self.teacher_lineage_state,
            self.redistribution_state,
        ) != metadata:
            raise ValueError("fixed policy metadata mismatch")
        if not set(self.required_controls).issubset(CONTROL_IDS) or not MINIMUMS[self.id].issubset(
            self.required_controls
        ):
            raise ValueError("required controls incomplete")
        if (
            self.artifact_approved_for_use is not False
            or self.privacy_review_state != "not_reviewed"
        ):
            raise ValueError("artifact or privacy state is not fail-closed")
        if self.policy_state in {"quarantined", "blocked"} and self.planned_uses:
            raise ValueError("quarantine/block cannot plan uses")
        if self.policy_state == "blocked" and (
            self.redistribution_state != "prohibited" or self.license != "unknown"
        ):
            raise ValueError("blocked source must be prohibited and unknown-license")
        if (
            self.authoring_mode == "model_generated_or_assisted"
            and self.teacher_lineage_state != "required_missing"
        ):
            raise ValueError("model-assisted source requires missing teacher lineage")
        if self.id == "amazon-massive-ptpt" and (
            self.planned_uses != ["external_control"]
            or self.locales != ["pt-PT"]
            or self.license != "CC-BY-4.0"
            or self.upstream_revision != "cf8448dca459d700b8a250b270e57f6d1444bf3e"
        ):
            raise ValueError("MASSIVE control policy mismatch")
        if (
            self.source_type in {"human_original", "deterministic_derivative"}
            and self.origin != "first_party"
        ):
            raise ValueError("first-party authoring mismatch")
        if self.license == "CC-BY-4.0" and not self.attribution:
            raise ValueError("attribution required")
        if self.license == "unknown" and self.license_url is not None:
            raise ValueError("unknown license cannot have license URL")
        third_party = self.origin == "third_party"
        if third_party != (self.source_url is not None and self.upstream_revision is not None):
            raise ValueError("third-party source needs immutable source metadata")
        if not third_party and (self.source_url is not None or self.upstream_revision is not None):
            raise ValueError("non-third-party source cannot declare upstream metadata")
        if self.license != "unknown" and self.license_url is None:
            raise ValueError("known license needs license URL")
        if (
            self.locales
            and ("any" in self.locales or "unspecified" in self.locales)
            and len(self.locales) != 1
        ):
            raise ValueError("any/unspecified locale must be alone")
        if self.id == "amazon-massive-ptpt" and "pt-BR" in self.locales:
            raise ValueError("pt-PT cannot authorize pt-BR")
        if self.id != "amazon-massive-ptpt" and "external_control" in self.planned_uses:
            raise ValueError("only MASSIVE may be external control")
        return self


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["training-data-source-policies.v1"]
    sources: list[Policy] = Field(min_length=8, max_length=8)
    exceptions: list[SyntheticExperimentException | Phase4EUniversalSyntheticException] = Field(
        min_length=2, max_length=2
    )

    @model_validator(mode="after")
    def exact_ids(self) -> Registry:
        if {source.id for source in self.sources} != set(IDENTITY):
            raise ValueError("registry must contain exactly the eight fixed IDs")
        if tuple(exception.id for exception in self.exceptions) != (
            "synthetic_experiment",
            "phase4e_saracura_universal_synthetic",
        ):
            raise ValueError("registry must contain exactly the two reviewed exceptions")
        if self.exceptions[0].model_dump(mode="json") != _PHASE3B_EXCEPTION:
            raise ValueError("Phase 3B exception content is immutable")
        return self


class SyntheticExperimentException(BaseModel):
    """Explicit workflow-bound exception; it never changes source policy state."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: Literal["synthetic_experiment"]
    workflow_revision: Literal["phase3b-synthetic-research-training.v2"]
    source_policy_id: Literal["provider-model-generated"]
    allowed_splits: list[Literal["synthetic_train", "synthetic_dev", "synthetic_holdout"]]
    allowed_uses: list[Literal["synthetic_only"]]
    author_model: Literal["qwen/qwen3.5-9b"]
    reviewer_model: Literal["mistralai/ministral-8b-2512"]
    canonical_training_authorized: Literal[False]
    calibration_authorized: Literal[False]
    blind_test_authorized: Literal[False]
    publication_authorized: Literal[False]
    quality_claims_allowed: Literal[False]
    automation_authorized: Literal[False]

    @model_validator(mode="after")
    def closed_exception(self) -> SyntheticExperimentException:
        if self.allowed_splits != ["synthetic_train", "synthetic_dev", "synthetic_holdout"]:
            raise ValueError("synthetic split order is fixed")
        if self.allowed_uses != ["synthetic_only"]:
            raise ValueError("synthetic use is fixed")
        return self


class Phase4EUniversalSyntheticException(BaseModel):
    """Second, non-inheriting exception for the planned Saracura-owned corpus."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: Literal["phase4e_saracura_universal_synthetic"]
    workflow_revision: Literal["phase4e-saracura-universal-synthetic.v1"]
    source_policy_id: Literal["provider-model-generated"]
    allowed_splits: list[Literal["synthetic_train", "synthetic_dev", "synthetic_holdout"]]
    allowed_uses: list[Literal["synthetic_only"]]
    author_model: Literal["qwen/qwen3-30b-a3b"]
    reviewer_model: Literal["meta-llama/llama-3.3-70b-instruct"]
    task_slots: Literal[1600]
    spend_ceiling_usd: float
    canonical_training_authorized: Literal[False]
    calibration_authorized: Literal[False]
    blind_test_authorized: Literal[False]
    publication_authorized: Literal[False]
    quality_claims_allowed: Literal[False]
    automation_authorized: Literal[False]

    @field_validator("spend_ceiling_usd", mode="before")
    @classmethod
    def exact_spend_ceiling_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("Phase 4E synthetic spend ceiling must be a JSON float")
        return value

    @model_validator(mode="after")
    def closed_exception(self) -> Phase4EUniversalSyntheticException:
        if self.allowed_splits != ["synthetic_train", "synthetic_dev", "synthetic_holdout"]:
            raise ValueError("Phase 4E synthetic split order is fixed")
        if self.allowed_uses != ["synthetic_only"]:
            raise ValueError("Phase 4E synthetic use is fixed")
        if self.spend_ceiling_usd != 17.0:
            raise ValueError("Phase 4E synthetic spend ceiling is fixed")
        return self


_PHASE3B_EXCEPTION = {
    "id": "synthetic_experiment",
    "workflow_revision": "phase3b-synthetic-research-training.v2",
    "source_policy_id": "provider-model-generated",
    "allowed_splits": ["synthetic_train", "synthetic_dev", "synthetic_holdout"],
    "allowed_uses": ["synthetic_only"],
    "author_model": "qwen/qwen3.5-9b",
    "reviewer_model": "mistralai/ministral-8b-2512",
    "canonical_training_authorized": False,
    "calibration_authorized": False,
    "blind_test_authorized": False,
    "publication_authorized": False,
    "quality_claims_allowed": False,
    "automation_authorized": False,
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_registry(raw: bytes | str) -> Registry:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        return Registry.model_validate(payload)
    except (ValueError, TypeError):
        raise ValueError("invalid data policy registry") from None


def bundled_registry_path() -> Path:
    return Path(__file__).parent / "manifests" / "training-data-source-policies.v1.json"


def load_bundled_registry() -> Registry:
    return load_registry(bundled_registry_path().read_bytes())


__all__ = [
    "Phase4EUniversalSyntheticException",
    "Policy",
    "Registry",
    "SyntheticExperimentException",
    "bundled_registry_path",
    "load_bundled_registry",
    "load_registry",
]
