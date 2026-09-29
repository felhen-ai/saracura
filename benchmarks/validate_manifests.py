"""Validate the narrow provenance contract for first-cycle self-authored fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.data_policy_registry import load_registry as load_data_policy_registry
from benchmarks.data_policy_registry import load_registry_v2 as load_data_policy_registry_v2
from benchmarks.data_policy_registry import load_registry_v3 as load_data_policy_registry_v3
from benchmarks.data_policy_registry import load_registry_v4 as load_data_policy_registry_v4
from benchmarks.encoder_registry import load_registry as load_encoder_registry
from benchmarks.first_party_gate import validate_protocol_bytes
from benchmarks.saracura_universal_comparison import (
    COMPARISON_POLICY_PATH,
    validate_comparison_policy,
)
from benchmarks.saracura_universal_pilot import (
    RECOVERY_POLICY_V2_PATH,
    RECOVERY_POLICY_V3_PATH,
    RECOVERY_POLICY_V4_PATH,
    RECOVERY_POLICY_V5_PATH,
    RECOVERY_POLICY_V6_PATH,
    RECOVERY_POLICY_V7_PATH,
    RECOVERY_POLICY_V8_PATH,
    RECOVERY_POLICY_V9_PATH,
    RECOVERY_POLICY_V10_PATH,
    RECOVERY_POLICY_V11_PATH,
    validate_pilot_policy,
    validate_pilot_recovery_policy,
    validate_spend_baseline,
)
from benchmarks.saracura_universal_policy import (
    validate_phase4e_policy,
    validate_post_pilot_phase4e_policy,
)
from benchmarks.saracura_universal_training import validate_v4_training_grant_file
from benchmarks.synthetic_research import validate_synthetic_policy
from benchmarks.universal_bakeoff import load_candidate_registry, load_plan
from benchmarks.universal_local.plan import load_plan as load_universal_local_plan
from benchmarks.universal_local.registry import load_registry as load_universal_local_registry
from benchmarks.universal_remote.plan import validate_plan as validate_universal_remote_plan
from benchmarks.universal_remote_resume.plan import validate_plan as validate_resume_plan

REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "revision",
    "language",
    "native_language",
    "source",
    "license",
    "redistribution",
    "contains_external_data",
    "contains_personal_data",
    "purpose",
    "quality_claims_allowed",
}

HISTORICAL_PILOT_RECOVERY_POLICIES = {
    f"phase4e-protocol-pilot-recovery.v{version}": path
    for version, path in (
        (2, RECOVERY_POLICY_V2_PATH),
        (3, RECOVERY_POLICY_V3_PATH),
        (4, RECOVERY_POLICY_V4_PATH),
        (5, RECOVERY_POLICY_V5_PATH),
        (6, RECOVERY_POLICY_V6_PATH),
        (7, RECOVERY_POLICY_V7_PATH),
        (8, RECOVERY_POLICY_V8_PATH),
        (9, RECOVERY_POLICY_V9_PATH),
        (10, RECOVERY_POLICY_V10_PATH),
        (11, RECOVERY_POLICY_V11_PATH),
    )
}


def validate_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != REQUIRED_FIELDS:
        raise ValueError(f"{path}: manifest shape is not closed")
    if payload["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported schema version")
    if payload["source"] != "self-authored" or payload["contains_external_data"] is not False:
        raise ValueError(f"{path}: phase 1 allows only self-authored fixtures")
    if payload["contains_personal_data"] is not False:
        raise ValueError(f"{path}: fixture declares personal data")
    if payload["quality_claims_allowed"] is not False:
        raise ValueError(f"{path}: fixture cannot support quality claims")
    if payload["license"] != "Apache-2.0" or payload["redistribution"] != "allowed":
        raise ValueError(f"{path}: fixture license or redistribution is not approved")


def validate_routed_manifest(path: Path) -> None:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest root must be an object")
    schema_version = payload.get("schema_version")
    if schema_version == "encoder-candidates.v1":
        load_encoder_registry(raw)
        return
    if schema_version == "training-data-source-policies.v1":
        load_data_policy_registry(raw)
        return
    if schema_version == "training-data-source-policies.v2":
        load_data_policy_registry_v2(raw)
        return
    if schema_version == "training-data-source-policies.v3":
        load_data_policy_registry_v3(raw)
        return
    if schema_version == "training-data-source-policies.v4":
        load_data_policy_registry_v4(raw)
        return
    if schema_version == "phase5d-ptbr-faq-bacen-protocol.v1":
        from benchmarks.ptbr_native.models import load_protocol_manifest

        load_protocol_manifest(raw)
        return
    if schema_version == "phase5d1-position-ensemble.v1":
        from benchmarks.ptbr_native.models import load_position_ensemble_manifest

        load_position_ensemble_manifest(raw)
        return
    if schema_version == "phase4e-protocol-pilot-policy.v1":
        validate_pilot_policy(path)
        return
    if schema_version == "phase4e-protocol-pilot-recovery.v12":
        validate_pilot_recovery_policy(path)
        return
    if schema_version in HISTORICAL_PILOT_RECOVERY_POLICIES:
        canonical = HISTORICAL_PILOT_RECOVERY_POLICIES[schema_version]
        if raw != canonical.read_bytes():
            raise ValueError(f"{path}: historical recovery policy must be canonical")
        return
    if schema_version == "phase4e-spend-baseline.v1":
        validate_spend_baseline(path)
        return
    if schema_version == "support-routing-protocol.v1":
        validate_protocol_bytes(raw)
        return
    if schema_version == "synthetic-research-policy.v1":
        validate_synthetic_policy(path)
        return
    if schema_version == "phase4e-saracura-universal-policy.v1":
        validate_phase4e_policy(path)
        return
    if schema_version == "phase4e-saracura-universal-policy.v2":
        validate_post_pilot_phase4e_policy(path)
        return
    if schema_version == "decision-backend-candidates.v1":
        load_candidate_registry(raw)
        return
    if schema_version == "universal-bakeoff-plan.v1":
        load_plan(raw)
        return
    if schema_version == "decision-backend-candidates.v2":
        load_universal_local_registry(raw)
        return
    if schema_version == "universal-bakeoff-plan.v2":
        load_universal_local_plan(raw)
        return
    if schema_version == "universal-bakeoff-plan.v3":
        # v3 is bound to the actual checked-in predecessor bytes; validation
        # of an alternate raw payload remains available through its own CLI.
        if raw != (Path(__file__).parent / "manifests/universal-bakeoff-plan.v3.json").read_bytes():
            raise ValueError(f"{path}: v3 plan must be the canonical file")
        validate_universal_remote_plan()
        return
    if schema_version == "phase4c3c-resume-plan.v1":
        if raw != (Path(__file__).parent / "manifests/phase4c3c-resume-plan.v1.json").read_bytes():
            raise ValueError(f"{path}: recovery plan must be the canonical file")
        validate_resume_plan()
        return
    if schema_version == "phase4e-saracura-universal-training-authorization.v1":
        validate_v4_training_grant_file(path)
        return
    if schema_version == "phase4e-comparison-policy.v1":
        if raw != COMPARISON_POLICY_PATH.read_bytes():
            raise ValueError(f"{path}: comparison policy must be the canonical file")
        validate_comparison_policy(path)
        return
    if schema_version == "phase4e-comparison-policy.v2":
        canonical_v2 = Path(__file__).parent / "manifests" / "phase4e-comparison-policy.v2.json"
        if raw != canonical_v2.read_bytes():
            raise ValueError(f"{path}: comparison policy must be the canonical file")
        validate_comparison_policy(path)
        return
    if schema_version == "phase5-open-model-candidates.v1":
        validate_phase5_candidate_manifest(path)
        return
    if schema_version == "phase5-open-model-candidates.v2":
        validate_phase5_candidate_manifest_v2(path)
        return
    if schema_version == "phase5-public-readiness.v1":
        validate_phase5_readiness_manifest(path)
        return
    if schema_version == "phase5-kev4b-acquisition.v1":
        validate_phase5b_acquisition_descriptor(path)
        return
    if schema_version == "phase5-kev4b-managed-cuda.v1":
        validate_phase5_managed_cuda_runtime(path)
        return
    if schema_version == "phase5-managed-cuda-systems.v1":
        validate_phase5_managed_cuda_systems_protocol(path)
        return
    if schema_version == "phase5-open-model-candidates.v3":
        validate_phase5_candidate_manifest_v3(path)
        return
    if schema_version == 1:
        validate_manifest(path)
        return
    raise ValueError(f"{path}: unsupported manifest schema")


class _DispositionRule(NamedTuple):
    architecture_class: str | None
    revision_state: str
    license_review: str
    local_execution: str
    evidence_authority: str
    allowed_claims: tuple[str, ...]


class _GateContract(NamedTuple):
    required_evidence: tuple[str, ...]
    allowed_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]


_PHASE5_CANDIDATE_SCHEMA = "phase5-open-model-candidates.v1"
_PHASE5_READINESS_SCHEMA = "phase5-public-readiness.v1"
_PUBLIC_SARACURA_REPOSITORY = "https://github.com/felhen-ai/saracura"
_CANDIDATE_DISPOSITIONS = (
    "historical_baseline",
    "planned_acquisition",
    "external_control",
    "comparison_only",
)
_CANDIDATE_CLAIMS = (
    "historical_research_baseline",
    "candidate_for_evaluation",
    "external_product_control",
    "comparison_reference",
)
_ARCHITECTURE_CLASSES = frozenset(
    {
        "bi_encoder_projection_ranker",
        "option_marker_encoder",
        "constrained_causal_lm",
        "pointer_head_causal_lm",
        "native_remote",
        "mixture_of_experts_readout",
    }
)
_DISPOSITION_RULES = {
    "historical_baseline": _DispositionRule(
        "bi_encoder_projection_ranker",
        "private_verified",
        "not_applicable_private_baseline",
        "historical_only",
        "private_quality_observation",
        ("historical_research_baseline",),
    ),
    "planned_acquisition": _DispositionRule(
        None,
        "unpinned",
        "pending",
        "planned",
        "upstream_claims_only",
        ("candidate_for_evaluation",),
    ),
    "external_control": _DispositionRule(
        None,
        "external_service",
        "external_service",
        "not_applicable",
        "external_control_only",
        ("external_product_control",),
    ),
    "comparison_only": _DispositionRule(
        None,
        "comparison_reference",
        "pending",
        "not_supported",
        "comparison_context_only",
        ("comparison_reference",),
    ),
}
_READINESS_CLAIMS = (
    "research_only",
    "usable_developer_preview",
    "third_party_checkpoint_supported",
    "saracura_owned_checkpoint",
    "ptbr_measured",
    "bilingual_measured",
    "calibrated_for_declared_protocol",
    "comparative_claim_for_declared_task",
    "external_submission_ready",
    "production_ready",
    "automation_authorized",
)
_GATE_A_EVIDENCE = (
    "clean_machine_install",
    "pinned_license_reviewed_acquisition",
    "public_dev_quality_report",
    "local_systems_report",
    "offline_ci_contract",
    "public_ptbr_en_examples",
    "non_automation_safety_contract",
)
_GATE_B_EVIDENCE = (
    *_GATE_A_EVIDENCE,
    "saracura_owned_checkpoint",
    "sealed_ptbr_test",
    "sealed_bilingual_test",
    "disjoint_calibration_test",
    "clean_external_submission_bundle",
)
_GATE_A_ALLOWED = (
    "research_only",
    "usable_developer_preview",
    "third_party_checkpoint_supported",
)
_GATE_B_ALLOWED = (
    *_GATE_A_ALLOWED,
    "saracura_owned_checkpoint",
    "ptbr_measured",
    "bilingual_measured",
    "calibrated_for_declared_protocol",
    "comparative_claim_for_declared_task",
    "external_submission_ready",
)
_GATE_CONTRACTS = {
    "gate_a_developer_preview": _GateContract(
        _GATE_A_EVIDENCE,
        _GATE_A_ALLOWED,
        tuple(claim for claim in _READINESS_CLAIMS if claim not in _GATE_A_ALLOWED),
    ),
    "gate_b_saracura_checkpoint": _GateContract(
        _GATE_B_EVIDENCE,
        _GATE_B_ALLOWED,
        ("production_ready", "automation_authorized"),
    ),
}
_CANDIDATE_KEYS = frozenset(
    {
        "id",
        "display_name",
        "disposition",
        "architecture_class",
        "source_url",
        "declared_license",
        "license_review",
        "revision",
        "revision_state",
        "local_execution",
        "evidence_authority",
        "allowed_claims",
        "notes",
    }
)
_KEBAB_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DECLARED_LICENSE = re.compile(
    r"^(?:unknown|LicenseRef-[A-Za-z0-9.-]+|[A-Za-z0-9][A-Za-z0-9.+-]{0,63})$"
)
_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_NOTE_DIGEST = re.compile(r"[0-9a-fA-F]{32,}")
_PRIVATE_NOTE_MARKERS = (
    "felhen",
    "checkpoint.safetensors",
    "/users/",
    "/volumes/",
    "/home/",
    "gmail",
    "imap",
    "@",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate key: {key}")
        payload[key] = value
    return payload


def _load_closed_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest root must be an object")
    return payload


def _reject_packaged_phase5_copy(path: Path) -> None:
    parts = path.resolve().parts
    if "src" not in parts:
        return
    src_index = parts.index("src")
    if src_index + 1 < len(parts) and parts[src_index + 1] == "saracura":
        raise ValueError(f"{path}: phase 5 manifest must not live under src/saracura")


def _require_exact_keys(
    payload: dict[str, Any], expected: frozenset[str], path: Path, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{path}: {label} shape is not closed")


def _require_string_list(value: object, path: Path, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {label} must be a list of strings")
    return [item for item in value if isinstance(item, str)]


def _require_iso_date(value: object, path: Path) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{path}: reviewed_at must be an ISO calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{path}: reviewed_at must be an ISO calendar date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{path}: reviewed_at must be an ISO calendar date")


def _require_https_url(value: object, path: Path) -> str:
    if not isinstance(value, str) or any(character in value for character in " \t\r\n\\@"):
        raise ValueError(f"{path}: source_url must be an HTTPS public reference")
    if not value.startswith("https://"):
        raise ValueError(f"{path}: source_url must be an HTTPS public reference")
    host = value.removeprefix("https://").split("/", 1)[0]
    if not host or "." not in host or host.startswith(".") or ".." in host:
        raise ValueError(f"{path}: source_url must be an HTTPS public reference")
    return value


def _require_phase5a_revision(value: object, path: Path) -> None:
    # Phase 5A records no acquired revision. A later schema may allow only
    # immutable lowercase digests after a reviewed acquisition.
    if value is None:
        return
    if isinstance(value, str) and _IMMUTABLE_REVISION.fullmatch(value):
        raise ValueError(f"{path}: phase 5A revision must remain null")
    raise ValueError(f"{path}: revision is not null or an immutable lowercase hexadecimal digest")


def _require_public_note(value: object, path: Path) -> None:
    if not isinstance(value, str) or not value.endswith(".") or len(value) > 240:
        raise ValueError(f"{path}: notes must be one bounded public sentence")
    if any(character in value for character in "\n\r?!/\\~"):
        raise ValueError(f"{path}: notes must be one bounded public sentence")
    if ". " in value or "://" in value:
        raise ValueError(f"{path}: notes must be one bounded public sentence")
    lowered = value.casefold()
    if any(marker in lowered for marker in _PRIVATE_NOTE_MARKERS):
        raise ValueError(f"{path}: notes contain a private identifier or local path")
    if _NOTE_DIGEST.search(value):
        raise ValueError(f"{path}: notes contain a digest")


def _validate_candidate(candidate: object, path: Path, seen_ids: set[str]) -> None:
    if not isinstance(candidate, dict):
        raise ValueError(f"{path}: candidate shape is not closed")
    _require_exact_keys(candidate, _CANDIDATE_KEYS, path, "candidate")
    candidate_id = candidate["id"]
    if not isinstance(candidate_id, str) or _KEBAB_ID.fullmatch(candidate_id) is None:
        raise ValueError(f"{path}: candidate id must be stable lower-kebab")
    if candidate_id in seen_ids:
        raise ValueError(f"{path}: candidate id is duplicated")
    seen_ids.add(candidate_id)
    display_name = candidate["display_name"]
    if (
        not isinstance(display_name, str)
        or not display_name.strip()
        or display_name != display_name.strip()
    ):
        raise ValueError(f"{path}: display_name must be a non-empty label")
    disposition = candidate["disposition"]
    if disposition not in _DISPOSITION_RULES:
        raise ValueError(f"{path}: disposition is not allowed")
    rule = _DISPOSITION_RULES[str(disposition)]
    architecture_class = candidate["architecture_class"]
    if architecture_class not in _ARCHITECTURE_CLASSES:
        raise ValueError(f"{path}: architecture_class is not allowed")
    if rule.architecture_class is not None and architecture_class != rule.architecture_class:
        raise ValueError(f"{path}: disposition requires {rule.architecture_class}")
    source_url = _require_https_url(candidate["source_url"], path)
    if disposition == "historical_baseline" and source_url != _PUBLIC_SARACURA_REPOSITORY:
        raise ValueError(f"{path}: historical baseline must point at the public repository")
    declared_license = candidate["declared_license"]
    if (
        not isinstance(declared_license, str)
        or _DECLARED_LICENSE.fullmatch(declared_license) is None
        or "/" in declared_license
    ):
        raise ValueError(f"{path}: declared_license must be an SPDX identifier or unknown")
    if candidate["license_review"] != rule.license_review:
        raise ValueError(f"{path}: license_review does not match the disposition")
    _require_phase5a_revision(candidate["revision"], path)
    if candidate["revision_state"] != rule.revision_state:
        raise ValueError(f"{path}: revision_state does not match the disposition")
    if candidate["local_execution"] != rule.local_execution:
        raise ValueError(f"{path}: local_execution does not match the disposition")
    if candidate["evidence_authority"] != rule.evidence_authority:
        raise ValueError(f"{path}: evidence_authority does not match the disposition")
    allowed_claims = _require_string_list(candidate["allowed_claims"], path, "allowed_claims")
    if tuple(allowed_claims) != rule.allowed_claims:
        raise ValueError(f"{path}: allowed_claims do not match the disposition")
    _require_public_note(candidate["notes"], path)


_PHASE5_CANDIDATE_SCHEMA_V2 = "phase5-open-model-candidates.v2"
_PHASE5B_ACQUISITION_SCHEMA = "phase5-kev4b-acquisition.v1"

_CANDIDATE_DISPOSITIONS_V2 = (
    "historical_baseline",
    "planned_acquisition",
    "reviewed_acquisition",
    "blocked_upstream",
    "conditional",
    "continue",
    "reject_local",
    "external_control",
    "comparison_only",
)

_CANDIDATE_KEYS_V2 = _CANDIDATE_KEYS | {
    "source_revision",
    "base_model_id",
    "base_model_revision",
    "acquisition_descriptor_sha256",
}

_ACQUISITION_FIELDS = (
    "source_revision",
    "base_model_id",
    "base_model_revision",
    "acquisition_descriptor_sha256",
)

_REVIEWED_ACQUISITION_RULE = _DispositionRule(
    "pointer_head_causal_lm",
    "immutable_pinned",
    "reviewed_for_candidate_artifacts",
    "research_service",
    "reviewed_upstream_identity",
    ("candidate_for_evaluation",),
)


def _validate_candidate_v2(candidate: object, path: Path, seen_ids: set[str]) -> None:
    if not isinstance(candidate, dict):
        raise ValueError(f"{path}: candidate shape is not closed")
    _require_exact_keys(candidate, _CANDIDATE_KEYS_V2, path, "candidate")
    candidate_id = candidate["id"]
    if not isinstance(candidate_id, str) or _KEBAB_ID.fullmatch(candidate_id) is None:
        raise ValueError(f"{path}: candidate id must be stable lower-kebab")
    if candidate_id in seen_ids:
        raise ValueError(f"{path}: candidate id is duplicated")
    seen_ids.add(candidate_id)
    display_name = candidate["display_name"]
    if (
        not isinstance(display_name, str)
        or not display_name.strip()
        or display_name != display_name.strip()
    ):
        raise ValueError(f"{path}: display_name must be a non-empty label")
    disposition = candidate["disposition"]
    if disposition not in _CANDIDATE_DISPOSITIONS_V2:
        raise ValueError(f"{path}: disposition is not allowed")
    architecture_class = candidate["architecture_class"]
    if architecture_class not in _ARCHITECTURE_CLASSES:
        raise ValueError(f"{path}: architecture_class is not allowed")
    source_url = _require_https_url(candidate["source_url"], path)
    declared_license = candidate["declared_license"]
    if (
        not isinstance(declared_license, str)
        or _DECLARED_LICENSE.fullmatch(declared_license) is None
        or "/" in declared_license
    ):
        raise ValueError(f"{path}: declared_license must be an SPDX identifier or unknown")

    acquisition_fields = (
        candidate["source_revision"],
        candidate["base_model_id"],
        candidate["base_model_revision"],
        candidate["acquisition_descriptor_sha256"],
    )
    if disposition in _REQUIRES_ACQUISITION_DISPOSITIONS:
        rule = _REVIEWED_ACQUISITION_RULE
        if architecture_class != rule.architecture_class:
            raise ValueError(f"{path}: disposition requires {rule.architecture_class}")
        if declared_license != "Apache-2.0":
            raise ValueError(f"{path}: reviewed acquisition requires declared Apache-2.0")
        if not all(value is not None for value in acquisition_fields):
            raise ValueError(f"{path}: reviewed acquisition requires acquisition fields")
        for key, value in (
            ("source_revision", candidate["source_revision"]),
            ("base_model_revision", candidate["base_model_revision"]),
        ):
            if not isinstance(value, str) or _IMMUTABLE_REVISION.fullmatch(value) is None:
                raise ValueError(f"{path}: {key} must be an immutable lowercase hexadecimal digest")
        base_model_id = candidate["base_model_id"]
        if not isinstance(base_model_id, str) or "/" not in base_model_id:
            raise ValueError(f"{path}: base_model_id must be an owner/model hub id")
        descriptor = candidate["acquisition_descriptor_sha256"]
        if not isinstance(descriptor, str) or re.fullmatch(r"[0-9a-f]{64}", descriptor) is None:
            raise ValueError(f"{path}: acquisition_descriptor_sha256 must be a sha256 digest")
    else:
        rule = _DISPOSITION_RULES_V2[disposition]
        if rule.architecture_class is not None and architecture_class != rule.architecture_class:
            raise ValueError(f"{path}: disposition requires {rule.architecture_class}")
        if disposition == "historical_baseline" and source_url != _PUBLIC_SARACURA_REPOSITORY:
            raise ValueError(f"{path}: historical baseline must point at the public repository")
        if disposition in ("planned_acquisition", "blocked_upstream"):
            if any(value is not None for value in acquisition_fields):
                raise ValueError(f"{path}: pre-execution acquisition fields must remain null")
        else:
            if any(value is not None for value in acquisition_fields):
                raise ValueError(
                    f"{path}: non-acquisition disposition requires null acquisition fields"
                )
        if disposition == "blocked_upstream":
            if candidate["revision_state"] != "unpinned":
                raise ValueError(f"{path}: blocked_upstream must remain unpinned")
        else:
            if candidate["revision_state"] != rule.revision_state:
                raise ValueError(f"{path}: revision_state does not match the disposition")

    if candidate["license_review"] != rule.license_review:
        raise ValueError(f"{path}: license_review does not match the disposition")
    if disposition in _REQUIRES_ACQUISITION_DISPOSITIONS:
        revision = candidate["revision"]
        if not isinstance(revision, str) or _IMMUTABLE_REVISION.fullmatch(revision) is None:
            raise ValueError(f"{path}: reviewed acquisition requires an immutable model revision")
    else:
        _require_phase5a_revision(candidate["revision"], path)
    if candidate["local_execution"] != rule.local_execution:
        raise ValueError(f"{path}: local_execution does not match the disposition")
    if candidate["evidence_authority"] != rule.evidence_authority:
        raise ValueError(f"{path}: evidence_authority does not match the disposition")
    allowed_claims = _require_string_list(candidate["allowed_claims"], path, "allowed_claims")
    if tuple(allowed_claims) != rule.allowed_claims:
        raise ValueError(f"{path}: allowed_claims do not match the disposition")
    _require_public_note(candidate["notes"], path)


_DISPOSITION_RULES_V2 = {
    "historical_baseline": _DISPOSITION_RULES["historical_baseline"],
    "planned_acquisition": _DISPOSITION_RULES["planned_acquisition"],
    "external_control": _DISPOSITION_RULES["external_control"],
    "comparison_only": _DISPOSITION_RULES["comparison_only"],
}

_REQUIRES_ACQUISITION_DISPOSITIONS = (
    "reviewed_acquisition",
    "conditional",
    "continue",
    "reject_local",
)


def validate_phase5_candidate_manifest_v2(path: Path) -> None:
    _reject_packaged_phase5_copy(path)
    payload = _load_closed_json(path)
    _require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "reviewed_at",
                "supersedes_manifest_sha256",
                "allowed_dispositions",
                "candidate_claim_vocabulary",
                "candidates",
            }
        ),
        path,
        "candidate manifest",
    )
    if payload["schema_version"] != _PHASE5_CANDIDATE_SCHEMA_V2:
        raise ValueError(f"{path}: unsupported schema version")
    _require_iso_date(payload["reviewed_at"], path)
    supersedes = payload["supersedes_manifest_sha256"]
    v1_path = Path(__file__).parent / "manifests" / "phase5-open-model-candidates.v1.json"
    if (
        not isinstance(supersedes, str)
        or re.fullmatch(r"[0-9a-f]{64}", supersedes) is None
        or supersedes != hashlib.sha256(v1_path.read_bytes()).hexdigest()
    ):
        raise ValueError(f"{path}: supersedes_manifest_sha256 must bind the exact v1 bytes")
    if _require_string_list(payload["allowed_dispositions"], path, "allowed_dispositions") != list(
        _CANDIDATE_DISPOSITIONS_V2
    ):
        raise ValueError(f"{path}: allowed_dispositions are not the closed vocabulary")
    if _require_string_list(
        payload["candidate_claim_vocabulary"], path, "candidate_claim_vocabulary"
    ) != list(_CANDIDATE_CLAIMS):
        raise ValueError(f"{path}: candidate_claim_vocabulary is not the closed vocabulary")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: candidates must be a non-empty list")
    seen_ids: set[str] = set()
    for candidate in candidates:
        _validate_candidate_v2(candidate, path, seen_ids)


def validate_phase5b_acquisition_descriptor(path: Path) -> None:
    _reject_packaged_phase5_copy(path)
    payload = _load_closed_json(path)
    _require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "reviewed_at",
                "candidate_id",
                "licenses",
                "source",
                "checkpoint",
                "base_model",
                "runtime",
                "limits",
                "thresholds",
                "allowed_readiness_evidence",
            }
        ),
        path,
        "acquisition descriptor",
    )
    if payload["schema_version"] != _PHASE5B_ACQUISITION_SCHEMA:
        raise ValueError(f"{path}: unsupported schema version")
    _require_iso_date(payload["reviewed_at"], path)
    if payload["candidate_id"] != "kev-4b":
        raise ValueError(f"{path}: candidate_id must be kev-4b")
    if tuple(sorted(payload["allowed_readiness_evidence"])) != (
        "local_systems_report",
        "pinned_license_reviewed_acquisition",
    ):
        raise ValueError(f"{path}: allowed_readiness_evidence is not the frozen pair")
    if payload["runtime"]["python"] != "3.12":
        raise ValueError(f"{path}: runtime python must be 3.12")
    if payload["runtime"]["backend"] != "mlx" or payload["runtime"]["dtype"] != "bf16":
        raise ValueError(f"{path}: runtime backend/dtype must be mlx bf16")
    if payload["runtime"]["loopback_host"] != "127.0.0.1":
        raise ValueError(f"{path}: runtime host must be fixed loopback")


def validate_phase5_candidate_manifest(path: Path) -> None:
    _reject_packaged_phase5_copy(path)
    payload = _load_closed_json(path)
    _require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "reviewed_at",
                "allowed_dispositions",
                "candidate_claim_vocabulary",
                "candidates",
            }
        ),
        path,
        "candidate manifest",
    )
    if payload["schema_version"] != _PHASE5_CANDIDATE_SCHEMA:
        raise ValueError(f"{path}: unsupported schema version")
    _require_iso_date(payload["reviewed_at"], path)
    if _require_string_list(payload["allowed_dispositions"], path, "allowed_dispositions") != list(
        _CANDIDATE_DISPOSITIONS
    ):
        raise ValueError(f"{path}: allowed_dispositions are not the closed vocabulary")
    if _require_string_list(
        payload["candidate_claim_vocabulary"], path, "candidate_claim_vocabulary"
    ) != list(_CANDIDATE_CLAIMS):
        raise ValueError(f"{path}: candidate_claim_vocabulary is not the closed vocabulary")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: candidates must be a non-empty list")
    seen_ids: set[str] = set()
    for candidate in candidates:
        _validate_candidate(candidate, path, seen_ids)


def _require_claim_partition(allowed: list[str], forbidden: list[str], path: Path) -> None:
    if len(allowed) != len(set(allowed)) or len(forbidden) != len(set(forbidden)):
        raise ValueError(f"{path}: gate claims contain duplicates")
    if set(allowed) & set(forbidden):
        raise ValueError(f"{path}: gate claims overlap")
    if set(allowed) | set(forbidden) != set(_READINESS_CLAIMS):
        raise ValueError(f"{path}: gate claims omit or invent vocabulary")
    if "production_ready" in allowed or "automation_authorized" in allowed:
        raise ValueError(f"{path}: production or automation claims are forbidden")


def _require_gate_semantics(
    gate_id: str, evidence: list[str], allowed: list[str], path: Path
) -> None:
    if (
        "calibrated_for_declared_protocol" in allowed
        and "disjoint_calibration_test" not in evidence
    ):
        raise ValueError(f"{path}: calibration claim requires disjoint calibration evidence")
    if "comparative_claim_for_declared_task" in allowed and (
        "sealed_ptbr_test" not in evidence or "sealed_bilingual_test" not in evidence
    ):
        raise ValueError(f"{path}: comparative claim requires sealed test evidence")
    if (
        gate_id == "gate_b_saracura_checkpoint"
        and tuple(evidence[: len(_GATE_A_EVIDENCE)]) != _GATE_A_EVIDENCE
    ):
        raise ValueError(f"{path}: gate B must retain gate A evidence in order")


def validate_phase5_readiness_manifest(path: Path) -> None:
    _reject_packaged_phase5_copy(path)
    payload = _load_closed_json(path)
    _require_exact_keys(
        payload,
        frozenset({"schema_version", "claim_vocabulary", "gates"}),
        path,
        "readiness manifest",
    )
    if payload["schema_version"] != _PHASE5_READINESS_SCHEMA:
        raise ValueError(f"{path}: unsupported schema version")
    if _require_string_list(payload["claim_vocabulary"], path, "claim_vocabulary") != list(
        _READINESS_CLAIMS
    ):
        raise ValueError(f"{path}: claim_vocabulary is not the closed vocabulary")
    gates = payload["gates"]
    if not isinstance(gates, list) or [
        gate.get("id") if isinstance(gate, dict) else None for gate in gates
    ] != [
        "gate_a_developer_preview",
        "gate_b_saracura_checkpoint",
    ]:
        raise ValueError(f"{path}: gates must be gate A then gate B")
    parsed: list[tuple[str, list[str], list[str]]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            raise ValueError(f"{path}: gate shape is not closed")
        _require_exact_keys(
            gate,
            frozenset({"id", "status", "required_evidence", "allowed_claims", "forbidden_claims"}),
            path,
            "gate",
        )
        gate_id = str(gate["id"])
        if gate["status"] != "not_met":
            raise ValueError(f"{path}: gate status must be not_met")
        evidence = _require_string_list(gate["required_evidence"], path, "required_evidence")
        allowed = _require_string_list(gate["allowed_claims"], path, "allowed_claims")
        forbidden = _require_string_list(gate["forbidden_claims"], path, "forbidden_claims")
        contract = _GATE_CONTRACTS[gate_id]
        if tuple(evidence) != contract.required_evidence:
            raise ValueError(f"{path}: required_evidence does not match the gate")
        if (
            tuple(allowed) != contract.allowed_claims
            or tuple(forbidden) != contract.forbidden_claims
        ):
            raise ValueError(f"{path}: gate claims do not match the closed contract")
        _require_claim_partition(allowed, forbidden, path)
        _require_gate_semantics(gate_id, evidence, allowed, path)
        parsed.append((gate_id, evidence, allowed))
    gate_a_allowed = set(parsed[0][2])
    gate_b_allowed = set(parsed[1][2])
    if not gate_a_allowed < gate_b_allowed:
        raise ValueError(f"{path}: gate B claims must be a strict superset of gate A")


_PHASE5_MANAGED_CUDA_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "acquisition_descriptor_sha256",
        "runtime",
        "command",
        "environment",
        "source_revision",
        "checkpoint_revision",
        "base_revision",
        "base_model_id",
    },
)

_PHASE5_MANAGED_CUDA_SYSTEMS_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "acquisition_descriptor_sha256",
        "managed_runtime_descriptor_sha256",
        "fixture_digest",
        "wire_contract",
        "workload_counts",
        "protocol_counts",
        "probe_sizes",
        "sampling",
        "thresholds",
        "report_schema",
    },
)

_PHASE5_OPEN_MODEL_CANDIDATES_V3_FIELDS = frozenset(
    {
        "schema_version",
        "reviewed_at",
        "supersedes_manifest_sha256",
        "allowed_dispositions",
        "candidate_claim_vocabulary",
        "candidates",
        "systems_report_path",
        "systems_report_digest",
        "protocol_path",
        "protocol_digest",
        "readiness_manifest_path",
        "readiness_manifest_sha256",
    },
)

_PHASE5_CANDIDATE_V3_CLAIMS = (
    "historical_research_baseline",
    "candidate_for_evaluation",
    "external_product_control",
    "comparison_reference",
)

_PHASE5_CANDIDATE_V3_ENTRY_FIELDS = frozenset(
    {
        "id",
        "display_name",
        "disposition",
        "architecture_class",
        "source_url",
        "declared_license",
        "license_review",
        "revision",
        "revision_state",
        "local_execution",
        "evidence_authority",
        "allowed_claims",
        "source_revision",
        "base_model_id",
        "base_model_revision",
        "acquisition_descriptor_sha256",
        "notes",
        "mitigation_markers",
    }
)


def validate_phase5_managed_cuda_runtime(path: Path) -> None:
    _reject_packaged_phase5_copy(path)
    payload = _load_closed_json(path)
    _require_exact_keys(
        payload, _PHASE5_MANAGED_CUDA_RUNTIME_FIELDS, path, "managed-runtime descriptor"
    )
    if payload["schema_version"] != "phase5-kev4b-managed-cuda.v1":
        raise ValueError(f"{path}: unsupported schema version")
    if payload["candidate_id"] != "kev-4b":
        raise ValueError(f"{path}: candidate_id must be kev-4b")
    runtime = payload["runtime"]
    _require_exact_keys(
        runtime,
        frozenset({"backend", "dtype", "host", "direct_upstream_port"}),
        path,
        "managed-runtime runtime",
    )
    if runtime["backend"] != "torch":
        raise ValueError(f"{path}: runtime backend must be torch")
    if runtime["dtype"] != "bf16":
        raise ValueError(f"{path}: runtime dtype must be bf16")
    if runtime["host"] != "127.0.0.1":
        raise ValueError(f"{path}: runtime host must be 127.0.0.1")
    if type(runtime["direct_upstream_port"]) is not int or runtime["direct_upstream_port"] != 8182:
        raise ValueError(f"{path}: direct_upstream_port must be the fixed port 8182")
    expected_command = {
        "module": "kev.serve",
        "python_basename": "python",
        "run_flag": "--run",
        "fallback_flag": "--fallback",
        "host_flag": "--host",
        "port_flag": "--port",
    }
    _require_exact_keys(
        pathayload := payload["command"],
        frozenset(expected_command),
        path,
        "managed-runtime command",
    )
    if pathayload != expected_command:
        raise ValueError(f"{path}: command contract differs from pinned kev.serve invocation")
    expected_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "KEV_BACKEND": "torch",
        "KEV_DTYPE": "bf16",
        "KEV_MERGE": "0",
        "KEV_FUSED": "0",
        "KEV_CUDA_GRAPHS": "0",
        "KEV_DATE_FACTS": "0",
        "KEV_LORA_SCALE": "1",
        "KEV_PREFIX_CACHE": "4",
        "KEV_PREFIX_MIN_TOKENS": "0",
        "KEV_PREFIX_MAX_TOKENS": "65536",
        "required_nonempty": ["KEV_API_KEY"],
    }
    _require_exact_keys(
        payload["environment"], frozenset(expected_environment), path, "managed-runtime environment"
    )
    if payload["environment"] != expected_environment:
        raise ValueError(f"{path}: environment contract differs from frozen values")
    if not re.fullmatch(r"[0-9a-f]{64}", payload["acquisition_descriptor_sha256"]):
        raise ValueError(f"{path}: acquisition_descriptor_sha256 must be a sha256 digest")
    for field in ("source_revision", "checkpoint_revision", "base_revision"):
        if not re.fullmatch(r"^[0-9a-f]{40}$", payload[field]):
            raise ValueError(f"{path}: {field} must be a git revision")
    acquisition_path = Path(__file__).parent / "manifests/phase5-kev4b-acquisition.v1.json"
    acquisition = _load_closed_json(acquisition_path)
    expected = {
        "acquisition_descriptor_sha256": hashlib.sha256(acquisition_path.read_bytes()).hexdigest(),
        "source_revision": acquisition["source"]["revision"],
        "checkpoint_revision": acquisition["checkpoint"]["revision"],
        "base_revision": acquisition["base_model"]["revision"],
        "base_model_id": acquisition["base_model"]["model_id"],
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise ValueError(
            f"{path}: managed runtime does not bind the immutable acquisition descriptor"
        )


def validate_phase5_managed_cuda_systems_protocol(path: Path) -> None:
    _reject_packaged_phase5_copy(path)
    payload = _load_closed_json(path)
    _require_exact_keys(payload, _PHASE5_MANAGED_CUDA_SYSTEMS_FIELDS, path, "systems protocol")
    if payload["schema_version"] != "phase5-managed-cuda-systems.v1":
        raise ValueError(f"{path}: unsupported schema version")
    if payload["candidate_id"] != "kev-4b":
        raise ValueError(f"{path}: candidate_id must be kev-4b")
    if payload["report_schema"] != "phase5b-managed-cuda-systems-report.v1":
        raise ValueError(f"{path}: report_schema is not the frozen public report schema")
    expected_wire = {
        "model": "kev-latest",
        "questions_shape": "mapping_by_question_id",
        "question_type": "choice",
        "instructions_field": "instructions",
        "criteria_shape": "mapping_by_criterion_id",
        "success_response_keys": ["model", "answers", "usage", "latency_ms"],
        "answer_keys": ["choice", "probabilities"],
        "probability_simplex_abs_tolerance": 0.00001,
        "input_token_counter": "input_tokens",
    }
    _require_exact_keys(payload["wire_contract"], frozenset(expected_wire), path, "wire contract")
    if payload["wire_contract"] != expected_wire:
        raise ValueError(f"{path}: source-shaped wire contract drifted")
    counts = {
        "warmups_per_cell": 3,
        "new_state_per_cell": 20,
        "cached_state_per_cell": 20,
        "matrix_cell_count": 12,
        "measured_request_count": 240,
    }
    _require_exact_keys(payload["protocol_counts"], frozenset(counts), path, "protocol counts")
    if payload["protocol_counts"] != counts:
        raise ValueError(f"{path}: request matrix counts drifted")
    cadence = {
        "procfs_interval_ms": 250,
        "procfs_max_gap_ms": 1000,
        "nvidia_smi_interval_ms": 1000,
        "nvidia_smi_max_gap_ms": 3000,
        "diagnostic_state_window_separate": True,
    }
    _require_exact_keys(payload["sampling"], frozenset(cadence), path, "sampling protocol")
    if payload["sampling"] != cadence:
        raise ValueError(f"{path}: sampling cadence drifted")
    if not re.fullmatch(r"[0-9a-f]{64}", payload["acquisition_descriptor_sha256"]):
        raise ValueError(f"{path}: acquisition_descriptor_sha256 must be a sha256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", payload["fixture_digest"]):
        raise ValueError(f"{path}: fixture_digest must be a sha256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", payload["managed_runtime_descriptor_sha256"]):
        raise ValueError(f"{path}: managed_runtime_descriptor_sha256 must be a sha256 digest")
    acquisition_path = Path(__file__).parent / "manifests/phase5-kev4b-acquisition.v1.json"
    acquisition = _load_closed_json(acquisition_path)
    if (
        payload["acquisition_descriptor_sha256"]
        != hashlib.sha256(acquisition_path.read_bytes()).hexdigest()
    ):
        raise ValueError(
            f"{path}: protocol acquisition digest does not match exact descriptor bytes"
        )
    runtime_path = Path(__file__).parent / "manifests/phase5-kev4b-managed-cuda.v1.json"
    if (
        payload["managed_runtime_descriptor_sha256"]
        != hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    ):
        raise ValueError(f"{path}: protocol runtime digest does not match exact descriptor bytes")
    from benchmarks.phase5_candidate.fixture import matrix

    fixture_digest = hashlib.sha256(
        json.dumps(matrix(), sort_keys=True, default=str).encode()
    ).hexdigest()
    if payload["fixture_digest"] != fixture_digest:
        raise ValueError(f"{path}: protocol fixture digest does not match exact fixture semantics")
    for name, value in (
        (
            "workload_counts",
            {"pt-BR": {"q1": 20, "q10": 20, "q50": 20}, "en": {"q1": 20, "q10": 20, "q50": 20}},
        ),
        (
            "probe_sizes",
            {
                "oversize_branch_instructions": 100000,
                "oversize_state_items": 100000,
                "valid_control_questions": 1,
            },
        ),
    ):
        if payload[name] != value:
            raise ValueError(f"{path}: {name} does not match the frozen protocol")
    _require_exact_keys(
        payload["workload_counts"], frozenset({"pt-BR", "en"}), path, "workload counts"
    )
    _require_exact_keys(
        payload["probe_sizes"],
        frozenset(
            {"oversize_branch_instructions", "oversize_state_items", "valid_control_questions"}
        ),
        path,
        "probe sizes",
    )
    thresholds = payload["thresholds"]
    expected_thresholds = {
        "q1_p95_ms",
        "q10_p95_ms",
        "q50_p95_ms",
        "max_request_ms",
        "repeat_probability_delta",
        "together_separate_delta",
        "cold_load_seconds",
        "peak_rss_gib",
        "peak_gpu_memory_gib",
        "device_min_ram_gib",
        "candidate_swap_gib",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected_thresholds:
        raise ValueError(f"{path}: thresholds shape mismatch")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in thresholds.values()
    ):
        raise ValueError(f"{path}: thresholds must be finite numbers")
    if thresholds["q1_p95_ms"] > 5000:
        raise ValueError(f"{path}: q1_p95_ms threshold exceeds 5 seconds")
    if thresholds["q10_p95_ms"] > 15000:
        raise ValueError(f"{path}: q10_p95_ms threshold exceeds 15 seconds")
    if thresholds["q50_p95_ms"] > 45000:
        raise ValueError(f"{path}: q50_p95_ms threshold exceeds 45 seconds")
    if thresholds["peak_rss_gib"] > 22:
        raise ValueError(f"{path}: peak_rss_gib exceeds 22 GiB")
    if thresholds["peak_gpu_memory_gib"] > 16:
        raise ValueError(f"{path}: peak_gpu_memory_gib exceeds 16 GiB")
    if thresholds["device_min_ram_gib"] < 24:
        raise ValueError(f"{path}: device_min_ram_gib must be at least 24 GiB")
    if thresholds["candidate_swap_gib"] > 1:
        raise ValueError(f"{path}: candidate_swap_gib exceeds 1 GiB")
    shared = acquisition["thresholds"]
    equal_or_stricter = {
        "q1_p95_ms": shared["p95_latency_seconds_q1"] * 1000,
        "q10_p95_ms": shared["p95_latency_seconds_q10"] * 1000,
        "q50_p95_ms": shared["p95_latency_seconds_q50"] * 1000,
        "max_request_ms": shared["max_latency_seconds"] * 1000,
        "repeat_probability_delta": shared["max_repeat_probability_delta"],
        "together_separate_delta": shared["max_together_separate_probability_delta"],
        "cold_load_seconds": acquisition["limits"]["startup_timeout_seconds"],
        "peak_rss_gib": shared["max_peak_rss_gib"],
        "candidate_swap_gib": shared["max_swap_delta_gib"],
        "peak_gpu_memory_gib": thresholds["peak_gpu_memory_gib"],
        "device_min_ram_gib": thresholds["device_min_ram_gib"],
    }
    for key, ceiling in equal_or_stricter.items():
        if thresholds[key] > ceiling:
            raise ValueError(f"{path}: {key} is looser than acquisition ceiling")


def validate_phase5_candidate_manifest_v3(path: Path, repository: Path | None = None) -> None:
    _reject_packaged_phase5_copy(path)
    payload = _load_closed_json(path)
    _require_exact_keys(
        payload, _PHASE5_OPEN_MODEL_CANDIDATES_V3_FIELDS, path, "candidate manifest v3"
    )
    if payload["schema_version"] != "phase5-open-model-candidates.v3":
        raise ValueError(f"{path}: unsupported schema version")
    _require_iso_date(payload["reviewed_at"], path)
    supersedes = payload["supersedes_manifest_sha256"]
    v2_path = Path(__file__).parent / "manifests" / "phase5-open-model-candidates.v2.json"
    if (
        not isinstance(supersedes, str)
        or re.fullmatch(r"[0-9a-f]{64}", supersedes) is None
        or supersedes != hashlib.sha256(v2_path.read_bytes()).hexdigest()
    ):
        raise ValueError(f"{path}: supersedes_manifest_sha256 must bind the exact v2 bytes")
    if _require_string_list(payload["allowed_dispositions"], path, "allowed_dispositions") != [
        "historical_baseline",
        "planned_acquisition",
        "reviewed_acquisition",
        "blocked_upstream",
        "conditional",
        "continue",
        "reject_local",
        "external_control",
        "comparison_only",
        "blocked_evidence",
    ]:
        raise ValueError(f"{path}: allowed_dispositions must be the v3 closed vocabulary")
    if _require_string_list(
        payload["candidate_claim_vocabulary"], path, "candidate_claim_vocabulary"
    ) != list(_PHASE5_CANDIDATE_V3_CLAIMS):
        raise ValueError(f"{path}: candidate_claim_vocabulary mismatch")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: candidates must be a non-empty list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"{path}: each candidate must be an object")
        _require_exact_keys(candidate, _PHASE5_CANDIDATE_V3_ENTRY_FIELDS, path, "v3 candidate")
        if candidate["disposition"] not in payload["allowed_dispositions"]:
            raise ValueError(f"{path}: candidate disposition is outside the closed vocabulary")
        claims = _require_string_list(candidate["allowed_claims"], path, "candidate allowed_claims")
        if set(claims) - set(_PHASE5_CANDIDATE_V3_CLAIMS):
            raise ValueError(f"{path}: candidate claims are outside the closed vocabulary")
        markers = _require_string_list(candidate["mitigation_markers"], path, "mitigation_markers")
        if markers not in ([], ["state_size_guard_required"]):
            raise ValueError(f"{path}: unknown mitigation marker")
    _validate_phase5_v3_bindings(payload, path, repository or Path(__file__).resolve().parents[1])


def _validate_phase5_v3_bindings(payload: dict[str, Any], path: Path, repository: Path) -> None:
    canonical_paths = {
        "systems_report_path": "benchmarks/results/phase5b-kev4b-managed-cuda-systems.json",
        "protocol_path": "benchmarks/manifests/phase5-managed-cuda-systems.v1.json",
        "readiness_manifest_path": "benchmarks/manifests/phase5-public-readiness.v1.json",
    }
    fields = (
        ("systems_report_path", "systems_report_digest"),
        ("protocol_path", "protocol_digest"),
        ("readiness_manifest_path", "readiness_manifest_sha256"),
    )
    references: dict[str, Path] = {}
    for path_key, digest_key in fields:
        relative = payload[path_key]
        digest = payload[digest_key]
        if relative is None and digest is None:
            continue
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"{path}: {path_key} and {digest_key} must be both null or both set")
        if relative != canonical_paths[path_key]:
            raise ValueError(f"{path}: {path_key} does not name the canonical artifact")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}: {digest_key} must be a SHA-256 digest")
        candidate_path = (repository / relative).resolve()
        if repository.resolve() not in candidate_path.parents:
            raise ValueError(f"{path}: {path_key} must resolve below the repository")
        if not candidate_path.is_file():
            raise ValueError(f"{path}: referenced {path_key} does not exist")
        if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path}: {digest_key} does not match exact referenced bytes")
        references[path_key] = candidate_path
    if not references:
        _validate_phase5_v3_schema_only_copy(payload, path)
        return
    if len(references) != 3:
        raise ValueError(f"{path}: report, protocol and readiness bindings are all required")

    report_path = references["systems_report_path"]
    report = _load_closed_json(report_path)
    from benchmarks.phase5_managed_cuda.models import PublicFailureReport, PublicSystemsReport

    schema_version = report.get("schema_version")
    if schema_version == "phase5b-managed-cuda-systems-report.v1":
        parsed_report: PublicFailureReport | PublicSystemsReport = (
            PublicSystemsReport.model_validate(report)
        )
    elif schema_version == "phase5b-managed-cuda-failure-report.v1":
        parsed_report = PublicFailureReport.model_validate(report)
    else:
        raise ValueError(f"{path}: bound report schema is not recognized")
    protocol_path = references["protocol_path"]
    validate_phase5_managed_cuda_systems_protocol(protocol_path)
    protocol = _load_closed_json(protocol_path)
    if isinstance(parsed_report, PublicSystemsReport):
        expected_report_bindings = {
            "protocol_digest": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "acquisition_descriptor_sha256": protocol["acquisition_descriptor_sha256"],
            "managed_runtime_descriptor_sha256": protocol["managed_runtime_descriptor_sha256"],
            "fixture_digest": protocol["fixture_digest"],
        }
        if any(
            getattr(parsed_report, field) != expected
            for field, expected in expected_report_bindings.items()
        ):
            raise ValueError(f"{path}: report provenance differs from the exact protocol bytes")
    elif not (
        parsed_report.classification == "reject_local"
        and parsed_report.disposition == "reject_local"
        and parsed_report.evidence_code == "candidate_invalid_response"
        and not parsed_report.operator_preempted
        and parsed_report.protected_workload_restored
    ):
        raise ValueError(f"{path}: failure report is not eligible for a v3 successor")
    readiness_path = references["readiness_manifest_path"]
    validate_phase5_readiness_manifest(readiness_path)
    readiness = _load_closed_json(readiness_path)
    gate_a = readiness["gates"][0]
    if gate_a["id"] != "gate_a_developer_preview" or gate_a["status"] != "not_met":
        raise ValueError(f"{path}: referenced readiness Gate A must remain not_met")

    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: candidates must be a non-empty list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"{path}: each candidate must be an object")
        _require_exact_keys(candidate, _PHASE5_CANDIDATE_V3_ENTRY_FIELDS, path, "v3 candidate")
        if candidate["disposition"] not in payload["allowed_dispositions"]:
            raise ValueError(f"{path}: candidate disposition is outside the closed vocabulary")
        if _require_string_list(candidate["allowed_claims"], path, "candidate allowed_claims"):
            unknown_claims = set(candidate["allowed_claims"]) - set(_PHASE5_CANDIDATE_V3_CLAIMS)
            if unknown_claims:
                raise ValueError(f"{path}: candidate claims are outside the closed vocabulary")
        if _require_string_list(
            candidate["mitigation_markers"], path, "mitigation_markers"
        ) and candidate["mitigation_markers"] != ["state_size_guard_required"]:
            raise ValueError(f"{path}: unknown mitigation marker")
    kev_matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("id") == "kev-4b"
    ]
    if len(kev_matches) != 1:
        raise ValueError(f"{path}: exactly one Kev-4B candidate is required")
    candidate = kev_matches[0]
    v2_path = Path(__file__).parent / "manifests" / "phase5-open-model-candidates.v2.json"
    v2_candidates = _load_closed_json(v2_path)["candidates"]
    predecessor_by_id = {entry["id"]: entry for entry in v2_candidates}
    successor_by_id = {entry["id"]: entry for entry in candidates}
    if len(successor_by_id) != len(candidates) or set(successor_by_id) != set(predecessor_by_id):
        raise ValueError(f"{path}: bound v3 candidate identities must match the v2 predecessor")
    for candidate_id, predecessor in predecessor_by_id.items():
        successor = successor_by_id[candidate_id]
        expected = {**predecessor, "mitigation_markers": []}
        if candidate_id != "kev-4b":
            if successor != expected:
                raise ValueError(
                    f"{path}: non-Kev candidate {candidate_id} must copy v2 semantics exactly"
                )
            continue
        mutable_promotion_fields = {"disposition", "allowed_claims", "mitigation_markers"}
        if set(successor) != set(expected) or any(
            successor[field] != value
            for field, value in expected.items()
            if field not in mutable_promotion_fields
        ):
            raise ValueError(f"{path}: Kev immutable identity and metadata must match v2")
    if candidate.get("disposition") != parsed_report.disposition:
        raise ValueError(f"{path}: candidate disposition must equal the bound report")
    claims = candidate.get("allowed_claims")
    markers = candidate.get("mitigation_markers")
    if not isinstance(claims, list) or not isinstance(markers, list):
        raise ValueError(f"{path}: candidate claims and mitigation_markers must be lists")
    if parsed_report.disposition == "conditional":
        if claims != ["candidate_for_evaluation"]:
            raise ValueError(f"{path}: conditional permits only candidate_for_evaluation")
        if markers != ["state_size_guard_required"] or not parsed_report.state_size_guard_required:
            raise ValueError(f"{path}: conditional requires the exact state-size guard marker")
    elif parsed_report.disposition == "reject_local":
        if claims or "candidate_for_evaluation" in claims or markers:
            raise ValueError(f"{path}: reject_local requires empty claims and mitigation markers")
    elif parsed_report.disposition == "blocked_evidence":
        raise ValueError(f"{path}: blocked_evidence must not publish a v3 successor")
    elif parsed_report.disposition == "continue":
        raise ValueError(f"{path}: v1 managed CUDA protocol cannot emit continue")


def _validate_phase5_v3_schema_only_copy(payload: dict[str, Any], path: Path) -> None:
    """Allow a schema-only v3 only when it preserves v2 candidate semantics."""
    v2_path = Path(__file__).parent / "manifests" / "phase5-open-model-candidates.v2.json"
    v2 = _load_closed_json(v2_path)
    expected_dispositions = [*v2["allowed_dispositions"], "blocked_evidence"]
    if payload["allowed_dispositions"] != expected_dispositions:
        raise ValueError(f"{path}: unbound v3 cannot change the v2 disposition vocabulary")
    if payload["candidate_claim_vocabulary"] != v2["candidate_claim_vocabulary"]:
        raise ValueError(f"{path}: unbound v3 cannot change the v2 claim vocabulary")
    expected_candidates = [
        {**candidate, "mitigation_markers": []} for candidate in v2["candidates"]
    ]
    if payload["candidates"] != expected_candidates:
        raise ValueError(f"{path}: unbound v3 must be an exact semantic copy of v2")


def validate_committed_phase5b_report(repository: Path | None = None) -> None:
    """Validate the designated P2 report whenever it is committed in a checkout."""
    root = repository or Path(__file__).resolve().parents[1]
    report_path = root / "benchmarks/results/phase5b-kev4b-managed-cuda-systems.json"
    if not report_path.exists():
        return
    _load_closed_json(report_path)
    from benchmarks.phase5_managed_cuda.models import PublicFailureReport, PublicSystemsReport

    payload = _load_closed_json(report_path)
    if payload.get("schema_version") == "phase5b-managed-cuda-failure-report.v1":
        report: PublicFailureReport | PublicSystemsReport = PublicFailureReport.model_validate(
            payload
        )
    else:
        report = PublicSystemsReport.model_validate(payload)
    from benchmarks.phase5_managed_cuda.cli import _assert_public_sanitized

    _assert_public_sanitized(report, ())


def validate_committed_phase5d_reports(repository: Path | None = None) -> None:
    """Bind every committed Phase 5D report to this checkout."""
    root = repository or Path(__file__).resolve().parents[1]
    from benchmarks.ptbr_native.runner import verify_report

    report_paths = (
        root / "benchmarks/results/phase5d-ptbr-faq-bacen-julia-cpu.json",
        root / "benchmarks/results/phase5d-ptbr-faq-bacen-saracura-universal-cpu.json",
        root / "benchmarks/results/phase5d1-ptbr-faq-bacen-julia-cyclic-mean-cpu.json",
    )
    for path in report_paths:
        verify_report(path)


def main() -> int:
    manifest_directory = Path(__file__).parent / "manifests"
    manifests = sorted(manifest_directory.glob("*.json"))
    if not manifests:
        raise ValueError("no first-party manifests found")
    for path in manifests:
        validate_routed_manifest(path)
    validate_committed_phase5b_report()
    validate_committed_phase5d_reports()
    print(f"validated {len(manifests)} routed manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
