"""Validate the narrow provenance contract for first-cycle self-authored fixtures."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.data_policy_registry import load_registry as load_data_policy_registry
from benchmarks.data_policy_registry import load_registry_v2 as load_data_policy_registry_v2
from benchmarks.data_policy_registry import load_registry_v3 as load_data_policy_registry_v3
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
    if schema_version == "phase5-public-readiness.v1":
        validate_phase5_readiness_manifest(path)
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


def main() -> int:
    manifest_directory = Path(__file__).parent / "manifests"
    manifests = sorted(manifest_directory.glob("*.json"))
    if not manifests:
        raise ValueError("no first-party manifests found")
    for path in manifests:
        validate_routed_manifest(path)
    print(f"validated {len(manifests)} routed manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
