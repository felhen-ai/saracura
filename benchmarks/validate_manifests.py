"""Validate the narrow provenance contract for first-cycle self-authored fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.data_policy_registry import load_registry as load_data_policy_registry
from benchmarks.data_policy_registry import load_registry_v2 as load_data_policy_registry_v2
from benchmarks.encoder_registry import load_registry as load_encoder_registry
from benchmarks.first_party_gate import validate_protocol_bytes
from benchmarks.saracura_universal_pilot import (
    RECOVERY_POLICY_V2_PATH,
    RECOVERY_POLICY_V3_PATH,
    RECOVERY_POLICY_V4_PATH,
    RECOVERY_POLICY_V5_PATH,
    RECOVERY_POLICY_V6_PATH,
    RECOVERY_POLICY_V7_PATH,
    RECOVERY_POLICY_V8_PATH,
    validate_pilot_policy,
    validate_pilot_recovery_policy,
    validate_spend_baseline,
)
from benchmarks.saracura_universal_policy import validate_phase4e_policy
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
    if schema_version == "phase4e-protocol-pilot-policy.v1":
        validate_pilot_policy(path)
        return
    if schema_version == "phase4e-protocol-pilot-recovery.v9":
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
    if schema_version == 1:
        validate_manifest(path)
        return
    raise ValueError(f"{path}: unsupported manifest schema")


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
