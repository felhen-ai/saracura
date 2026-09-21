"""Validate the narrow provenance contract for first-cycle self-authored fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.data_policy_registry import load_registry as load_data_policy_registry
from benchmarks.encoder_registry import load_registry as load_encoder_registry
from benchmarks.first_party_gate import validate_protocol_bytes
from benchmarks.synthetic_research import validate_synthetic_policy

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
    if schema_version == "support-routing-protocol.v1":
        validate_protocol_bytes(raw)
        return
    if schema_version == "synthetic-research-policy.v1":
        validate_synthetic_policy(path)
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
