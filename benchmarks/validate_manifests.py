"""Validate the narrow provenance contract for first-cycle self-authored fixtures."""

from __future__ import annotations

import json
from pathlib import Path

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


def main() -> int:
    manifest_directory = Path(__file__).parent / "manifests"
    manifests = sorted(manifest_directory.glob("*.json"))
    if not manifests:
        raise ValueError("no first-party manifests found")
    for path in manifests:
        validate_manifest(path)
    print(f"validated {len(manifests)} self-authored manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
