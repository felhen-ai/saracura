import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.validate_manifests import validate_manifest, validate_routed_manifest
from saracura.runtime.engine import FIXTURE_SPLIT_MANIFEST_SHA256


def test_first_party_fixture_manifest_passes_provenance_gate() -> None:
    path = Path(__file__).parents[1] / "benchmarks/manifests/ptbr-fixture-v1.json"
    validate_manifest(path)


def test_calibration_provenance_hash_matches_manifest_bytes() -> None:
    root = Path(__file__).parents[1]
    manifest = root / "benchmarks/manifests/ptbr-fixture-v1.json"
    artifact = json.loads(
        (root / "examples/ptbr-support-calibration.json").read_text(encoding="utf-8")
    )
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()

    assert actual == FIXTURE_SPLIT_MANIFEST_SHA256
    assert artifact["split_manifest_sha256"] == actual


def test_external_data_is_rejected_in_phase_one(tmp_path: Path) -> None:
    path = tmp_path / "external.json"
    source = Path(__file__).parents[1] / "benchmarks/manifests/ptbr-fixture-v1.json"
    payload = source.read_text(encoding="utf-8").replace(
        '"contains_external_data": false',
        '"contains_external_data": true',
    )
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="self-authored fixtures"):
        validate_manifest(path)


def test_phase4c_manifests_route_through_closed_validators() -> None:
    root = Path(__file__).parents[1] / "benchmarks/manifests"
    validate_routed_manifest(root / "decision-backend-candidates.v1.json")
    validate_routed_manifest(root / "universal-bakeoff-plan.v1.json")


def test_phase4e_recovery_manifests_route_through_closed_validators(tmp_path: Path) -> None:
    root = Path(__file__).parents[1] / "benchmarks/manifests"
    for version in range(2, 13):
        source = root / f"phase4e-protocol-pilot-recovery.v{version}.json"
        validate_routed_manifest(source)

        tampered = tmp_path / source.name
        payload = json.loads(source.read_bytes())
        payload["cost_report_interval_usd"] = "11.00"
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            validate_routed_manifest(tampered)
