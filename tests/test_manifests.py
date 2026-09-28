import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.validate_manifests import (
    validate_manifest,
    validate_phase5_candidate_manifest,
    validate_phase5_readiness_manifest,
    validate_routed_manifest,
)
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


def _phase5_manifest(name: str) -> Path:
    return Path(__file__).parents[1] / "benchmarks/manifests" / name


def _write_tampered_manifest(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase5_candidate_manifest_is_closed_and_routed() -> None:
    path = _phase5_manifest("phase5-open-model-candidates.v1.json")
    validate_phase5_candidate_manifest(path)
    validate_routed_manifest(path)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda payload: payload.__setitem__("reviewed_at", "2026-09-27T00:00:00Z"),
            "ISO calendar date",
        ),
        (
            lambda payload: payload["candidates"][0].__setitem__(
                "architecture_class", "unknown_architecture"
            ),
            "architecture_class",
        ),
        (
            lambda payload: payload["candidates"][1].__setitem__("revision", "main"),
            "revision",
        ),
        (
            lambda payload: payload["candidates"][1].__setitem__(
                "disposition", "historical_baseline"
            ),
            "disposition requires",
        ),
        (
            lambda payload: payload["candidates"][2].__setitem__(
                "allowed_claims", ["candidate_for_evaluation"]
            ),
            "allowed_claims",
        ),
    ],
)
def test_phase5_candidate_manifest_rejects_contract_tampering(
    tmp_path: Path, mutate: object, error: str
) -> None:
    payload = json.loads(_phase5_manifest("phase5-open-model-candidates.v1.json").read_bytes())
    mutate(payload)  # type: ignore[operator]
    path = tmp_path / "phase5-open-model-candidates.v1.json"
    _write_tampered_manifest(path, payload)
    with pytest.raises(ValueError, match=error):
        validate_routed_manifest(path)


def test_phase5_candidate_manifest_rejects_packaged_copy(tmp_path: Path) -> None:
    source = _phase5_manifest("phase5-open-model-candidates.v1.json")
    path = tmp_path / "src" / "saracura" / source.name
    path.parent.mkdir(parents=True)
    path.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="must not live under src/saracura"):
        validate_routed_manifest(path)


def test_phase5_readiness_manifest_is_closed_and_routed() -> None:
    path = _phase5_manifest("phase5-public-readiness.v1.json")
    validate_phase5_readiness_manifest(path)
    validate_routed_manifest(path)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda payload: payload.__setitem__("claim_vocabulary", []), "claim_vocabulary"),
        (
            lambda payload: payload["gates"][0].__setitem__("status", "met"),
            "status",
        ),
        (
            lambda payload: payload["gates"][1]["required_evidence"].pop(),
            "required_evidence",
        ),
        (
            lambda payload: payload["gates"][1]["allowed_claims"].append("production_ready"),
            "gate claims",
        ),
        (
            lambda payload: payload["gates"][0].__setitem__(
                "forbidden_claims", ["production_ready"]
            ),
            "gate claims",
        ),
    ],
)
def test_phase5_readiness_manifest_rejects_contract_tampering(
    tmp_path: Path, mutate: object, error: str
) -> None:
    payload = json.loads(_phase5_manifest("phase5-public-readiness.v1.json").read_bytes())
    mutate(payload)  # type: ignore[operator]
    path = tmp_path / "phase5-public-readiness.v1.json"
    _write_tampered_manifest(path, payload)
    with pytest.raises(ValueError, match=error):
        validate_routed_manifest(path)


def _managed_cuda_manifest(name: str) -> Path:
    return Path(__file__).parents[1] / "benchmarks" / "manifests" / name


def test_phase5_managed_cuda_runtime_manifest_is_closed_and_routed() -> None:
    path = _managed_cuda_manifest("phase5-kev4b-managed-cuda.v1.json")
    validate_routed_manifest(path)


def test_phase5_managed_cuda_systems_manifest_is_closed_and_routed() -> None:
    path = _managed_cuda_manifest("phase5-managed-cuda-systems.v1.json")
    validate_routed_manifest(path)


def test_phase5_managed_cuda_manifests_bind_acquisition_digest() -> None:
    root = Path(__file__).parents[1]
    acq_path = root / "benchmarks" / "manifests" / "phase5-kev4b-acquisition.v1.json"
    digest = hashlib.sha256(acq_path.read_bytes()).hexdigest()
    for name in ("phase5-kev4b-managed-cuda.v1.json", "phase5-managed-cuda-systems.v1.json"):
        manifest = json.loads(_managed_cuda_manifest(name).read_text())
        assert manifest["acquisition_descriptor_sha256"] == digest
