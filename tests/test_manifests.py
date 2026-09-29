import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.validate_manifests import (
    validate_manifest,
    validate_phase5_candidate_manifest,
    validate_phase5_readiness_manifest,
    validate_routed_manifest,
    validate_v02_protocol_manifest,
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


def test_v02_protocol_manifest_is_closed_and_routed() -> None:
    path = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    validate_v02_protocol_manifest(path)
    validate_routed_manifest(path)


def test_v02_protocol_manifest_development_lane_binding() -> None:
    path = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        payload["development_lane"]["bound_manifest"]
        == "benchmarks/manifests/phase5d-ptbr-native.v1.json"
    )
    assert (
        payload["development_lane"]["bound_manifest_sha256"]
        == "af983ce181018a2f6c5abec351ce1b78e875764616875aed3f59aa416a499ff6"
    )
    assert payload["development_lane"]["presented_as_held_out"] is False


def test_v02_protocol_manifest_is_immutable(tmp_path: Path) -> None:
    source = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["extra_field"] = "tamper"
    path = tmp_path / "v02-model-evaluation-protocol.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="v02 protocol manifest"):
        validate_v02_protocol_manifest(path)
    with pytest.raises(ValueError, match="v02 protocol manifest"):
        validate_routed_manifest(path)


def test_v02_protocol_manifest_controls_are_frozen(tmp_path: Path) -> None:
    source = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["controls"] = [
        {"id": "laya-multilingual", "revision": "0" * 64},
        {"id": "julia-1-cyclic-mean", "revision": "0" * 64},
        {"id": "kev-4b", "revision": "0" * 64},
        {"id": "lexical-baseline", "revision": "0" * 64},
        {"id": "unknown-control", "revision": "0" * 64},
    ]
    path = tmp_path / "v02-model-evaluation-protocol.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown control"):
        validate_v02_protocol_manifest(path)


def test_v02_protocol_manifest_kev_rendering_contract_frozen(tmp_path: Path) -> None:
    source = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["kev_rendering_contract"]["max_rendered_input_tokens"] = 1024
    path = tmp_path / "v02-model-evaluation-protocol.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="512"):
        validate_v02_protocol_manifest(path)


def test_v02_protocol_manifest_prohibits_calibration_and_automation(tmp_path: Path) -> None:
    source = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["prohibited_claims"] = [
        "automation_authorized",
        "general_superiority",
    ]
    path = tmp_path / "v02-model-evaluation-protocol.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration_metrics"):
        validate_v02_protocol_manifest(path)


def test_v02_protocol_manifest_presents_dev_lane_as_development_only(tmp_path: Path) -> None:
    source = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["development_lane"]["presented_as_held_out"] = True
    path = tmp_path / "v02-model-evaluation-protocol.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="held out"):
        validate_v02_protocol_manifest(path)


def test_v02_protocol_manifest_rejects_nested_contract_drift(tmp_path: Path) -> None:
    source = _phase5_manifest("v02-model-evaluation-protocol.v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["held_out_payload"]["permanently_ineligible_for"].remove("calibration_fitting")
    path = tmp_path / "v02-model-evaluation-protocol.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest bytes drifted"):
        validate_v02_protocol_manifest(path)
