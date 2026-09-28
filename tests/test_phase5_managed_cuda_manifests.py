from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import cast

import pytest

from benchmarks.validate_manifests import validate_routed_manifest


def test_phase5_managed_cuda_manifest_is_routed() -> None:
    root = Path(__file__).parents[1]
    runtime_path = root / "benchmarks" / "manifests" / "phase5-kev4b-managed-cuda.v1.json"
    validate_routed_manifest(runtime_path)
    proto_path = root / "benchmarks" / "manifests" / "phase5-managed-cuda-systems.v1.json"
    validate_routed_manifest(proto_path)


def test_phase5_managed_cuda_manifests_bind_acquisition() -> None:
    root = Path(__file__).parents[1]
    acq_path = root / "benchmarks" / "manifests" / "phase5-kev4b-acquisition.v1.json"
    runtime_path = root / "benchmarks" / "manifests" / "phase5-kev4b-managed-cuda.v1.json"
    acq_digest = hashlib.sha256(acq_path.read_bytes()).hexdigest()
    runtime = json.loads(runtime_path.read_text())
    assert runtime["acquisition_descriptor_sha256"] == acq_digest


def test_phase5_managed_cuda_runtime_has_fixed_host() -> None:
    root = Path(__file__).parents[1]
    runtime_path = root / "benchmarks" / "manifests" / "phase5-kev4b-managed-cuda.v1.json"
    runtime = json.loads(runtime_path.read_text())
    assert runtime["runtime"]["host"] == "127.0.0.1"
    assert runtime["runtime"]["backend"] == "torch"
    assert runtime["runtime"]["dtype"] == "bf16"
    assert isinstance(runtime["runtime"]["direct_upstream_port"], int)
    assert runtime["runtime"]["direct_upstream_port"] > 0


def test_phase5_managed_cuda_protocol_has_workload_counts() -> None:
    root = Path(__file__).parents[1]
    proto_path = root / "benchmarks" / "manifests" / "phase5-managed-cuda-systems.v1.json"
    proto = json.loads(proto_path.read_text())
    assert proto["workload_counts"]["pt-BR"]["q1"] == 20
    assert proto["workload_counts"]["pt-BR"]["q10"] == 20
    assert proto["workload_counts"]["pt-BR"]["q50"] == 20
    assert proto["workload_counts"]["en"]["q1"] == 20
    assert proto["workload_counts"]["en"]["q50"] == 20


def test_phase5_managed_cuda_protocol_binds_exact_report_schema(tmp_path: Path) -> None:
    from benchmarks.validate_manifests import validate_phase5_managed_cuda_systems_protocol

    source = Path(__file__).parents[1] / "benchmarks/manifests/phase5-managed-cuda-systems.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["report_schema"] = "phase5b-managed-cuda-systems-report.v2"
    changed = tmp_path / "systems-protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="report_schema is not the frozen public report schema"):
        validate_phase5_managed_cuda_systems_protocol(changed)


def test_phase5_managed_cuda_protocol_probe_sizes() -> None:
    root = Path(__file__).parents[1]
    proto_path = root / "benchmarks" / "manifests" / "phase5-managed-cuda-systems.v1.json"
    proto = json.loads(proto_path.read_text())
    assert proto["probe_sizes"]["oversize_state_items"] == 100000
    assert proto["probe_sizes"]["valid_control_questions"] == 1
    assert proto["probe_sizes"]["oversize_branch_instructions"] == 100000


def test_phase5_managed_cuda_thresholds_match_spec() -> None:
    root = Path(__file__).parents[1]
    proto_path = root / "benchmarks" / "manifests" / "phase5-managed-cuda-systems.v1.json"
    proto = json.loads(proto_path.read_text())
    t = proto["thresholds"]
    assert t["q1_p95_ms"] == 5000
    assert t["q10_p95_ms"] == 15000
    assert t["q50_p95_ms"] == 45000
    assert t["max_request_ms"] == 60000
    assert t["peak_rss_gib"] == 22
    assert t["peak_gpu_memory_gib"] == 16
    assert t["device_min_ram_gib"] == 24
    assert t["candidate_swap_gib"] == 1


def test_phase5_v3_schema_accepts_only_nullable_result_bindings() -> None:
    from benchmarks.validate_manifests import (
        _PHASE5_OPEN_MODEL_CANDIDATES_V3_FIELDS,
        validate_phase5_candidate_manifest_v3,
    )

    root = Path(__file__).parents[1]
    payload = {
        "schema_version": "phase5-open-model-candidates.v3",
        "reviewed_at": "2026-09-28",
        "supersedes_manifest_sha256": hashlib.sha256(
            (root / "benchmarks/manifests/phase5-open-model-candidates.v2.json").read_bytes()
        ).hexdigest(),
        "allowed_dispositions": [
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
        ],
        "candidate_claim_vocabulary": [
            "historical_research_baseline",
            "candidate_for_evaluation",
            "external_product_control",
            "comparison_reference",
        ],
        "candidates": [
            {**candidate, "mitigation_markers": []}
            for candidate in json.loads(
                (root / "benchmarks/manifests/phase5-open-model-candidates.v2.json").read_text()
            )["candidates"]
        ],
        "systems_report_path": None,
        "systems_report_digest": None,
        "protocol_path": None,
        "protocol_digest": None,
        "readiness_manifest_path": None,
        "readiness_manifest_sha256": None,
    }
    assert set(payload) == _PHASE5_OPEN_MODEL_CANDIDATES_V3_FIELDS
    path = root / "benchmarks/manifests/v3-schema-only-test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_phase5_candidate_manifest_v3(path)
        schema_only_candidates = cast(list[dict[str, object]], payload["candidates"])
        kev = next(item for item in schema_only_candidates if item["id"] == "kev-4b")
        kev["disposition"] = "conditional"
        kev["allowed_claims"] = ["candidate_for_evaluation"]
        kev["mitigation_markers"] = ["state_size_guard_required"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="exact semantic copy of v2"):
            validate_phase5_candidate_manifest_v3(path)
        kev["disposition"] = "reviewed_acquisition"
        kev["allowed_claims"] = ["candidate_for_evaluation"]
        kev["mitigation_markers"] = []
        payload["systems_report_digest"] = "a" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="both null or both set"):
            validate_phase5_candidate_manifest_v3(path)
    finally:
        path.unlink(missing_ok=True)


def test_phase5_v3_binding_rejects_exact_byte_digest_mismatch(tmp_path: Path) -> None:
    from benchmarks.validate_manifests import _validate_phase5_v3_bindings

    payload = {
        "systems_report_path": "benchmarks/results/phase5b-kev4b-managed-cuda-systems.json",
        "systems_report_digest": "0" * 64,
        "protocol_path": "benchmarks/manifests/phase5-managed-cuda-systems.v1.json",
        "protocol_digest": "1" * 64,
        "readiness_manifest_path": "benchmarks/manifests/phase5-public-readiness.v1.json",
        "readiness_manifest_sha256": "2" * 64,
    }
    for relative in payload.values():
        if isinstance(relative, str) and relative.endswith(".json"):
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exact referenced bytes"):
        _validate_phase5_v3_bindings(payload, tmp_path / "manifest.json", tmp_path)


def test_direct_phase5b_report_validation_is_optional_until_report_exists(tmp_path: Path) -> None:
    from benchmarks.validate_manifests import validate_committed_phase5b_report

    validate_committed_phase5b_report(tmp_path)
    report = tmp_path / "benchmarks/results/phase5b-kev4b-managed-cuda-systems.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"schema_version":"unknown"}', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_committed_phase5b_report(tmp_path)


def test_direct_phase5b_failure_report_validates_without_v3(tmp_path: Path) -> None:
    from benchmarks.phase5_managed_cuda.models import PublicFailureReport
    from benchmarks.validate_manifests import validate_committed_phase5b_report

    report = PublicFailureReport(
        schema_version="phase5b-managed-cuda-failure-report.v1",
        report_kind="incomplete_execution",
        stage="direct_upstream_auth",
        classification="blocked_evidence",
        evidence_code="direct_upstream_server_error",
        operator_preempted=False,
        protected_workload_restored=True,
        disposition="blocked_evidence",
        scope_exclusions=(
            "quality",
            "calibration",
            "production",
            "automation",
            "runtime_registration",
            "readiness_gate_a",
        ),
    )
    path = tmp_path / "benchmarks/results/phase5b-kev4b-managed-cuda-systems.json"
    path.parent.mkdir(parents=True)
    path.write_text(report.model_dump_json(), encoding="utf-8")
    validate_committed_phase5b_report(tmp_path)


def _bound_v3_repository(
    tmp_path: Path, disposition: str = "conditional"
) -> tuple[Path, dict[str, object]]:
    import shutil

    from benchmarks.phase5_managed_cuda.cli import finalize
    from benchmarks.phase5_managed_cuda.models import OperatorReceipt

    cli_tests = Path(__file__).with_name("test_phase5_managed_cuda_cli.py")
    spec = importlib.util.spec_from_file_location(
        "phase5_managed_cuda_cli_test_fixtures", cli_tests
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Phase 5B report fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _provisional = module._provisional

    root = tmp_path
    protocol_source = (
        Path(__file__).parents[1] / "benchmarks/manifests/phase5-managed-cuda-systems.v1.json"
    )
    readiness_source = (
        Path(__file__).parents[1] / "benchmarks/manifests/phase5-public-readiness.v1.json"
    )
    v2_source = (
        Path(__file__).parents[1] / "benchmarks/manifests/phase5-open-model-candidates.v2.json"
    )
    protocol_path = root / "benchmarks/manifests/phase5-managed-cuda-systems.v1.json"
    readiness_path = root / "benchmarks/manifests/phase5-public-readiness.v1.json"
    report_path = root / "benchmarks/results/phase5b-kev4b-managed-cuda-systems.json"
    manifest_path = root / "benchmarks/manifests/phase5-open-model-candidates.v3.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(protocol_source, protocol_path)
    shutil.copyfile(readiness_source, readiness_path)
    protocol = json.loads(protocol_path.read_text())
    evidence = _provisional().model_dump()
    evidence.update(
        {
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "acquisition_descriptor_sha256": protocol["acquisition_descriptor_sha256"],
            "managed_runtime_sha256": protocol["managed_runtime_descriptor_sha256"],
            "fixture_sha256": protocol["fixture_digest"],
            "sensitive_values": ("private-v3-sensitive-93821",),
        }
    )
    if disposition == "reject_local":
        evidence["primary_resources"]["peak_rss_bytes"] = 23 * 1024**3
    private = root / "private.json"
    private.write_text(json.dumps(evidence), encoding="utf-8")
    operator = root / "operator.json"
    operator.write_text(
        OperatorReceipt(
            schema_version="phase5b-managed-cuda-operator-receipt.v1",
            preempted=disposition == "blocked_evidence",
            protected_workload_restored=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    finalize(
        private_input=private,
        operator_receipt=operator,
        public_output=report_path,
        host_facts=lambda: ("linux", "x86_64", "32-63"),
    )
    v2 = json.loads(v2_source.read_text())
    candidates = [{**candidate, "mitigation_markers": []} for candidate in v2["candidates"]]
    kev = next(item for item in candidates if item["id"] == "kev-4b")
    kev["disposition"] = disposition
    kev["allowed_claims"] = ["candidate_for_evaluation"] if disposition == "conditional" else []
    kev["mitigation_markers"] = (
        ["state_size_guard_required"] if disposition == "conditional" else []
    )
    payload: dict[str, object] = {
        "allowed_dispositions": [
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
        ],
        "candidate_claim_vocabulary": v2["candidate_claim_vocabulary"],
        "schema_version": "phase5-open-model-candidates.v3",
        "reviewed_at": "2026-09-28",
        "supersedes_manifest_sha256": hashlib.sha256(v2_source.read_bytes()).hexdigest(),
        "candidates": candidates,
        "systems_report_path": "benchmarks/results/phase5b-kev4b-managed-cuda-systems.json",
        "systems_report_digest": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "protocol_path": "benchmarks/manifests/phase5-managed-cuda-systems.v1.json",
        "protocol_digest": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "readiness_manifest_path": "benchmarks/manifests/phase5-public-readiness.v1.json",
        "readiness_manifest_sha256": hashlib.sha256(readiness_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, payload


@pytest.mark.parametrize("disposition", ["conditional", "reject_local"])
def test_v3_end_to_end_binds_valid_conditional_and_reject_local(
    tmp_path: Path, disposition: str
) -> None:
    from benchmarks.validate_manifests import validate_phase5_candidate_manifest_v3

    manifest, _payload = _bound_v3_repository(tmp_path, disposition)
    validate_phase5_candidate_manifest_v3(manifest, repository=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("disposition", "must equal"),
        ("missing_marker", "exact state-size guard"),
        ("extra_marker", "unknown mitigation"),
        ("missing_claim", "only candidate_for_evaluation"),
        ("extra_claim", "only candidate_for_evaluation"),
    ],
)
def test_v3_end_to_end_rejects_conditional_claim_marker_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from benchmarks.validate_manifests import validate_phase5_candidate_manifest_v3

    manifest, payload = _bound_v3_repository(tmp_path)
    candidates = cast(list[dict[str, object]], payload["candidates"])
    kev = next(item for item in candidates if item["id"] == "kev-4b")
    if mutation == "disposition":
        kev["disposition"] = "reject_local"
    elif mutation == "missing_marker":
        kev["mitigation_markers"] = []
    elif mutation == "extra_marker":
        kev["mitigation_markers"] = ["state_size_guard_required", "other"]
    elif mutation == "missing_claim":
        kev["allowed_claims"] = []
    else:
        kev["allowed_claims"] = ["candidate_for_evaluation", "comparison_reference"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_phase5_candidate_manifest_v3(manifest, repository=tmp_path)


@pytest.mark.parametrize(
    ("candidate_id", "field", "value", "message"),
    [
        ("kev-4b", "revision", "0" * 40, "Kev immutable identity"),
        ("jev", "display_name", "Changed external control", "non-Kev candidate jev"),
    ],
)
def test_v3_bound_promotion_rejects_predecessor_identity_drift(
    tmp_path: Path, candidate_id: str, field: str, value: str, message: str
) -> None:
    from benchmarks.validate_manifests import validate_phase5_candidate_manifest_v3

    manifest, payload = _bound_v3_repository(tmp_path, "conditional")
    candidates = cast(list[dict[str, object]], payload["candidates"])
    candidate = next(item for item in candidates if item["id"] == candidate_id)
    candidate[field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_phase5_candidate_manifest_v3(manifest, repository=tmp_path)


def test_v3_end_to_end_rejects_gate_a_drift_and_blocked_successor(tmp_path: Path) -> None:
    from benchmarks.validate_manifests import validate_phase5_candidate_manifest_v3

    manifest, payload = _bound_v3_repository(tmp_path)
    readiness = tmp_path / cast(str, payload["readiness_manifest_path"])
    readiness_payload = json.loads(readiness.read_text())
    readiness_payload["gates"][0]["status"] = "met"
    readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")
    payload["readiness_manifest_sha256"] = hashlib.sha256(readiness.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_phase5_candidate_manifest_v3(manifest, repository=tmp_path)

    blocked_root = tmp_path / "blocked"
    blocked_root.mkdir()
    blocked_manifest, _ = _bound_v3_repository(blocked_root, "blocked_evidence")
    with pytest.raises(ValueError, match="must not publish"):
        validate_phase5_candidate_manifest_v3(blocked_manifest, repository=blocked_root)
