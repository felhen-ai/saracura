"""Phase 5B managed CUDA two-step CLI."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from benchmarks.phase5_managed_cuda.disposition import derive_provisional_disposition
from benchmarks.phase5_managed_cuda.models import (
    FailureClassification,
    FailureCode,
    FailureStage,
    OperatorReceipt,
    PrivateFailureProvisional,
    PrivateProvisional,
    PublicFailureReport,
    PublicSystemsReport,
    RuntimeIdentityEvidence,
)
from benchmarks.phase5_managed_cuda.runner import (
    CandidateRejected,
    EvidenceBlocked,
    ResourceMonitor,
    execute_matrix,
    request_json,
    run_oversize_probes,
    run_supplemental_probes,
    validate_loopback_url,
    validate_runtime_identity,
)
from benchmarks.validate_manifests import (
    validate_phase5_managed_cuda_runtime,
    validate_phase5_managed_cuda_systems_protocol,
    validate_phase5b_acquisition_descriptor,
)


def _validate_pid(pid: int) -> None:
    if pid <= 0 or os.geteuid() == 0:
        raise ValueError("runner requires a positive PID and refuses root")
    try:
        os.kill(pid, 0)
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            raise ValueError("candidate PID is not owned by the invoking user")
    except (ProcessLookupError, FileNotFoundError, PermissionError) as error:
        raise ValueError("candidate PID identity is unavailable") from error


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benchmark_package_digest(package_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fixture_digest() -> str:
    from benchmarks.phase5_candidate.fixture import matrix

    return hashlib.sha256(json.dumps(matrix(), sort_keys=True, default=str).encode()).hexdigest()


def _atomic_create(path: Path, payload: bytes, mode: int) -> None:
    """Write and fsync a sibling file, then atomically link without clobbering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.chmod(path, mode)
    except FileExistsError as error:
        raise ValueError(f"output already exists, refusing to clobber: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def execute(
    *,
    gateway_url: str,
    direct_upstream_url: str,
    expected_checkpoint: str,
    pid: int,
    cold_load_evidence: str,
    acquisition_descriptor_path: Path,
    managed_runtime_path: Path,
    commit: str,
    archive_digest: str,
    private_output: Path,
    sampler_factory: Any = ResourceMonitor,
    request: Any = request_json,
    matrix_runner: Any = execute_matrix,
    supplemental_runner: Any = run_supplemental_probes,
    oversize_runner: Any = run_oversize_probes,
    identity_validator: Any = validate_runtime_identity,
) -> Path:
    """Run identity, 240 requests, supplemental probes, and both resource windows."""
    if not Path(expected_checkpoint).is_absolute():
        raise ValueError("expected checkpoint must be an absolute path")
    gateway_url = validate_loopback_url(gateway_url)
    direct_upstream_url = validate_loopback_url(direct_upstream_url)
    if direct_upstream_url != "http://127.0.0.1:8182":
        raise ValueError("direct upstream must use the pinned 127.0.0.1:8182 endpoint")
    repository_root = Path(__file__).resolve().parents[2]
    output = private_output.resolve()
    if output == repository_root or repository_root in output.parents:
        raise ValueError("private provisional output must be outside the repository")
    if output.exists():
        raise ValueError(f"output already exists, refusing to clobber: {output}")

    validate_phase5b_acquisition_descriptor(acquisition_descriptor_path)
    validate_phase5_managed_cuda_runtime(managed_runtime_path)
    protocol_path = (
        Path(__file__).resolve().parents[1] / "manifests/phase5-managed-cuda-systems.v1.json"
    )
    validate_phase5_managed_cuda_systems_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_bytes())
    acquisition_digest = _digest_file(acquisition_descriptor_path)
    runtime_digest = _digest_file(managed_runtime_path)
    if acquisition_digest != protocol["acquisition_descriptor_sha256"]:
        raise ValueError("acquisition descriptor bytes do not match the canonical systems protocol")
    if runtime_digest != protocol["managed_runtime_descriptor_sha256"]:
        raise ValueError(
            "managed-runtime descriptor bytes do not match the canonical systems protocol"
        )
    _validate_pid(pid)
    acquisition_data = json.loads(acquisition_descriptor_path.read_text(encoding="utf-8"))

    try:
        cold_seconds = float(cold_load_evidence)
    except (TypeError, ValueError) as error:
        raise ValueError("cold-load evidence must be operator-attested seconds") from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a full lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", archive_digest):
        raise ValueError("archive digest must be SHA-256")

    sensitive_values = tuple(
        value
        for value in (gateway_url, direct_upstream_url, expected_checkpoint, str(pid))
        if value
    )

    def persist_failure(stage: FailureStage, error: BaseException) -> None:
        classification: FailureClassification = (
            "reject_local" if isinstance(error, CandidateRejected) else "blocked_evidence"
        )
        if classification == "reject_local":
            code: FailureCode = "candidate_invalid_response"
        elif stage == "runtime_identity":
            code = "identity_invalid_or_changed"
        elif stage == "direct_upstream_auth":
            code = (
                "direct_upstream_server_error"
                if str(error) == "direct-upstream no-authentication probe returned server error"
                else "transport_or_sampling_failure"
            )
        elif stage == "oversize":
            code = "oversize_protocol_drift"
        elif stage in {
            "primary_monitor_readiness",
            "diagnostic_monitor_readiness",
            "resource_finalization",
        }:
            code = "preemption_or_identity_drift"
        elif isinstance(error, EvidenceBlocked):
            code = "transport_or_sampling_failure"
        else:
            code = "unexpected_execution_failure"
        failure = PrivateFailureProvisional(
            schema_version="phase5b-managed-cuda-failure-provisional.v1",
            saracura_commit=commit,
            archive_sha256=archive_digest,
            stage=stage,
            classification=classification,
            evidence_code=code,
            sensitive_values=sensitive_values,
        )
        _atomic_create(
            output,
            json.dumps(failure.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode(),
            0o600,
        )

    stage: FailureStage = "runtime_identity"
    try:
        runtime_identity = RuntimeIdentityEvidence.model_validate(
            identity_validator(
                pid=pid,
                expected_checkpoint=expected_checkpoint,
                host="127.0.0.1",
                direct_upstream_port=8182,
                gateway_url=gateway_url,
            )
        ).model_dump()
    except (EvidenceBlocked, ValidationError, OSError) as error:
        blocked = (
            error
            if isinstance(error, EvidenceBlocked)
            else EvidenceBlocked("validated runtime identity is incomplete or unavailable")
        )
        persist_failure(stage, blocked)
        raise blocked from error

    stage = "direct_upstream_auth"
    try:
        direct = request(
            f"{direct_upstream_url}/v1/systemone",
            {"model": "kev-latest", "state": {}, "questions": {}},
            timeout=10,
        )
        if direct.status >= 500:
            raise EvidenceBlocked("direct-upstream no-authentication probe returned server error")
        if direct.status != 401:
            raise CandidateRejected("unauthenticated direct-upstream request was accepted")
    except (EvidenceBlocked, CandidateRejected) as error:
        persist_failure(stage, error)
        raise

    primary_monitor = sampler_factory(pid, expected_identity=runtime_identity)
    stage = "primary_monitor_readiness"
    try:
        primary_monitor.start()
    except EvidenceBlocked as error:
        persist_failure(stage, error)
        raise
    primary_resources: dict[str, Any] | None = None
    diagnostic_resources: dict[str, Any] | None = None
    diagnostic_monitor: Any | None = None
    primary_stop_attempted = False
    diagnostic_started = False
    diagnostic_stop_attempted = False

    def stop_primary() -> dict[str, Any]:
        nonlocal primary_stop_attempted
        if primary_stop_attempted:
            raise EvidenceBlocked("primary resource monitor finalization was attempted twice")
        primary_stop_attempted = True
        return cast(dict[str, Any], primary_monitor.stop())

    def stop_diagnostic() -> dict[str, Any]:
        nonlocal diagnostic_stop_attempted
        if diagnostic_monitor is None or not diagnostic_started or diagnostic_stop_attempted:
            raise EvidenceBlocked("diagnostic resource monitor is not finalizable")
        diagnostic_stop_attempted = True
        return cast(dict[str, Any], diagnostic_monitor.stop())

    def finish_primary() -> None:
        nonlocal primary_resources, stage
        if primary_resources is None:
            previous_stage = stage
            stage = "resource_finalization"
            primary_resources = stop_primary()
            stage = previous_stage

    def diagnostic_request(call: Any) -> Any:
        nonlocal diagnostic_monitor, diagnostic_resources
        nonlocal diagnostic_started, stage
        stage = "diagnostic_monitor_readiness"
        diagnostic_monitor = sampler_factory(pid, expected_identity=runtime_identity)
        diagnostic_monitor.start()
        diagnostic_started = True
        stage = "oversize"
        try:
            return call()
        finally:
            previous_stage = stage
            stage = "resource_finalization"
            diagnostic_resources = stop_diagnostic()
            stage = previous_stage

    try:
        stage = "matrix"
        matrix_evidence = matrix_runner(gateway_url)
        stage = "supplemental"
        supplemental = supplemental_runner(gateway_url)
        stage = "oversize"
        oversize = oversize_runner(
            gateway_url,
            timeout=60,
            on_primary_complete=finish_primary,
            diagnostic_request=diagnostic_request,
        )
        stage = "resource_finalization"
        finish_primary()
    except (EvidenceBlocked, CandidateRejected) as error:
        closing_blocker: EvidenceBlocked | None = None
        if not primary_stop_attempted:
            try:
                stop_primary()
            except EvidenceBlocked as closing_error:
                closing_blocker = closing_error
            except BaseException as closing_error:
                closing_blocker = EvidenceBlocked("primary resource monitor finalization failed")
                closing_blocker.__cause__ = closing_error
        if diagnostic_started and not diagnostic_stop_attempted:
            try:
                stop_diagnostic()
            except EvidenceBlocked as closing_error:
                closing_blocker = closing_error
            except BaseException as closing_error:
                closing_blocker = EvidenceBlocked("diagnostic resource monitor finalization failed")
                closing_blocker.__cause__ = closing_error
        if closing_blocker is not None:
            stage = "resource_finalization"
            persist_failure(stage, closing_blocker)
            raise closing_blocker from error
        persist_failure(stage, error)
        raise
    if primary_resources is None or diagnostic_resources is None:
        missing_resources_error = EvidenceBlocked(
            "primary and diagnostic resource windows are both required"
        )
        persist_failure("resource_finalization", missing_resources_error)
        raise missing_resources_error

    try:
        provisional = PrivateProvisional.model_validate(
            {
                "schema_version": "phase5b-managed-cuda-provisional.v1",
                "saracura_commit": commit,
                "archive_sha256": archive_digest,
                "acquisition_descriptor_sha256": acquisition_digest,
                "managed_runtime_sha256": runtime_digest,
                "source_revision": acquisition_data["source"]["revision"],
                "checkpoint_revision": acquisition_data["checkpoint"]["revision"],
                "base_revision": acquisition_data["base_model"]["revision"],
                "base_model_id": acquisition_data["base_model"]["model_id"],
                "protocol_sha256": _digest_file(protocol_path),
                "fixture_sha256": _fixture_digest(),
                "uv_lock_sha256": _digest_file(repository_root / "uv.lock"),
                "benchmark_package_sha256": _benchmark_package_digest(Path(__file__).parent),
                "cold_load_seconds": cold_seconds,
                "direct_upstream_401": True,
                "runtime_identity": runtime_identity,
                "matrix": matrix_evidence,
                "supplemental": supplemental,
                "oversize": oversize,
                "primary_resources": primary_resources,
                "diagnostic_resources": diagnostic_resources,
                "invalid_outputs": 0,
                "unauthorized_accepted": 0,
                "sensitive_values": sensitive_values,
            }
        )
    except ValidationError as error:
        blocked = EvidenceBlocked("final private evidence is incomplete or malformed")
        persist_failure("resource_finalization", blocked)
        raise blocked from error
    _atomic_create(
        output,
        json.dumps(provisional.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode(),
        0o600,
    )
    return output


_SENSITIVE_PATTERN = re.compile(
    r"(?:https?://|\b(?:\d{1,3}\.){3}\d{1,3}\b|/(?:Users|private|Volumes|tmp|home)/|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\bBearer\s+\S+|"
    r"\b[A-Za-z0-9-]+\.(?:com|net|org|local|internal|home|io)\b)",
    re.I,
)


def _assert_public_sanitized(
    report: PublicSystemsReport | PublicFailureReport, exact_sensitive: tuple[str, ...]
) -> None:
    serialized = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    for value in exact_sensitive:
        if value and value in serialized:
            raise ValueError("public report contains an exact private value")
    if _SENSITIVE_PATTERN.search(serialized):
        raise ValueError(
            "public report contains a prohibited host, path, URL, or credential pattern"
        )


def _memory_bucket_gib() -> str:
    values = re.findall(
        r"^(?:MemTotal):\s+(\d+)\s+kB$", Path("/proc/meminfo").read_text(encoding="ascii"), re.M
    )
    if len(values) != 1:
        raise EvidenceBlocked("host memory capacity is unavailable")
    gib = int(values[0]) / (1024**2)
    return "16-31" if gib < 32 else "32-63" if gib < 64 else "64-127" if gib < 128 else "128+"


def _host_facts() -> tuple[str, str, str]:
    if platform.system() != "Linux":
        raise EvidenceBlocked("public host facts require the Linux evaluation host")
    return "linux", platform.machine(), _memory_bucket_gib()


def finalize(
    *,
    private_input: Path,
    operator_receipt: Path,
    public_output: Path,
    host_facts: Any = _host_facts,
) -> Path:
    """Validate private/operator inputs, derive disposition, sanitize, and publish."""
    if not private_input.is_file():
        raise ValueError("private provisional evidence is required")
    if not operator_receipt.is_file():
        raise ValueError("closed post-run operator receipt is required")
    if public_output.exists():
        raise ValueError(f"output already exists, refusing to clobber: {public_output}")
    operator = OperatorReceipt.model_validate_json(operator_receipt.read_text(encoding="utf-8"))
    private_payload = json.loads(private_input.read_text(encoding="utf-8"))
    if private_payload.get("schema_version") == "phase5b-managed-cuda-failure-provisional.v1":
        failure = PrivateFailureProvisional.model_validate(private_payload)
        failure_disposition: FailureClassification = (
            "blocked_evidence"
            if operator.preempted or not operator.protected_workload_restored
            else failure.classification
        )
        failure_report = PublicFailureReport(
            schema_version="phase5b-managed-cuda-failure-report.v1",
            report_kind="incomplete_execution",
            stage=failure.stage,
            classification=failure.classification,
            evidence_code=failure.evidence_code,
            operator_preempted=operator.preempted,
            protected_workload_restored=operator.protected_workload_restored,
            disposition=failure_disposition,
            scope_exclusions=(
                "quality",
                "calibration",
                "production",
                "automation",
                "runtime_registration",
                "readiness_gate_a",
            ),
        )
        _assert_public_sanitized(failure_report, failure.sensitive_values)
        _atomic_create(
            public_output,
            json.dumps(
                failure_report.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
            ).encode(),
            0o644,
        )
        return public_output
    provisional = PrivateProvisional.model_validate(private_payload)
    disposition = derive_provisional_disposition(provisional, operator)
    primary = provisional.primary_resources
    os_family, architecture, memory_bucket = host_facts()
    report = PublicSystemsReport.model_validate(
        {
            "schema_version": "phase5b-managed-cuda-systems-report.v1",
            "saracura_commit": provisional.saracura_commit,
            "acquisition_descriptor_sha256": provisional.acquisition_descriptor_sha256,
            "managed_runtime_descriptor_sha256": provisional.managed_runtime_sha256,
            "source_revision": provisional.source_revision,
            "checkpoint_revision": provisional.checkpoint_revision,
            "base_revision": provisional.base_revision,
            "base_model_id": provisional.base_model_id,
            "protocol_digest": provisional.protocol_sha256,
            "fixture_digest": provisional.fixture_sha256,
            "archive_digest": provisional.archive_sha256,
            "uv_lock_digest": provisional.uv_lock_sha256,
            "benchmark_package_digest": provisional.benchmark_package_sha256,
            "os_family": os_family,
            "architecture": architecture,
            "physical_memory_bucket_gib": memory_bucket,
            "vram_bucket_gib": "24-31"
            if primary.device_total_mib < 32768
            else "32-47"
            if primary.device_total_mib < 49152
            else "48+",
            "candidate_runtime": "torch",
            "candidate_dtype": "bfloat16",
            "cold_load_seconds": provisional.cold_load_seconds,
            "timing_cells": {
                key: {"p95_ms": cell.p95_ms, "max_ms": cell.max_ms, "n": cell.n}
                for key, cell in provisional.matrix.cells.items()
            },
            "aggregate_p95_ms": max(cell.p95_ms for cell in provisional.matrix.cells.values()),
            "aggregate_max_ms": max(cell.max_ms for cell in provisional.matrix.cells.values()),
            "resources": {
                "peak_rss_gib": primary.peak_rss_bytes / (1024**3),
                "peak_gpu_memory_gib": primary.gpu_peak_mib / 1024,
                "candidate_swap_delta_gib": primary.candidate_swap_delta_bytes / (1024**3),
                "host_swap_delta_gib": primary.host_swap_delta_bytes / (1024**3),
                "device_total_gib": primary.device_total_mib / 1024,
            },
            "diagnostic_resources": {
                "peak_rss_gib": provisional.diagnostic_resources.peak_rss_bytes / (1024**3),
                "peak_gpu_memory_gib": provisional.diagnostic_resources.gpu_peak_mib / 1024,
                "candidate_swap_delta_gib": (
                    provisional.diagnostic_resources.candidate_swap_delta_bytes / (1024**3)
                ),
                "host_swap_delta_gib": provisional.diagnostic_resources.host_swap_delta_bytes
                / (1024**3),
                "device_total_gib": provisional.diagnostic_resources.device_total_mib / 1024,
            },
            "statuses": {
                "runtime_identity_valid": all(provisional.runtime_identity.model_dump().values()),
                "direct_upstream_auth_required": provisional.direct_upstream_401,
                "invalid_contract_rejected": provisional.oversize.invalid_status == 422,
                "oversized_branch_rejected": provisional.oversize.branch_status == 422,
                "oversized_state_truncated": provisional.oversize.state_status == 200,
                "matrix_responses_valid": all(
                    cell.last_response_valid for cell in provisional.matrix.cells.values()
                ),
                "together_separate_choice_stable": (
                    provisional.supplemental.together_separate_choice_stable
                ),
                "operator_preempted": operator.preempted,
                "protected_workload_restored": operator.protected_workload_restored,
                "evidence_ambiguous": any(
                    evidence.host_swap_delta_bytes > 0 and evidence.candidate_swap_delta_bytes == 0
                    for evidence in (
                        provisional.primary_resources,
                        provisional.diagnostic_resources,
                    )
                ),
            },
            "direct_upstream_401": provisional.direct_upstream_401,
            "invalid_outputs": provisional.invalid_outputs,
            "unauthorized_accepted": provisional.unauthorized_accepted,
            "oversize_state_outcome": provisional.oversize.state_outcome,
            "reported_input_tokens": provisional.oversize.reported_input_tokens,
            "repeat_probability_delta": provisional.supplemental.repeat_probability_delta,
            "together_separate_probability_delta": (
                provisional.supplemental.together_separate_probability_delta
            ),
            "option_permutation_probability_delta": (
                provisional.supplemental.option_permutation_probability_delta
            ),
            "disposition": disposition,
            "state_size_guard_required": disposition == "conditional",
            "scope_exclusions": (
                "quality",
                "calibration",
                "production",
                "automation",
                "runtime_registration",
                "readiness_gate_a",
            ),
        }
    )
    _assert_public_sanitized(report, provisional.sensitive_values)
    _atomic_create(
        public_output,
        json.dumps(
            report.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
        ).encode(),
        0o644,
    )
    return public_output
