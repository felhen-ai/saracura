"""Closed Pydantic models for Phase 5B managed CUDA systems evidence."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_EVIDENCE_DIGEST = r"^[0-9a-f]{64}$"


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MatrixCell(ReportModel):
    n: Literal[20]
    p95_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    max_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    latencies_ms: tuple[Annotated[float, Field(ge=0, allow_inf_nan=False)], ...]
    last_response_valid: bool

    @model_validator(mode="after")
    def has_exact_sample_count(self) -> MatrixCell:
        if len(self.latencies_ms) != 20:
            raise ValueError("each matrix cell must retain exactly 20 timings")
        return self


class MatrixEvidence(ReportModel):
    measured_requests: Literal[240]
    warmups: Literal[36]
    cells: dict[str, MatrixCell]

    @model_validator(mode="after")
    def exact_cells(self) -> MatrixEvidence:
        expected = {
            f"{locale}/{workload}/{kind}"
            for locale in ("pt-BR", "en")
            for workload in ("q1", "q10", "q50")
            for kind in ("new_state", "cached_state")
        }
        if set(self.cells) != expected:
            raise ValueError("matrix must contain all twelve frozen cells")
        return self


class SupplementalEvidence(ReportModel):
    repeat_probability_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    together_separate_probability_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    option_permutation_probability_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    together_separate_choice_stable: bool


class GpuSample(ReportModel):
    started: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    ended: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    duration_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    compute_pids: tuple[Annotated[int, Field(gt=0)], ...]
    candidate_gpu_mib: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    device_total_mib: Annotated[float, Field(ge=24576, allow_inf_nan=False)]


class ResourceEvidence(ReportModel):
    peak_rss_bytes: Annotated[int, Field(gt=0)]
    peak_hwm_bytes: Annotated[int, Field(gt=0)]
    peak_vmswap_bytes: Annotated[int, Field(ge=0)]
    candidate_swap_delta_bytes: Annotated[int, Field(ge=0)]
    host_swap_delta_bytes: Annotated[int, Field(ge=0)]
    gpu_peak_mib: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    device_total_mib: Annotated[float, Field(ge=24576, allow_inf_nan=False)]
    proc_timestamps: tuple[Annotated[float, Field(ge=0, allow_inf_nan=False)], ...]
    nvidia_samples: tuple[GpuSample, ...]
    closing_proc_timestamp: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    closing_nvidia_sample: GpuSample
    identity_survived: Literal[True]

    @model_validator(mode="after")
    def complete_sampling(self) -> ResourceEvidence:
        if len(self.proc_timestamps) < 1 or len(self.nvidia_samples) < 1:
            raise ValueError("resource window lacks mandatory samples")
        if any(b - a > 1.0 for a, b in pairwise(self.proc_timestamps)):
            raise ValueError("procfs sampling gap exceeds one second")
        gpu_times = [sample.started for sample in self.nvidia_samples]
        if any(b - a > 3.0 for a, b in pairwise(gpu_times)):
            raise ValueError("nvidia sampling gap exceeds three seconds")
        if self.closing_proc_timestamp - self.proc_timestamps[-1] > 1.0:
            raise ValueError("closing procfs observation exceeds one-second cadence")
        if self.closing_nvidia_sample.started - gpu_times[-1] > 3.0:
            raise ValueError("closing nvidia observation exceeds three-second cadence")
        if self.closing_nvidia_sample.compute_pids != self.nvidia_samples[-1].compute_pids:
            raise ValueError("closing GPU attribution differs from the resource window")
        if self.closing_nvidia_sample.device_total_mib < 24576:
            raise ValueError("closing GPU capacity is below 24 GiB")
        return self


class OversizeEvidence(ReportModel):
    control_status: Literal[200]
    invalid_status: Literal[422]
    branch_status: Literal[422]
    state_status: Literal[200]
    state_outcome: Literal["accepted_with_pinned_source_truncation"]
    reported_input_tokens: Annotated[int, Field(ge=65536, lt=73728)]


class RuntimeIdentityEvidence(ReportModel):
    cmdline_valid: Literal[True]
    environment_valid: Literal[True]
    socket_valid: Literal[True]
    cpu_attribution_valid: Literal[True]
    pid: Annotated[int, Field(gt=0)]
    uid: Annotated[int, Field(ge=0)]
    starttime: Annotated[int, Field(gt=0)]
    socket_inode: Annotated[int, Field(gt=0)]


FailureStage = Literal[
    "runtime_identity",
    "direct_upstream_auth",
    "primary_monitor_readiness",
    "matrix",
    "supplemental",
    "oversize",
    "diagnostic_monitor_readiness",
    "resource_finalization",
]

FailureClassification = Literal["blocked_evidence", "reject_local"]
FailureCode = Literal[
    "identity_invalid_or_changed",
    "direct_upstream_server_error",
    "transport_or_sampling_failure",
    "missing_or_malformed_telemetry",
    "preemption_or_identity_drift",
    "oversize_protocol_drift",
    "candidate_invalid_response",
    "unexpected_execution_failure",
]


class PrivateFailureProvisional(ReportModel):
    """Closed minimal private evidence for a run that could not be completed."""

    schema_version: Literal["phase5b-managed-cuda-failure-provisional.v1"]
    saracura_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    archive_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    stage: FailureStage
    classification: FailureClassification
    evidence_code: FailureCode
    sensitive_values: tuple[str, ...]


class PrivateProvisional(ReportModel):
    schema_version: Literal["phase5b-managed-cuda-provisional.v1"]
    saracura_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    archive_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    acquisition_descriptor_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    managed_runtime_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    source_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    base_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    base_model_id: Literal["Qwen/Qwen3.5-4B-Base"]
    protocol_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    fixture_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    uv_lock_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    benchmark_package_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    cold_load_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    direct_upstream_401: Literal[True]
    runtime_identity: RuntimeIdentityEvidence
    matrix: MatrixEvidence
    supplemental: SupplementalEvidence
    oversize: OversizeEvidence
    primary_resources: ResourceEvidence
    diagnostic_resources: ResourceEvidence
    invalid_outputs: Annotated[int, Field(ge=0)]
    unauthorized_accepted: Annotated[int, Field(ge=0)]
    sensitive_values: tuple[str, ...]

    @model_validator(mode="after")
    def resource_pid_matches_validated_runtime(self) -> PrivateProvisional:
        for window in (self.primary_resources, self.diagnostic_resources):
            if any(
                sample.compute_pids != (self.runtime_identity.pid,)
                for sample in window.nvidia_samples
            ):
                raise ValueError("GPU samples are not attributed to the validated candidate PID")
        return self


class OperatorReceipt(ReportModel):
    schema_version: Literal["phase5b-managed-cuda-operator-receipt.v1"]
    preempted: bool
    protected_workload_restored: bool


class PublicCellTiming(ReportModel):
    p95_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    max_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    n: Literal[20]


class PublicResourceEvidence(ReportModel):
    peak_rss_gib: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    peak_gpu_memory_gib: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_swap_delta_gib: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    host_swap_delta_gib: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    device_total_gib: Annotated[float, Field(ge=24, allow_inf_nan=False)]


class PublicStatusEvidence(ReportModel):
    runtime_identity_valid: bool
    direct_upstream_auth_required: bool
    invalid_contract_rejected: bool
    oversized_branch_rejected: bool
    oversized_state_truncated: bool
    matrix_responses_valid: bool
    together_separate_choice_stable: bool
    operator_preempted: bool
    protected_workload_restored: bool
    evidence_ambiguous: bool


class PublicSystemsReport(ReportModel):
    schema_version: Literal["phase5b-managed-cuda-systems-report.v1"]
    saracura_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    acquisition_descriptor_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    managed_runtime_descriptor_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    source_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    base_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    base_model_id: Annotated[str, Field(min_length=3, max_length=120)]
    protocol_digest: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    fixture_digest: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    archive_digest: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    uv_lock_digest: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    benchmark_package_digest: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    os_family: Literal["linux"]
    architecture: Literal["x86_64", "aarch64"]
    physical_memory_bucket_gib: Literal["16-31", "32-63", "64-127", "128+"]
    vram_bucket_gib: Literal["24-31", "32-47", "48+"]
    candidate_runtime: Literal["torch"]
    candidate_dtype: Literal["bfloat16"]
    cold_load_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    timing_cells: dict[str, PublicCellTiming]
    aggregate_p95_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    aggregate_max_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    resources: PublicResourceEvidence
    diagnostic_resources: PublicResourceEvidence
    statuses: PublicStatusEvidence
    direct_upstream_401: Literal[True]
    invalid_outputs: Literal[0]
    unauthorized_accepted: Literal[0]
    oversize_state_outcome: Literal["accepted_with_pinned_source_truncation"]
    reported_input_tokens: Annotated[int, Field(ge=65536, lt=73728)]
    repeat_probability_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    together_separate_probability_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    option_permutation_probability_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    disposition: Literal["conditional", "reject_local", "blocked_evidence"]
    state_size_guard_required: bool
    scope_exclusions: tuple[
        Literal[
            "quality",
            "calibration",
            "production",
            "automation",
            "runtime_registration",
            "readiness_gate_a",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def safe_public_shape(self) -> PublicSystemsReport:
        expected = {
            f"{locale}/{workload}/{kind}"
            for locale in ("pt-BR", "en")
            for workload in ("q1", "q10", "q50")
            for kind in ("new_state", "cached_state")
        }
        if set(self.timing_cells) != expected:
            raise ValueError("public report must retain all twelve timing cells")
        if self.disposition == "conditional" and not self.state_size_guard_required:
            raise ValueError("conditional disposition requires state size guard")
        if self.disposition != "conditional" and self.state_size_guard_required:
            raise ValueError("state size guard marker is reserved for conditional")
        if self.disposition != self.derived_disposition():
            raise ValueError("public disposition does not match validated evidence")
        return self

    def derived_disposition(self) -> Literal["conditional", "reject_local", "blocked_evidence"]:
        status = self.statuses
        if (
            status.operator_preempted
            or not status.protected_workload_restored
            or status.evidence_ambiguous
            or not status.runtime_identity_valid
            or not status.direct_upstream_auth_required
        ):
            return "blocked_evidence"
        if (
            not status.invalid_contract_rejected
            or not status.oversized_branch_rejected
            or not status.matrix_responses_valid
            or not status.together_separate_choice_stable
        ):
            return "reject_local"
        if self.invalid_outputs or self.unauthorized_accepted:
            return "reject_local"
        thresholds = {"q1": 5000.0, "q10": 15000.0, "q50": 45000.0}
        if any(
            cell.p95_ms > thresholds[key.split("/")[1]] or cell.max_ms > 60000.0
            for key, cell in self.timing_cells.items()
        ):
            return "reject_local"
        primary = self.resources
        if (
            self.cold_load_seconds > 600.0
            or primary.peak_rss_gib > 22.0
            or primary.peak_gpu_memory_gib > 16.0
            or primary.candidate_swap_delta_gib > 1.0
        ):
            return "reject_local"
        if (
            self.repeat_probability_delta > 0.000001
            or self.together_separate_probability_delta > 0.04
        ):
            return "reject_local"
        if status.oversized_state_truncated:
            return "conditional"
        return "blocked_evidence"


class PublicFailureReport(ReportModel):
    """Minimal public variant for mandatory failures; contains no measurements."""

    schema_version: Literal["phase5b-managed-cuda-failure-report.v1"]
    report_kind: Literal["incomplete_execution"]
    stage: FailureStage
    classification: FailureClassification
    evidence_code: FailureCode
    operator_preempted: bool
    protected_workload_restored: bool
    disposition: Literal["blocked_evidence", "reject_local"]
    scope_exclusions: tuple[
        Literal[
            "quality",
            "calibration",
            "production",
            "automation",
            "runtime_registration",
            "readiness_gate_a",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def disposition_obeys_operator_precedence(self) -> PublicFailureReport:
        expected = (
            "blocked_evidence"
            if self.operator_preempted or not self.protected_workload_restored
            else self.classification
        )
        if self.disposition != expected:
            raise ValueError("failure report disposition does not match closed evidence")
        return self


class OversizeStateOutcome(ReportModel):
    kind: Literal["state"]
    control_status: int
    probe_status: int
    outcome: Literal["rejected_without_truncation", "accepted_with_pinned_source_truncation"]
    reported_input_tokens: int

    @model_validator(mode="after")
    def bounded_truncation(self) -> OversizeStateOutcome:
        if self.outcome == "accepted_with_pinned_source_truncation" and not (
            65536 <= self.reported_input_tokens < 73728
        ):
            raise ValueError("truncation tokens must be in [65536, 73728)")
        return self


class OversizeBranchOutcome(ReportModel):
    kind: Literal["branch"]
    control_status: Literal[200]
    probe_status: Literal[422]


class OversizeOutcome(ReportModel):
    branch: OversizeBranchOutcome
    state: OversizeStateOutcome
