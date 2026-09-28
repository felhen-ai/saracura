"""Phase 5B closed acquisition descriptor and deterministic disposition.

This module is intentionally free of any optional import.  It runs in the
default environment and owns the offline-validatable contract: the closed
descriptor schema, the frozen thresholds, and the deterministic disposition
derivation.  The heavier orchestration (describe/prepare/serve/evaluate) lives
in sibling modules and lazily imports ``platformdirs``, ``psutil`` and
``huggingface_hub`` only when an explicit command is invoked.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DIGEST = r"^[0-9a-f]{64}$"
_FULL_GIT_REV = r"^[0-9a-f]{40}$"
_PUBLIC_PORT_RANGE = (49152, 65535)

CANDIDATE_ID = "kev-4b"
SCHEMA_VERSION = "phase5-kev4b-acquisition.v1"

FROZEN_RUNTIME = {
    "python": "3.12",
    "loopback_host": "127.0.0.1",
    "backend": "mlx",
    "dtype": "bf16",
}

ALLOWED_READINESS_EVIDENCE = (
    "pinned_license_reviewed_acquisition",
    "local_systems_report",
)


class DescriptorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


class LicenseBlock(DescriptorModel):
    # SPDX identifier or explicit review marker per reviewed artifact.
    source: str
    adapter: str
    base: str
    source_url: Annotated[str, Field(min_length=1)]
    adapter_url: Annotated[str, Field(min_length=1)]
    base_url: Annotated[str, Field(min_length=1)]


class FileLedgerEntry(DescriptorModel):
    path: str
    bytes: int
    sha256: Annotated[str, Field(pattern=_DIGEST)]
    role: Literal["source", "adapter", "base"]


class SourceBlock(DescriptorModel):
    repository: str
    revision: Annotated[str, Field(pattern=_FULL_GIT_REV)]
    python_constraint: str
    lock_sha256: Annotated[str, Field(pattern=_DIGEST)]
    files: tuple[FileLedgerEntry, ...]

    @model_validator(mode="after")
    def ordered_unique_paths(self) -> SourceBlock:
        paths = [entry.path for entry in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("source file ledger must be ordered with unique paths")
        return self


class SnapshotBlock(DescriptorModel):
    model_id: str
    revision: Annotated[str, Field(pattern=_FULL_GIT_REV)]
    files: tuple[FileLedgerEntry, ...]

    @model_validator(mode="after")
    def ordered_unique_paths(self) -> SnapshotBlock:
        paths = [entry.path for entry in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("snapshot file ledger must be ordered with unique paths")
        return self


class RuntimeBlock(DescriptorModel):
    python: Literal["3.12"]
    loopback_host: Literal["127.0.0.1"]
    port_range: tuple[Annotated[int, Field(ge=49152)], Annotated[int, Field(le=65535)]]
    backend: Literal["mlx"]
    dtype: Literal["bf16"]
    offline_environment_flags: tuple[str, ...]
    lock_digest: Annotated[str, Field(pattern=_DIGEST)]

    @model_validator(mode="after")
    def port_bounds(self) -> RuntimeBlock:
        if self.port_range[0] > self.port_range[1]:
            raise ValueError("port range lower bound exceeds upper bound")
        return self


class LimitsBlock(DescriptorModel):
    max_total_bytes: int
    max_source_file_count: int
    max_source_bytes: int
    min_free_disk_bytes: int
    min_physical_memory_bytes: int
    startup_timeout_seconds: int
    request_timeout_seconds: int


class ThresholdsBlock(DescriptorModel):
    max_latency_seconds: float
    p95_latency_seconds_q1: float
    p95_latency_seconds_q10: float
    p95_latency_seconds_q50: float
    max_repeat_probability_delta: float
    max_together_separate_probability_delta: float
    near_tie_margin: float
    max_peak_rss_gib: float
    max_peak_physical_footprint_gib: float
    preferred_swap_delta_gib: float
    degraded_swap_delta_gib: float
    max_swap_delta_gib: float
    cold_readiness_timeout_seconds: int

    @model_validator(mode="after")
    def swap_ladder_is_monotonic(self) -> ThresholdsBlock:
        if not (
            self.preferred_swap_delta_gib <= self.degraded_swap_delta_gib <= self.max_swap_delta_gib
        ):
            raise ValueError("swap delta ladder must be monotonic")
        return self


class AcquisitionDescriptor(DescriptorModel):
    schema_version: Literal["phase5-kev4b-acquisition.v1"]
    reviewed_at: str
    candidate_id: Literal["kev-4b"]
    licenses: LicenseBlock
    source: SourceBlock
    checkpoint: SnapshotBlock
    base_model: SnapshotBlock
    runtime: RuntimeBlock
    limits: LimitsBlock
    thresholds: ThresholdsBlock
    allowed_readiness_evidence: tuple[
        Literal["pinned_license_reviewed_acquisition"], Literal["local_systems_report"]
    ]


Disposition = Literal["continue", "conditional", "reject_local", "blocked_upstream"]


def derive_disposition(
    *,
    functional_pass: bool,
    identity_pass: bool,
    security_pass: bool,
    cold_readiness_seconds: float,
    latency_p95_pass: dict[int, bool],
    any_request_over_60s: bool,
    peak_rss_gib: float,
    peak_footprint_gib: float,
    swap_delta_gib: float,
    memory_pressure_critical: bool | None,
    invalid_outputs: int,
    unauthorized_accepted: int,
    oversize_rejected_correctly: bool,
    invalid_probe_rejected_correctly: bool,
    matrix_completed: bool,
    thresholds: ThresholdsBlock,
) -> Disposition:
    """Derive exactly one disposition from the frozen descriptor thresholds.

    The precedence is a fail-closed ladder.  ``reject_local`` wins over every
    capacity-band nuance; ``conditional`` only exists when every functional,
    identity and security threshold holds and only a soft latency / RSS /
    preferred-swap bound is exceeded without a hard capacity failure.
    """
    soft_latency_failure = any(
        not passed
        for question_count, passed in latency_p95_pass.items()
        if question_count in (1, 10, 50)
    )
    if (
        cold_readiness_seconds > thresholds.cold_readiness_timeout_seconds
        or any_request_over_60s
        or not functional_pass
        or not identity_pass
        or not security_pass
        or invalid_outputs != 0
        or unauthorized_accepted != 0
        or not oversize_rejected_correctly
        or not invalid_probe_rejected_correctly
        or not matrix_completed
        or swap_delta_gib > thresholds.max_swap_delta_gib
        or memory_pressure_critical is True
    ):
        return "reject_local"
    if (
        soft_latency_failure
        or peak_rss_gib > thresholds.max_peak_rss_gib
        or peak_footprint_gib > thresholds.max_peak_physical_footprint_gib
        or swap_delta_gib > thresholds.preferred_swap_delta_gib
    ):
        return "conditional"
    return "continue"
